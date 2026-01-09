#!/usr/bin/env python
"""Energy-loss training entry point.

Combines Hydra/YAML config loading from train.py with the energy-loss
specific setup from train_energy_loss.py.

How to run (example):
    python scripts/train/train_energy.py scripts/train/configs/structure.yaml \
        model.training_args.use_energy_loss=true \
        freeze_trunk=true \
        freeze_confidence=true

The YAML file already contains:
  • data.datasets[*].target_dir / msa_dir (ground-truth references)
  • pretrained checkpoint path
  • model architecture & training_args
  • freeze_trunk / freeze_confidence / train_denoiser_only flags

This script will:
  1. Load config via Hydra
  2. Build the DataModule (BoltzTrainingDataModule) with ground-truth data paths
  3. Optionally load pretrained weights
  4. Freeze trunk / confidence when requested
  5. Optionally re-initialise denoiser weights (Kaiming) when init_denoiser_randomly=true
  6. Train with BoltzEnergyLoss when model.training_args.use_energy_loss=true
"""
from __future__ import annotations

import math
import os
import random
import string
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn.init as init
from omegaconf import OmegaConf, listconfig
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

import wandb as wandb_module

from boltz.data.module.training import BoltzTrainingDataModule, DataConfig
from boltz.model.utils.components import freeze_components, infer_component

# Optional: import alignment for energy loss
try:
    from boltz.utils.alignment import weighted_rigid_align_centered
except ImportError:
    weighted_rigid_align_centered = None


# ---------------------------------------------------------------------------
# Config dataclass (mirrors train.py but adds energy-loss relevant fields)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training configuration (YAML → dataclass).

    Attributes
    ----------
    data : DataConfig
        Data paths, filters, tokenizer, featurizer, etc.
    model : LightningModule
        Instantiated Boltz1 model.
    output : str
        Directory for checkpoints / logs.
    trainer : Optional[dict]
        PyTorch-Lightning Trainer kwargs.
    resume : Optional[str]
        Path to resume training from.
    pretrained : Optional[str]
        Path to pretrained checkpoint (loaded before training).
    wandb : Optional[dict]
        W&B logger config.
    disable_checkpoint : bool
        If True, skip saving checkpoints.
    matmul_precision : Optional[str]
        Float32 matmul precision.
    find_unused_parameters : bool
        DDP flag.
    save_top_k : int
        Keep top-k checkpoints by val/lddt.
    validation_only : bool
        Run validation loop only (no training).
    debug : bool
        Debugging mode (single device, no W&B).
    strict_loading : bool
        Fail if checkpoint keys mismatch.
    load_confidence_from_trunk : bool
        Copy trunk weights → confidence module before training.
    freeze_trunk : bool
        Freeze trunk parameters.
    freeze_confidence : bool
        Freeze confidence head parameters.
    train_denoiser_only : bool
        Convenience: freeze trunk+confidence, zero non-diffusion losses.
    init_denoiser_randomly : bool
        Re-initialise denoiser (structure_module) weights (Kaiming).
    """

    data: DataConfig
    model: LightningModule
    output: str
    trainer: Optional[dict] = None
    resume: Optional[str] = None
    pretrained: Optional[str] = None
    wandb: Optional[dict] = None
    disable_checkpoint: bool = False
    matmul_precision: Optional[str] = None
    find_unused_parameters: bool = False
    save_top_k: int = 1
    validation_only: bool = False
    debug: bool = False
    strict_loading: bool = True
    load_confidence_from_trunk: bool = False
    freeze_trunk: bool = False
    freeze_confidence: bool = False
    train_denoiser_only: bool = False
    init_denoiser_randomly: bool = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _kaiming_init_param(param: torch.nn.Parameter, scale: float = 0.1) -> None:
    """Re-initialise a parameter with scaled Xavier uniform for stability.
    
    Use smaller scale than default to prevent NaN gradients in early training.
    """
    if param.dim() >= 2:
        # Use xavier with small scale for numerical stability
        init.xavier_uniform_(param)
        param.data *= scale  # Scale down to prevent extreme activations
    else:
        init.zeros_(param)


class FullCheckpointCallback(Callback):
    """Callback to verify checkpoints include all model weights (frozen + trained).
    
    PyTorch Lightning saves all parameters by default, but this callback
    logs verification info and ensures the checkpoint is complete.
    """
    
    def __init__(self, component_classifier_fn=None):
        super().__init__()
        self.component_classifier_fn = component_classifier_fn
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Called when saving a checkpoint - verify all weights are included."""
        state_dict = checkpoint.get("state_dict", {})
        
        # Count parameters by component
        component_counts = {}
        total_params = 0
        
        for name, param in state_dict.items():
            total_params += param.numel()
            
            if self.component_classifier_fn:
                component = self.component_classifier_fn(name)
            else:
                # Simple classification based on prefix
                if name.startswith("structure_module"):
                    component = "denoiser"
                elif name.startswith("trunk") or name.startswith("pairformer"):
                    component = "trunk"
                elif name.startswith("confidence"):
                    component = "confidence"
                elif name.startswith("msa"):
                    component = "msa"
                else:
                    component = "other"
            
            if component not in component_counts:
                component_counts[component] = {"keys": 0, "params": 0}
            component_counts[component]["keys"] += 1
            component_counts[component]["params"] += param.numel()
        
        print(f"\n{'='*60}")
        print(f"Checkpoint saved with {len(state_dict)} keys, {total_params:,} total parameters")
        print(f"Components in checkpoint:")
        for comp, info in sorted(component_counts.items()):
            print(f"  - {comp}: {info['keys']} keys, {info['params']:,} params")
        print(f"{'='*60}\n")
    
    def on_fit_start(self, trainer, pl_module):
        """Log initial model state before training."""
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in pl_module.parameters() if not p.requires_grad)
        total = trainable + frozen
        
        print(f"\n{'='*60}")
        print(f"Model parameters:")
        print(f"  - Trainable (denoiser): {trainable:,}")
        print(f"  - Frozen (trunk/confidence): {frozen:,}")
        print(f"  - Total: {total:,}")
        print(f"  - Checkpoints will include ALL {total:,} parameters")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(raw_config_path: str, cli_overrides: list[str]) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run training with energy-loss support.

    Parameters
    ----------
    raw_config_path : str
        Path to YAML config file.
    cli_overrides : list[str]
        Dot-list overrides from command line.
    """
    # Register custom resolver for timestamps
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt))

    # Load YAML config
    raw_config = OmegaConf.load(raw_config_path)

    # Merge CLI overrides
    cli_config = OmegaConf.from_dotlist(cli_overrides)
    raw_config = OmegaConf.merge(raw_config, cli_config)

    # Instantiate via Hydra
    cfg = hydra.utils.instantiate(raw_config)
    cfg = TrainConfig(**cfg)

    # Matmul precision
    if cfg.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)

    # Trainer dict
    trainer_kwargs = cfg.trainer if cfg.trainer else {}

    # Debug mode adjustments
    devices = trainer_kwargs.get("devices", 1)
    wandb_cfg = cfg.wandb
    if cfg.debug:
        if isinstance(devices, int):
            devices = 1
        elif isinstance(devices, (list, listconfig.ListConfig)):
            devices = [devices[0]]
        trainer_kwargs["devices"] = devices
        cfg.data.num_workers = 0
        wandb_cfg = None

    # Build DataModule (contains ground-truth paths from YAML)
    data_config = DataConfig(**cfg.data)
    data_module = BoltzTrainingDataModule(data_config)

    # Model
    model_module = cfg.model

    # -------------------------------------------------------------------------
    # Load pretrained weights (optional)
    # -------------------------------------------------------------------------
    if cfg.pretrained and not cfg.resume:
        file_path = cfg.pretrained

        # Optionally copy trunk → confidence
        if cfg.load_confidence_from_trunk:
            checkpoint = torch.load(cfg.pretrained, map_location="cpu", weights_only=False)
            new_state_dict = {}
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("structure_module") and not key.startswith("distogram_module"):
                    new_state_dict["confidence_module." + key] = value
            new_state_dict.update(checkpoint["state_dict"])
            checkpoint["state_dict"] = new_state_dict

            rand_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
            file_path = os.path.join(os.path.dirname(cfg.pretrained), rand_suffix + ".ckpt")
            print(f"Saving modified checkpoint (trunk → confidence) to {file_path}")
            torch.save(checkpoint, file_path)

        print(f"Loading pretrained weights from {file_path}")

        if cfg.init_denoiser_randomly:
            # Load checkpoint, filter out denoiser weights
            checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
            filtered_state_dict = {
                k: v for k, v in checkpoint["state_dict"].items()
                if infer_component(k) != "denoiser"
            }
            # Load model architecture + non-denoiser weights
            model_module = type(model_module).load_from_checkpoint(
                file_path, map_location="cpu", strict=False, **(model_module.hparams)
            )
            model_module.load_state_dict(filtered_state_dict, strict=False)
            print("Loaded trunk/confidence weights; denoiser will be randomly initialised.")

            # Re-init denoiser params (Kaiming)
            reinit_count = 0
            for name, param in model_module.named_parameters():
                if infer_component(name) == "denoiser":
                    _kaiming_init_param(param)
                    reinit_count += 1
            print(f"Re-initialised {reinit_count} denoiser parameters (Kaiming)")
        else:
            model_module = type(model_module).load_from_checkpoint(
                file_path, map_location="cpu", strict=False, **(model_module.hparams)
            )

        # Cleanup temp checkpoint
        if cfg.load_confidence_from_trunk:
            os.remove(file_path)

    # -------------------------------------------------------------------------
    # Freeze components
    # -------------------------------------------------------------------------
    components_to_freeze = set()
    if cfg.freeze_trunk or cfg.train_denoiser_only:
        components_to_freeze.add("trunk")
    if cfg.freeze_confidence or cfg.train_denoiser_only:
        components_to_freeze.add("confidence")
    if components_to_freeze:
        freeze_components(model_module, components_to_freeze)
        print(f"Frozen components: {components_to_freeze}")

    # Zero non-diffusion losses when training denoiser only
    if cfg.train_denoiser_only:
        for attr in ("confidence_loss_weight", "distogram_loss_weight", "bfactor_loss_weight"):
            if hasattr(model_module.training_args, attr):
                setattr(model_module.training_args, attr, 0.0)

    # -------------------------------------------------------------------------
    # Energy loss is now configured in __init__ via training_args
    # The alignment function is automatically set to weighted_rigid_align_centered
    # when use_energy_loss=True (see boltz1.py __init__)
    # -------------------------------------------------------------------------
    use_energy = getattr(model_module.training_args, "use_energy_loss", False)
    if use_energy:
        print(f"Energy loss is ENABLED")
        print(f"  - energy_loss module: {model_module.energy_loss}")
        if model_module.energy_loss is not None:
            print(f"  - align_fn: {model_module.energy_loss.align_fn}")
            print(f"  - use_contra: {model_module.energy_loss.use_contra}")
            print(f"  - loss_kwargs: {model_module.energy_loss.loss_kwargs}")

    # -------------------------------------------------------------------------
    # Callbacks & loggers
    # -------------------------------------------------------------------------
    callbacks = []
    
    # Add checkpoint verification callback (ensures all weights saved)
    callbacks.append(FullCheckpointCallback(component_classifier_fn=infer_component))
    
    if not cfg.disable_checkpoint:
        mc = ModelCheckpoint(
            monitor="val/lddt",
            save_top_k=cfg.save_top_k,
            save_last=True,
            mode="max",
            every_n_epochs=1,
        )
        callbacks.append(mc)

    loggers = []
    if wandb_cfg:
        wdb_logger = WandbLogger(
            name=wandb_cfg["name"],
            group=wandb_cfg["name"],
            save_dir=cfg.output,
            project=wandb_cfg["project"],
            entity=wandb_cfg["entity"],
            log_model=False,
            settings=wandb_module.Settings(start_method="fork"),
        )
        loggers.append(wdb_logger)

        @rank_zero_only
        def save_config_to_wandb() -> None:
            config_out = Path(wdb_logger.experiment.dir) / "run.yaml"
            with open(config_out, "w") as f:
                OmegaConf.save(raw_config, f)
            wdb_logger.experiment.save(str(config_out))

        save_config_to_wandb()
        
        # Log energy loss hyperparameters to W&B config
        @rank_zero_only
        def log_energy_config() -> None:
            if use_energy and model_module.energy_loss is not None:
                energy_hparams = {
                    "energy_loss_enabled": True,
                    "energy_use_contra": getattr(model_module.energy_loss, "use_contra", False),
                    "energy_t_gen": getattr(model_module.training_args, "t_gen", 1.0),
                    "energy_scale_override": getattr(model_module.training_args, "energy_scale_override", None),
                    "train_denoiser_only": cfg.train_denoiser_only,
                    "freeze_trunk": cfg.freeze_trunk,
                    "freeze_confidence": cfg.freeze_confidence,
                    "init_denoiser_randomly": getattr(cfg, "init_denoiser_randomly", False),
                }
                # Add loss_kwargs
                loss_kwargs = getattr(model_module.energy_loss, "loss_kwargs", {})
                for k, v in loss_kwargs.items():
                    energy_hparams[f"energy_{k}"] = str(v) if isinstance(v, list) else v
                
                # Update W&B config
                wdb_logger.experiment.config.update(energy_hparams, allow_val_change=True)
        
        log_energy_config()

    # -------------------------------------------------------------------------
    # Strategy
    # -------------------------------------------------------------------------
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, (list, listconfig.ListConfig)) and len(devices) > 1
    ):
        strategy = DDPStrategy(find_unused_parameters=cfg.find_unused_parameters)

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------
    trainer = pl.Trainer(
        default_root_dir=cfg.output,
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=not cfg.disable_checkpoint,
        reload_dataloaders_every_n_epochs=1,
        **trainer_kwargs,
    )

    if not cfg.strict_loading:
        model_module.strict_loading = False

    # -------------------------------------------------------------------------
    # Increase LayerNorm epsilon for numerical stability
    # This helps prevent NaN gradients from random denoiser initialization
    # -------------------------------------------------------------------------
    def increase_layernorm_eps(module, eps=1e-4):
        """Recursively increase epsilon in LayerNorm layers."""
        count = 0
        for name, child in module.named_modules():
            if isinstance(child, nn.LayerNorm):
                old_eps = child.eps
                child.eps = max(child.eps, eps)
                if child.eps != old_eps:
                    count += 1
        return count
    
    # Only apply to structure_module (denoiser) to maintain frozen parts unchanged
    if hasattr(model_module, 'structure_module'):
        modified = increase_layernorm_eps(model_module.structure_module, eps=1e-4)
        print(f"[INFO] Increased LayerNorm eps to 1e-4 in {modified} layers of structure_module")

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------
    if cfg.validation_only:
        trainer.validate(model_module, datamodule=data_module, ckpt_path=cfg.resume)
    else:
        trainer.fit(model_module, datamodule=data_module, ckpt_path=cfg.resume)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_energy.py <config.yaml> [overrides...]")
        sys.exit(1)
    config_path = sys.argv[1]
    overrides = sys.argv[2:]
    train(config_path, overrides)

