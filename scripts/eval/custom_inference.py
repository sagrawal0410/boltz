#!/usr/bin/env python3
"""
Custom inference script for running predictions on validation set.

This script:
1. Loads validation IDs from a text file
2. Filters the manifest to only include validation records
3. Loads checkpoint and sets up model
4. Sets up inference data module
5. Runs predictions
6. Generates predictions in CIF format for evaluation

Usage:
    python scripts/eval/custom_inference.py \
        --validation-ids validation_ids.txt \
        --target-dir /data/scratch-oc40/shaurya10/rcsb_processed_targets \
        --msa-dir /data/scratch-oc40/shaurya10/rcsb_processed_msa \
        --checkpoint /path/to/checkpoint.ckpt \
        --output-dir ./predictions
"""

import argparse
import os
import platform
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Set

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

# Add parent directory to path to import boltz modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.types import Manifest, Record
from boltz.data.write.writer import BoltzWriter
from boltz.main import (
    BoltzDiffusionParams,
    BoltzProcessedInput,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgs,
)
from boltz.model.models.boltz1 import Boltz1


def parse_validation_ids(ids_file: Path) -> Set[str]:
    """Parse validation IDs from a text file.
    
    Expected format: one ID per line (lowercase PDB IDs), e.g.:
        8cgl
        8x7x
        8agg
        7ynx
    
    Parameters
    ----------
    ids_file : Path
        Path to validation IDs file
    
    Returns
    -------
    Set[str]
        Set of validation IDs (lowercase)
    """
    if not ids_file.exists():
        raise FileNotFoundError(f"Validation IDs file not found: {ids_file}")
    
    with ids_file.open("r") as f:
        ids = {line.strip().lower() for line in f if line.strip()}
    
    print(f"Loaded {len(ids)} validation IDs from {ids_file}")
    return ids


def filter_manifest_by_ids(manifest: Manifest, validation_ids: Set[str]) -> Manifest:
    """Filter manifest to only include records matching validation IDs.
    
    Parameters
    ----------
    manifest : Manifest
        The full manifest to filter
    validation_ids : Set[str]
        Set of validation IDs to keep (lowercase)
    
    Returns
    -------
    Manifest
        Filtered manifest containing only validation records
    """
    filtered_records = []
    matched_ids = set()
    
    for record in manifest.records:
        # Compare record ID (case-insensitive)
        record_id_lower = record.id.lower()
        
        # Check if record ID matches any validation ID
        # Handle both exact matches and cases where record ID might have additional info
        if record_id_lower in validation_ids:
            filtered_records.append(record)
            matched_ids.add(record_id_lower)
        else:
            # Also check if validation ID is a substring of record ID
            # (e.g., if record ID is "8cgl_chainA" and validation ID is "8cgl")
            for val_id in validation_ids:
                if record_id_lower.startswith(val_id) or val_id in record_id_lower:
                    filtered_records.append(record)
                    matched_ids.add(record_id_lower)
                    break
    
    # Report matching statistics
    print(f"\nManifest filtering results:")
    print(f"  Total records in manifest: {len(manifest.records)}")
    print(f"  Validation IDs provided: {len(validation_ids)}")
    print(f"  Records matched: {len(filtered_records)}")
    
    # Check for unmatched validation IDs
    unmatched = validation_ids - matched_ids
    if unmatched:
        print(f"\nWarning: {len(unmatched)} validation IDs not found in manifest:")
        for unmatched_id in sorted(list(unmatched))[:10]:  # Show first 10
            print(f"    - {unmatched_id}")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched) - 10} more")
    
    if len(filtered_records) == 0:
        print("\nERROR: No records matched! Please check:")
        print("  1. Validation IDs format matches manifest record IDs")
        print("  2. Manifest file is correct")
        print("\nSample record IDs from manifest:")
        for i, record in enumerate(manifest.records[:5]):
            print(f"    {i+1}. {record.id}")
        raise ValueError("No matching records found in manifest")
    
    return Manifest(filtered_records)


def main():
    parser = argparse.ArgumentParser(
        description="Custom inference script for validation set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--validation-ids",
        type=Path,
        required=True,
        help="Path to validation IDs file (one ID per line, lowercase PDB IDs)"
    )
    
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Path to processed targets directory (contains manifest.json)"
    )
    
    parser.add_argument(
        "--msa-dir",
        type=Path,
        required=True,
        help="Path to processed MSA directory"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.ckpt file)"
    )
    
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Path to base checkpoint (boltz1_conf.ckpt) for loading missing weights (confidence module, trunk, etc.)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./predictions"),
        help="Output directory for predictions (default: ./predictions)"
    )
    
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use (default: 1)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader workers (default: 2)"
    )
    
    parser.add_argument(
        "--recycling-steps",
        type=int,
        default=3,
        help="Number of recycling steps (default: 3, matching validation config)"
    )
    
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=200,
        help="Number of sampling steps (default: 200)"
    )
    
    parser.add_argument(
        "--diffusion-samples",
        type=int,
        default=5,
        help="Number of diffusion samples (default: 5)"
    )
    
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Custom path to manifest.json (default: {target_dir}/manifest.json)"
    )
    
    parser.add_argument(
        "--no-kernels",
        action="store_true",
        help="Disable optimized CUDA kernels (use if you get CUDA library errors)"
    )
    
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use EMA weights from checkpoint (recommended for best performance)"
    )
    
    parser.add_argument(
        "--symmetry-correction",
        action="store_true",
        default=True,
        help="Enable symmetry correction (default: True, matching validation config)"
    )
    
    args = parser.parse_args()
    
    # Step 1: Load validation IDs
    print("=" * 60)
    print("Step 1: Loading validation IDs")
    print("=" * 60)
    validation_ids = parse_validation_ids(args.validation_ids)
    print(f"Sample IDs: {sorted(list(validation_ids))[:5]}")
    
    # Step 2: Load manifest
    print("\n" + "=" * 60)
    print("Step 2: Loading manifest")
    print("=" * 60)
    if args.manifest_path is not None:
        manifest_path = args.manifest_path
    else:
        manifest_path = args.target_dir / "manifest.json"
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    print(f"Loading manifest from: {manifest_path}")
    manifest = Manifest.load(manifest_path)
    print(f"Loaded {len(manifest.records)} records from manifest")
    
    # Step 3: Filter manifest
    print("\n" + "=" * 60)
    print("Step 3: Filtering manifest by validation IDs")
    print("=" * 60)
    filtered_manifest = filter_manifest_by_ids(manifest, validation_ids)
    
    print(f"\nFiltered manifest contains {len(filtered_manifest.records)} records")
    print("\nSample filtered record IDs:")
    for i, record in enumerate(filtered_manifest.records[:10]):
        print(f"    {i+1}. {record.id}")
    if len(filtered_manifest.records) > 10:
        print(f"    ... and {len(filtered_manifest.records) - 10} more")
    
    # Step 4: Set up processed input structure
    print("\n" + "=" * 60)
    print("Step 4: Setting up processed input")
    print("=" * 60)
    
    # Check if structures subdirectory exists, otherwise use target_dir directly
    structures_dir = args.target_dir / "structures"
    if structures_dir.exists():
        targets_dir = structures_dir
    else:
        targets_dir = args.target_dir
    
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=targets_dir,
        msa_dir=args.msa_dir,
        constraints_dir=None,  # No constraints for validation
        template_dir=None,  # No templates for validation
        extra_mols_dir=None,  # No extra mols for validation
    )
    print(f"Targets directory: {processed.targets_dir}")
    print(f"MSA directory: {processed.msa_dir}")
    
    # Step 5: Load checkpoint and model
    print("\n" + "=" * 60)
    print("Step 5: Loading checkpoint and model")
    print("=" * 60)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    
    # Try to load hyperparameters from checkpoint first
    try:
        checkpoint_data = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
        hparams = checkpoint_data.get("hyper_parameters", {})
        print(f"Found hyperparameters in checkpoint")
        
        # Extract hyperparameters if available
        if "diffusion_process_args" in hparams:
            diffusion_params_dict = hparams["diffusion_process_args"]
            diffusion_params = BoltzDiffusionParams(**diffusion_params_dict)
            print(f"  Using step_scale from checkpoint: {diffusion_params.step_scale}")
        else:
            diffusion_params = BoltzDiffusionParams()
            diffusion_params.step_scale = 1.638  # Boltz1 default
            print(f"  Using default step_scale: {diffusion_params.step_scale}")
        
        if "pairformer_args" in hparams:
            pairformer_args = PairformerArgs(**hparams["pairformer_args"])
            print(f"  Using pairformer_args from checkpoint")
        else:
            pairformer_args = PairformerArgs()
            print(f"  Using default pairformer_args")
        
        if "msa_args" in hparams:
            msa_args_dict = hparams["msa_args"]
            msa_args = MSAModuleArgs(**msa_args_dict)
            print(f"  Using msa_args from checkpoint: subsample_msa={msa_args.subsample_msa}")
        else:
            msa_args = MSAModuleArgs(
                subsample_msa=False,
                num_subsampled_msa=1024,
                use_paired_feature=False,  # Boltz1 doesn't use paired features
            )
            print(f"  Using default msa_args: subsample_msa=False")
        
        if "steering_args" in hparams:
            steering_args = BoltzSteeringParams(**hparams["steering_args"])
            print(f"  Using steering_args from checkpoint")
        else:
            steering_args = BoltzSteeringParams()
            print(f"  Using default steering_args")
            
        # Check if EMA was used during training
        use_ema_from_checkpoint = hparams.get("ema", False)
        if use_ema_from_checkpoint and args.use_ema:
            print(f"  EMA was used during training, will use EMA weights")
        elif use_ema_from_checkpoint and not args.use_ema:
            print(f"  WARNING: EMA was used during training but --use-ema not set!")
            print(f"  This may significantly reduce performance. Consider using --use-ema")
        
        # Extract confidence-related hyperparameters (critical for confidence module initialization)
        # These determine whether the confidence module is initialized
        confidence_prediction_from_checkpoint = hparams.get("confidence_prediction", True)  # Default True for boltz1_conf.ckpt
        confidence_model_args_from_checkpoint = hparams.get("confidence_model_args", {})
        confidence_imitate_trunk_from_checkpoint = hparams.get("confidence_imitate_trunk", False)
        alpha_pae_from_checkpoint = hparams.get("alpha_pae", 0.0)
        
        print(f"  confidence_prediction: {confidence_prediction_from_checkpoint}")
        if confidence_model_args_from_checkpoint:
            print(f"  confidence_model_args: {list(confidence_model_args_from_checkpoint.keys())}")
        print(f"  confidence_imitate_trunk: {confidence_imitate_trunk_from_checkpoint}")
        print(f"  alpha_pae: {alpha_pae_from_checkpoint}")
        
        # Store checkpoint_data for later use (for state_dict filtering)
        checkpoint_data_for_loading = checkpoint_data
    except Exception as e:
        print(f"Warning: Could not load hyperparameters from checkpoint: {e}")
        print(f"  Using default hyperparameters")
        # Fall back to defaults
        diffusion_params = BoltzDiffusionParams()
        diffusion_params.step_scale = 1.638  # Boltz1 default
        pairformer_args = PairformerArgs()
        msa_args = MSAModuleArgs(
            subsample_msa=False,
            num_subsampled_msa=1024,
            use_paired_feature=False,  # Boltz1 doesn't use paired features
        )
        steering_args = BoltzSteeringParams()
        use_ema_from_checkpoint = False
        # Default confidence settings (boltz1_conf.ckpt typically has confidence_prediction=True)
        confidence_prediction_from_checkpoint = True  # Default to True for inference
        confidence_model_args_from_checkpoint = {}
        confidence_imitate_trunk_from_checkpoint = False
        alpha_pae_from_checkpoint = 0.0
        # Load checkpoint data separately for state_dict filtering
        checkpoint_data_for_loading = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
        # Try to extract confidence_prediction from checkpoint even if other hparams failed
        try:
            hparams_fallback = checkpoint_data_for_loading.get("hyper_parameters", {})
            confidence_prediction_from_checkpoint = hparams_fallback.get("confidence_prediction", True)
            confidence_model_args_from_checkpoint = hparams_fallback.get("confidence_model_args", {})
            confidence_imitate_trunk_from_checkpoint = hparams_fallback.get("confidence_imitate_trunk", False)
            alpha_pae_from_checkpoint = hparams_fallback.get("alpha_pae", 0.0)
            print(f"  Extracted confidence_prediction from checkpoint: {confidence_prediction_from_checkpoint}")
        except Exception:
            print(f"  Using default confidence_prediction=True")
    
    # Predict args (matching validation settings from config)
    # Note: symmetry_correction is part of validation_args, not predict_args
    # It will be loaded from checkpoint's hyper_parameters automatically
    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "diffusion_samples": args.diffusion_samples,
        "max_parallel_samples": args.diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    
    print(f"\nPrediction settings:")
    print(f"  Recycling steps: {predict_args['recycling_steps']}")
    print(f"  Sampling steps: {predict_args['sampling_steps']}")
    print(f"  Diffusion samples: {predict_args['diffusion_samples']}")
    print(f"  Symmetry correction: {args.symmetry_correction} (will be set via validation_args)")
    print(f"  Using EMA weights: {args.use_ema}")
    
    # Load checkpoint state_dict and filter out problematic msa_proj layer if size mismatch exists
    # Use checkpoint_data_for_loading if we already loaded it, otherwise load fresh
    if 'checkpoint_data_for_loading' not in locals():
        checkpoint_data_for_loading = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    
    checkpoint_state = checkpoint_data_for_loading
    msa_proj_weight = None
    msa_proj_bias = None
    confidence_msa_proj_weight = None
    confidence_msa_proj_bias = None
    
    # Check what modules are present in checkpoint
    checkpoint_keys = list(checkpoint_state["state_dict"].keys())
    confidence_keys = [k for k in checkpoint_keys if k.startswith("confidence_module")]
    trunk_keys = [k for k in checkpoint_keys if k.startswith("trunk")]
    structure_keys = [k for k in checkpoint_keys if k.startswith("structure_module")]
    
    print(f"\nCheckpoint contains:")
    print(f"  Confidence module keys: {len(confidence_keys)}")
    if confidence_keys:
        print(f"    Sample confidence keys: {confidence_keys[:3]}...")
    print(f"  Trunk keys: {len(trunk_keys)}")
    print(f"  Structure module (denoiser) keys: {len(structure_keys)}")
    print(f"  Total keys: {len(checkpoint_keys)}")
    
    # Check if we need to load missing weights from base checkpoint
    needs_base_checkpoint = len(confidence_keys) == 0 or len(trunk_keys) == 0
    
    if needs_base_checkpoint:
        print(f"\n⚠ Checkpoint appears to be from denoiser-only training:")
        print(f"   - Confidence module keys: {len(confidence_keys)} {'✗ MISSING' if len(confidence_keys) == 0 else '✓'}")
        print(f"   - Trunk keys: {len(trunk_keys)} {'✗ MISSING' if len(trunk_keys) == 0 else '✓'}")
        
        if args.base_checkpoint is None:
            print(f"\n✗ ERROR: Missing weights require --base-checkpoint to be specified!")
            print(f"  Please provide the path to boltz1_conf.ckpt using --base-checkpoint")
            print(f"  Example: --base-checkpoint /path/to/boltz1_conf.ckpt")
            raise ValueError("Missing confidence/trunk weights. Use --base-checkpoint to provide boltz1_conf.ckpt")
        
        print(f"\n Loading missing weights from base checkpoint: {args.base_checkpoint}")
        base_checkpoint = torch.load(str(args.base_checkpoint), map_location="cpu", weights_only=False)
        base_state_dict = base_checkpoint["state_dict"]
        base_hparams = base_checkpoint.get("hyper_parameters", {})
        
        # Count what's in base checkpoint
        base_confidence_keys = [k for k in base_state_dict.keys() if k.startswith("confidence_module")]
        base_trunk_keys = [k for k in base_state_dict.keys() if k.startswith("trunk")]
        print(f"  Base checkpoint confidence keys: {len(base_confidence_keys)}")
        print(f"  Base checkpoint trunk keys: {len(base_trunk_keys)}")
        
        # Merge: use base checkpoint as foundation, then override with retrained checkpoint
        merged_state_dict = {}
        
        # First, copy all weights from base checkpoint
        for key, value in base_state_dict.items():
            merged_state_dict[key] = value
        
        # Then, override with weights from retrained checkpoint (denoiser weights)
        override_count = 0
        for key, value in checkpoint_state["state_dict"].items():
            merged_state_dict[key] = value
            override_count += 1
        
        print(f"  Merged state_dict: {len(merged_state_dict)} total keys")
        print(f"  Overrode {override_count} keys from retrained checkpoint")
        
        # Use the merged state_dict
        checkpoint_state["state_dict"] = merged_state_dict
        
        # Use confidence hyperparameters from base checkpoint
        confidence_prediction_from_checkpoint = base_hparams.get("confidence_prediction", True)
        confidence_model_args_from_checkpoint = base_hparams.get("confidence_model_args", {})
        confidence_imitate_trunk_from_checkpoint = base_hparams.get("confidence_imitate_trunk", False)
        alpha_pae_from_checkpoint = base_hparams.get("alpha_pae", 0.0)
        
        print(f"\n  Using hyperparameters from base checkpoint:")
        print(f"    confidence_prediction: {confidence_prediction_from_checkpoint}")
        print(f"    confidence_model_args keys: {list(confidence_model_args_from_checkpoint.keys())}")
        
        # Update checkpoint keys list for the filtering step
        checkpoint_keys = list(checkpoint_state["state_dict"].keys())
    
    # Check for size mismatch and filter out problematic keys
    filtered_state_dict = {}
    for key, value in checkpoint_state["state_dict"].items():
        if key == "msa_module.msa_proj.weight":
            # Store for later manual handling (main model's msa_proj)
            msa_proj_weight = value
            # Don't add to filtered_state_dict - we'll handle it manually
            continue
        elif key == "msa_module.msa_proj.bias":
            # Store for later manual handling (main model's msa_proj)
            msa_proj_bias = value
            # Don't add to filtered_state_dict - we'll handle it manually
            continue
        elif key == "confidence_module.msa_module.msa_proj.weight":
            # Store for later manual handling (confidence module's msa_proj)
            confidence_msa_proj_weight = value
            # Don't add to filtered_state_dict - we'll handle it manually
            continue
        elif key == "confidence_module.msa_module.msa_proj.bias":
            # Store for later manual handling (confidence module's msa_proj)
            confidence_msa_proj_bias = value
            # Don't add to filtered_state_dict - we'll handle it manually
            continue
        else:
            filtered_state_dict[key] = value
    
    # Create a temporary checkpoint dict with filtered state_dict
    # IMPORTANT: Preserve all hyper_parameters so load_from_checkpoint can use them
    temp_checkpoint = checkpoint_state.copy()
    temp_checkpoint["state_dict"] = filtered_state_dict
    
    # If we used base checkpoint, update hyperparameters to enable confidence module
    if needs_base_checkpoint and "hyper_parameters" in temp_checkpoint:
        temp_checkpoint["hyper_parameters"]["confidence_prediction"] = confidence_prediction_from_checkpoint
        if confidence_model_args_from_checkpoint:
            temp_checkpoint["hyper_parameters"]["confidence_model_args"] = confidence_model_args_from_checkpoint
        temp_checkpoint["hyper_parameters"]["confidence_imitate_trunk"] = confidence_imitate_trunk_from_checkpoint
        temp_checkpoint["hyper_parameters"]["alpha_pae"] = alpha_pae_from_checkpoint
        print(f"\n  Updated temp checkpoint hyperparameters for confidence module")
    
    # Verify hyper_parameters are preserved
    if "hyper_parameters" in temp_checkpoint:
        temp_hparams = temp_checkpoint["hyper_parameters"]
        print(f"\nTemp checkpoint hyper_parameters:")
        print(f"  confidence_prediction: {temp_hparams.get('confidence_prediction', 'NOT FOUND')}")
        print(f"  confidence_model_args present: {'confidence_model_args' in temp_hparams}")
    else:
        print(f"\n⚠ WARNING: No hyper_parameters in temp checkpoint!")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.ckpt') as tmp_file:
        temp_checkpoint_path = tmp_file.name
        torch.save(temp_checkpoint, tmp_file.name)
    
    try:
        # Build kwargs for load_from_checkpoint, including all hyperparameters from checkpoint
        load_kwargs = {
            "strict": False,  # Allow partial loading
            "predict_args": predict_args,
            "map_location": "cpu",
            "diffusion_process_args": asdict(diffusion_params),
            "ema": args.use_ema,  # Use EMA weights if requested
            "use_kernels": not args.no_kernels,  # Use optimized kernels (disable if CUDA lib issues)
            "pairformer_args": asdict(pairformer_args),
            "msa_args": asdict(msa_args),
            "steering_args": asdict(steering_args),
        }
        
        # Add confidence-related parameters (critical for confidence module)
        # Always pass these explicitly to ensure confidence module is initialized
        # Default to True if not found in checkpoint (boltz1_conf.ckpt typically has it True)
        if 'confidence_prediction_from_checkpoint' in locals():
            load_kwargs["confidence_prediction"] = confidence_prediction_from_checkpoint
        else:
            # Fallback: default to True for inference (boltz1_conf.ckpt has confidence)
            load_kwargs["confidence_prediction"] = True
            print(f"  WARNING: confidence_prediction not found in checkpoint, defaulting to True")
        
        if 'confidence_model_args_from_checkpoint' in locals() and confidence_model_args_from_checkpoint:
            load_kwargs["confidence_model_args"] = confidence_model_args_from_checkpoint
        else:
            load_kwargs["confidence_model_args"] = {}
        
        if 'confidence_imitate_trunk_from_checkpoint' in locals():
            load_kwargs["confidence_imitate_trunk"] = confidence_imitate_trunk_from_checkpoint
        else:
            load_kwargs["confidence_imitate_trunk"] = False
        
        if 'alpha_pae_from_checkpoint' in locals():
            load_kwargs["alpha_pae"] = alpha_pae_from_checkpoint
        else:
            load_kwargs["alpha_pae"] = 0.0
        
        print(f"\nPassing to load_from_checkpoint:")
        print(f"  confidence_prediction: {load_kwargs['confidence_prediction']}")
        print(f"  confidence_model_args keys: {list(load_kwargs.get('confidence_model_args', {}).keys())}")
        print(f"  confidence_imitate_trunk: {load_kwargs['confidence_imitate_trunk']}")
        print(f"  alpha_pae: {load_kwargs['alpha_pae']}")
        
        # Load model from filtered checkpoint (without msa_proj)
        # Note: load_from_checkpoint will extract other hyperparameters from checkpoint automatically
        model_module = Boltz1.load_from_checkpoint(
            temp_checkpoint_path,
            **load_kwargs,
        )
        
        # Now manually handle the msa_proj layer
        if msa_proj_weight is not None:
            current_msa_proj_weight = model_module.msa_module.msa_proj.weight
            if msa_proj_weight.shape[1] != current_msa_proj_weight.shape[1]:
                print(f"\nHandling msa_proj size mismatch:")
                print(f"  Checkpoint: {msa_proj_weight.shape}")
                print(f"  Current model: {current_msa_proj_weight.shape}")
                
                # Copy the compatible part
                min_features = min(msa_proj_weight.shape[1], current_msa_proj_weight.shape[1])
                current_msa_proj_weight.data[:, :min_features].copy_(msa_proj_weight[:, :min_features])
                
                # Initialize the extra feature dimension if current model has more features
                if current_msa_proj_weight.shape[1] > msa_proj_weight.shape[1]:
                    import torch.nn.init as init
                    init.xavier_uniform_(current_msa_proj_weight.data[:, min_features:])
                    print(f"  Initialized {current_msa_proj_weight.shape[1] - min_features} new feature dimensions with Xavier uniform")
            else:
                # Shapes match, can copy directly
                current_msa_proj_weight.data.copy_(msa_proj_weight)
                print(f"  Copied msa_proj.weight (shapes matched)")
        
        # Handle bias if it exists
        if msa_proj_bias is not None and model_module.msa_module.msa_proj.bias is not None:
            if msa_proj_bias.shape == model_module.msa_module.msa_proj.bias.shape:
                model_module.msa_module.msa_proj.bias.data.copy_(msa_proj_bias)
                print(f"  Copied msa_proj.bias")
        
        # Now manually handle the confidence module's msa_proj layer (if confidence module exists)
        if confidence_msa_proj_weight is not None:
            if hasattr(model_module, 'confidence_module') and model_module.confidence_module is not None:
                if hasattr(model_module.confidence_module, 'msa_module'):
                    current_conf_msa_proj_weight = model_module.confidence_module.msa_module.msa_proj.weight
                    if confidence_msa_proj_weight.shape[1] != current_conf_msa_proj_weight.shape[1]:
                        print(f"\nHandling confidence_module.msa_proj size mismatch:")
                        print(f"  Checkpoint: {confidence_msa_proj_weight.shape}")
                        print(f"  Current model: {current_conf_msa_proj_weight.shape}")
                        
                        # Copy the compatible part
                        min_features = min(confidence_msa_proj_weight.shape[1], current_conf_msa_proj_weight.shape[1])
                        current_conf_msa_proj_weight.data[:, :min_features].copy_(confidence_msa_proj_weight[:, :min_features])
                        
                        # Initialize the extra feature dimension if current model has more features
                        if current_conf_msa_proj_weight.shape[1] > confidence_msa_proj_weight.shape[1]:
                            import torch.nn.init as init
                            init.xavier_uniform_(current_conf_msa_proj_weight.data[:, min_features:])
                            print(f"  Initialized {current_conf_msa_proj_weight.shape[1] - min_features} new feature dimensions with Xavier uniform")
                    else:
                        # Shapes match, can copy directly
                        current_conf_msa_proj_weight.data.copy_(confidence_msa_proj_weight)
                        print(f"  Copied confidence_module.msa_proj.weight (shapes matched)")
                    
                    # Handle confidence module bias if it exists
                    if confidence_msa_proj_bias is not None and model_module.confidence_module.msa_module.msa_proj.bias is not None:
                        if confidence_msa_proj_bias.shape == model_module.confidence_module.msa_module.msa_proj.bias.shape:
                            model_module.confidence_module.msa_module.msa_proj.bias.data.copy_(confidence_msa_proj_bias)
                            print(f"  Copied confidence_module.msa_proj.bias")
                else:
                    print(f"\n⚠ WARNING: confidence_module.msa_proj.weight found in checkpoint but confidence_module.msa_module not found in model")
            else:
                print(f"\n⚠ WARNING: confidence_module.msa_proj.weight found in checkpoint but confidence_module not initialized")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_checkpoint_path):
            os.unlink(temp_checkpoint_path)
    
    model_module.eval()
    print("\n" + "=" * 60)
    print("Model loaded successfully")
    
    # CRITICAL: Verify confidence module was initialized correctly
    print(f"\nVerifying confidence module initialization:")
    print(f"  model_module.confidence_prediction = {getattr(model_module, 'confidence_prediction', 'ATTRIBUTE NOT FOUND')}")
    print(f"  hasattr(model_module, 'confidence_module') = {hasattr(model_module, 'confidence_module')}")
    
    if hasattr(model_module, 'confidence_prediction'):
        if model_module.confidence_prediction:
            if hasattr(model_module, 'confidence_module') and model_module.confidence_module is not None:
                print(f"✓ Confidence module initialized successfully")
                # Check if confidence module has weights loaded
                confidence_params = sum(p.numel() for p in model_module.confidence_module.parameters())
                print(f"  Confidence module has {confidence_params:,} parameters")
                
                # Check if confidence module has any weights that are non-zero (indicating they were loaded)
                confidence_has_weights = False
                for name, param in model_module.confidence_module.named_parameters():
                    if param.requires_grad and param.numel() > 0:
                        if torch.any(param != 0):
                            confidence_has_weights = True
                            break
                print(f"  Confidence module has non-zero weights: {confidence_has_weights}")
            else:
                print(f"✗ ERROR: confidence_prediction=True but confidence_module is None!")
                print(f"  This will cause KeyError: 'complex_plddt' during prediction!")
                raise RuntimeError("Confidence module not initialized despite confidence_prediction=True")
        else:
            print(f"✗ ERROR: confidence_prediction=False - confidence module not initialized")
            print(f"  This will cause KeyError: 'complex_plddt' during prediction!")
            print(f"\nAttempting to fix by checking checkpoint hyperparameters...")
            # Try to force enable confidence_prediction
            if 'confidence_prediction_from_checkpoint' in locals() and confidence_prediction_from_checkpoint:
                print(f"  Checkpoint had confidence_prediction=True, but model has False")
                print(f"  This suggests load_from_checkpoint didn't use checkpoint hyperparameters correctly")
                raise RuntimeError("Model loaded with confidence_prediction=False but checkpoint has True")
    else:
        print(f"✗ ERROR: model_module.confidence_prediction attribute not found!")
        raise RuntimeError("Model missing confidence_prediction attribute")
    
    # Verify all modules have weights loaded (not just denoiser)
    total_params = sum(p.numel() for p in model_module.parameters())
    trunk_params = sum(p.numel() for p in model_module.trunk.parameters()) if hasattr(model_module, 'trunk') else 0
    structure_params = sum(p.numel() for p in model_module.structure_module.parameters()) if hasattr(model_module, 'structure_module') else 0
    confidence_params = sum(p.numel() for p in model_module.confidence_module.parameters()) if hasattr(model_module, 'confidence_module') and model_module.confidence_module is not None else 0
    
    print(f"\nModel parameter counts:")
    print(f"  Total parameters: {total_params:,}")
    if trunk_params > 0:
        print(f"  Trunk parameters: {trunk_params:,}")
    if structure_params > 0:
        print(f"  Structure module (denoiser) parameters: {structure_params:,}")
    if confidence_params > 0:
        print(f"  Confidence module parameters: {confidence_params:,}")
    
    # Step 6: Set up inference data module
    print("\n" + "=" * 60)
    print("Step 6: Setting up inference data module")
    print("=" * 60)
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=args.num_workers,
        constraints_dir=processed.constraints_dir,
    )
    print("Data module created")
    
    # Step 7: Set up prediction writer
    print("\n" + "=" * 60)
    print("Step 7: Setting up prediction writer")
    print("=" * 60)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_writer = BoltzWriter(
        data_dir=str(processed.targets_dir),
        output_dir=str(args.output_dir),
        output_format="mmcif",  # CIF format expected by eval scripts
        boltz2=False,  # Using Boltz1
        write_embeddings=False,
    )
    print(f"Predictions will be written to: {args.output_dir}")
    print("Output format: mmcif (CIF files)")
    print("File format: {record_id}/{record_id}_model_{model_id}.cif")
    
    # Step 8: Set up trainer
    print("\n" + "=" * 60)
    print("Step 8: Setting up trainer")
    print("=" * 60)
    strategy = "auto"
    if args.devices > 1:
        start_method = "fork" if platform.system() != "Windows" else "spawn"
        strategy = DDPStrategy(start_method=start_method)
        if len(filtered_manifest.records) < args.devices:
            print(f"Warning: Number of devices ({args.devices}) > number of records ({len(filtered_manifest.records)})")
            print(f"Using {len(filtered_manifest.records)} devices")
            devices = len(filtered_manifest.records)
        else:
            devices = args.devices
    else:
        devices = args.devices
    
    trainer = Trainer(
        default_root_dir=str(args.output_dir),
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        precision=32,  # Boltz1 uses fp32
    )
    print(f"Trainer configured with {devices} device(s)")
    print(f"Accelerator: {trainer.accelerator}")
    
    # Step 9: Run predictions
    print("\n" + "=" * 60)
    print("Step 9: Running predictions")
    print("=" * 60)
    if len(filtered_manifest.records) == 0:
        print("ERROR: No records to predict!")
        return
    
    print(f"Running predictions on {len(filtered_manifest.records)} structures...")
    print("This may take a while depending on the number of structures and samples.")
    
    trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=False,
    )
    
    # Step 10: Summary
    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)
    print(f"\nPredictions written to: {args.output_dir}")
    print(f"Total structures predicted: {len(filtered_manifest.records)}")
    print(f"Samples per structure: {predict_args['diffusion_samples']}")
    print(f"Recycling steps: {predict_args['recycling_steps']}")
    print(f"Sampling steps: {predict_args['sampling_steps']}")
    print("\nFile structure:")
    print(f"  {args.output_dir}/")
    print(f"    {{record_id}}/")
    print(f"      {{record_id}}_model_0.cif")
    print(f"      {{record_id}}_model_1.cif")
    print(f"      ...")
    print(f"      confidence_{{record_id}}_model_{{model_id}}.json")
    print("\nThis format matches what the eval scripts expect!")
    print("\nNext step: Run evaluation with:")
    print(f"  python scripts/eval/run_evals.py \\")
    print(f"      {args.output_dir} \\")
    print(f"      <reference_pdb_dir> \\")
    print(f"      <eval_output_dir> \\")
    print(f"      --format boltz \\")
    print(f"      --testset test")


if __name__ == "__main__":
    main()

