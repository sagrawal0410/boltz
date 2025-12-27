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
import platform
import sys
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
        default=1,
        help="Number of recycling steps (default: 1)"
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
    
    # Set up model parameters (using Boltz1 defaults)
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = 1.638  # Boltz1 default
    pairformer_args = PairformerArgs()
    msa_args = MSAModuleArgs(
        subsample_msa=False,
        num_subsampled_msa=1024,
        use_paired_feature=False,  # Boltz1 doesn't use paired features
    )
    steering_args = BoltzSteeringParams()
    
    # Predict args (matching validation settings from config)
    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "diffusion_samples": args.diffusion_samples,
        "max_parallel_samples": args.diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    
    # Load model from checkpoint
    # Note: We need to load with the model's hyperparameters from the checkpoint
    # The checkpoint should contain all necessary model config
    model_module = Boltz1.load_from_checkpoint(
        str(args.checkpoint),
        strict=False,  # Allow some flexibility in loading
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,  # Don't use EMA weights for inference
        use_kernels=True,  # Use optimized kernels
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model_module.eval()
    print("Model loaded successfully")
    
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

