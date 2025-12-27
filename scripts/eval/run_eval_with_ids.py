#!/usr/bin/env python3
"""
Helper script to run evaluation with validation IDs and custom checkpoints.

This script helps you:
1. Parse validation IDs from a text file
2. Download reference structures from PDB
3. Run evaluation on your predictions

Usage:
    # Step 1: Download reference structures
    python scripts/eval/run_eval_with_ids.py download-refs \
        --ids-file validation_ids.txt \
        --output-dir ./references

    # Step 2: After generating predictions with your checkpoint, run evaluation
    python scripts/eval/run_eval_with_ids.py evaluate \
        --ids-file validation_ids.txt \
        --predictions-dir ./predictions \
        --references-dir ./references \
        --output-dir ./evaluation_results \
        --format boltz
"""

import argparse
import gzip
import shutil
import subprocess
from pathlib import Path
from typing import Set

import requests
from tqdm import tqdm


def parse_ids_file(ids_file: Path) -> list[str]:
    """Parse validation IDs from a text file.
    
    Expected format: one ID per line, e.g.:
        WF6_7FZC
        WHI_7FZP
        LM9_7FWB
    
    Returns the list of IDs (keeping the full format).
    """
    with ids_file.open("r") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def extract_pdb_ids(ids: list[str]) -> Set[str]:
    """Extract unique PDB IDs from validation IDs.
    
    Assumes format: {ligand_code}_{pdb_id}
    Returns set of unique PDB IDs (lowercase).
    """
    pdb_ids = set()
    for val_id in ids:
        # Split by underscore and take the last part (PDB ID)
        parts = val_id.split("_")
        if len(parts) >= 2:
            pdb_id = parts[-1].lower()
            pdb_ids.add(pdb_id)
        else:
            # If no underscore, assume the whole thing is a PDB ID
            pdb_ids.add(val_id.lower())
    return pdb_ids


def download_pdb_structure(pdb_id: str, output_dir: Path, compressed: bool = True) -> bool:
    """Download a PDB structure from RCSB.
    
    Parameters
    ----------
    pdb_id : str
        PDB ID (e.g., '7fzc')
    output_dir : Path
        Directory to save the structure
    compressed : bool
        Whether to download compressed (.cif.gz) or uncompressed (.cif)
    
    Returns
    -------
    bool
        True if download successful, False otherwise
    """
    pdb_id_upper = pdb_id.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if compressed:
        url = f"https://files.rcsb.org/download/{pdb_id_upper}.cif.gz"
        output_file = output_dir / f"{pdb_id.lower()}.cif.gz"
    else:
        url = f"https://files.rcsb.org/download/{pdb_id_upper}.cif"
        output_file = output_dir / f"{pdb_id.lower()}.cif"
    
    # Skip if already exists
    if output_file.exists():
        return True
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with output_file.open("wb") as f:
            f.write(response.content)
        
        return True
    except Exception as e:
        print(f"Failed to download {pdb_id}: {e}")  # noqa: T201
        return False


def download_references(ids_file: Path, output_dir: Path, compressed: bool = True):
    """Download reference structures for validation IDs.
    
    Parameters
    ----------
    ids_file : Path
        Path to validation IDs file
    output_dir : Path
        Directory to save reference structures
    compressed : bool
        Whether to download compressed files
    """
    print(f"Parsing IDs from {ids_file}...")  # noqa: T201
    ids = parse_ids_file(ids_file)
    print(f"Found {len(ids)} validation IDs")  # noqa: T201
    
    pdb_ids = extract_pdb_ids(ids)
    print(f"Extracted {len(pdb_ids)} unique PDB IDs")  # noqa: T201
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading structures to {output_dir}...")  # noqa: T201
    successful = 0
    failed = 0
    
    for pdb_id in tqdm(sorted(pdb_ids)):
        if download_pdb_structure(pdb_id, output_dir, compressed=compressed):
            successful += 1
        else:
            failed += 1
    
    print(f"\nDownload complete: {successful} successful, {failed} failed")  # noqa: T201


def check_predictions_exist(predictions_dir: Path, ids: list[str], format: str, num_samples: int = 5) -> dict[str, bool]:
    """Check which predictions exist for the given IDs.
    
    Parameters
    ----------
    predictions_dir : Path
        Directory containing predictions
    ids : list[str]
        List of validation IDs
    format : str
        Format of predictions ('boltz', 'af3', 'chai')
    num_samples : int
        Number of samples expected per structure
    
    Returns
    -------
    dict[str, bool]
        Mapping from ID to whether predictions exist
    """
    results = {}
    
    for val_id in ids:
        val_id_lower = val_id.lower()
        pred_folder = predictions_dir / val_id_lower
        
        if not pred_folder.exists():
            results[val_id] = False
            continue
        
        # Check for expected prediction files based on format
        found_samples = 0
        for model_id in range(num_samples):
            if format == "boltz":
                name_file = val_id_lower
                pred_path = pred_folder / f"{name_file}_model_{model_id}.cif"
            elif format == "af3":
                pred_path = pred_folder / f"seed-1_sample-{model_id}" / "model.cif"
            elif format == "chai":
                pred_path = pred_folder / f"pred.model_idx_{model_id}.cif"
            else:
                pred_path = None
            
            if pred_path and pred_path.exists():
                found_samples += 1
        
        results[val_id] = found_samples >= num_samples
    
    return results


def run_evaluation(
    ids_file: Path,
    predictions_dir: Path,
    references_dir: Path,
    output_dir: Path,
    format: str = "boltz",
    testset: str = "test",
    mount: str = None,
    max_workers: int = 32,
    num_samples: int = 5,
):
    """Run evaluation on predictions.
    
    Parameters
    ----------
    ids_file : Path
        Path to validation IDs file
    predictions_dir : Path
        Directory containing predictions (should have subdirectories for each ID)
    references_dir : Path
        Directory containing reference PDB/CIF files
    output_dir : Path
        Directory to save evaluation results
    format : str
        Format of predictions ('boltz', 'af3', 'chai')
    testset : str
        Test set type ('test' or 'casp')
    mount : str
        Mount point for Docker (if None, uses current working directory)
    max_workers : int
        Maximum number of parallel workers
    num_samples : int
        Number of samples per structure
    """
    print(f"Parsing IDs from {ids_file}...")  # noqa: T201
    ids = parse_ids_file(ids_file)
    print(f"Found {len(ids)} validation IDs")  # noqa: T201
    
    # Check which predictions exist
    print("Checking for existing predictions...")  # noqa: T201
    pred_status = check_predictions_exist(predictions_dir, ids, format, num_samples)
    
    missing = [vid for vid, exists in pred_status.items() if not exists]
    if missing:
        print(f"Warning: Missing predictions for {len(missing)} IDs: {missing[:5]}...")  # noqa: T201
        print("Make sure you've run predictions first with: boltz predict ...")  # noqa: T201
    
    # Create filtered predictions directory with only structures that have predictions
    filtered_pred_dir = output_dir / "filtered_predictions"
    filtered_pred_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating filtered predictions directory...")  # noqa: T201
    for val_id, exists in pred_status.items():
        if exists:
            val_id_lower = val_id.lower()
            source = predictions_dir / val_id_lower
            dest = filtered_pred_dir / val_id_lower
            if not dest.exists():
                # Create symlink or copy
                try:
                    dest.symlink_to(source.absolute())
                except OSError:
                    # If symlink fails, copy instead
                    shutil.copytree(source, dest)
    
    # Set mount point
    if mount is None:
        mount = str(Path.cwd().absolute())
    
    # Run the evaluation script
    eval_output_dir = output_dir / "evals"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running evaluation...")  # noqa: T201
    print(f"  Predictions: {filtered_pred_dir}")  # noqa: T201
    print(f"  References: {references_dir}")  # noqa: T201
    print(f"  Output: {eval_output_dir}")  # noqa: T201
    
    # Import and run the evaluation
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from scripts.eval.run_evals import main as eval_main
    
    class Args:
        def __init__(self):
            self.data = filtered_pred_dir
            self.pdb = references_dir
            self.outdir = eval_output_dir
            self.format = format
            self.testset = testset
            self.mount = mount
            self.executable = "/bin/bash"
            self.max_workers = max_workers
    
    eval_main(Args())


def main():
    parser = argparse.ArgumentParser(
        description="Helper script to run evaluation with validation IDs and checkpoints"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Download references command
    download_parser = subparsers.add_parser(
        "download-refs",
        help="Download reference structures from PDB"
    )
    download_parser.add_argument(
        "--ids-file",
        type=Path,
        required=True,
        help="Path to validation IDs file (one ID per line)"
    )
    download_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save reference structures"
    )
    download_parser.add_argument(
        "--uncompressed",
        action="store_true",
        help="Download uncompressed CIF files (default: compressed .cif.gz)"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Run evaluation on predictions"
    )
    eval_parser.add_argument(
        "--ids-file",
        type=Path,
        required=True,
        help="Path to validation IDs file"
    )
    eval_parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory containing predictions (with subdirectories for each ID)"
    )
    eval_parser.add_argument(
        "--references-dir",
        type=Path,
        required=True,
        help="Directory containing reference PDB/CIF files"
    )
    eval_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save evaluation results"
    )
    eval_parser.add_argument(
        "--format",
        type=str,
        default="boltz",
        choices=["boltz", "af3", "chai"],
        help="Format of predictions (default: boltz)"
    )
    eval_parser.add_argument(
        "--testset",
        type=str,
        default="test",
        choices=["test", "casp"],
        help="Test set type (default: test)"
    )
    eval_parser.add_argument(
        "--mount",
        type=str,
        default=None,
        help="Mount point for Docker (default: current directory)"
    )
    eval_parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Maximum number of parallel workers (default: 32)"
    )
    eval_parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples per structure (default: 5)"
    )
    
    args = parser.parse_args()
    
    if args.command == "download-refs":
        download_references(
            args.ids_file,
            args.output_dir,
            compressed=not args.uncompressed
        )
    elif args.command == "evaluate":
        run_evaluation(
            args.ids_file,
            args.predictions_dir,
            args.references_dir,
            args.output_dir,
            format=args.format,
            testset=args.testset,
            mount=args.mount,
            max_workers=args.max_workers,
            num_samples=args.num_samples,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

