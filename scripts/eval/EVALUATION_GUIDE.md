# Evaluation Guide: Using Your Own Checkpoints with Validation IDs

This guide walks you through running evaluation on your own checkpoints using validation IDs.

## Overview

The workflow consists of three main steps:
1. **Generate predictions** using your checkpoint
2. **Download reference structures** from PDB
3. **Run evaluation** comparing predictions to references

## Step 1: Generate Predictions

First, you need to generate predictions for each structure in your validation set using your checkpoint.

### 1.1 Prepare Input Files

For each validation ID (e.g., `WF6_7FZC`), you need:
- **Input YAML file** with sequences, MSAs, and ligand information
- The PDB ID is `7FZC` (the part after the underscore)

You can either:
- **Option A**: Use pre-computed MSAs (if you have them)
- **Option B**: Use `--use_msa_server` to auto-generate MSAs

### 1.2 Run Predictions

```bash
# Example: Generate predictions for a directory of YAML files
boltz predict ./input_yamls/ \
    --checkpoint /path/to/your/checkpoint.ckpt \
    --out_dir ./predictions \
    --diffusion_samples 5 \
    --use_msa_server  # if you want auto-generated MSAs
```

**Important**: Make sure your predictions are organized as:
```
predictions/
├── wf6_7fzc/
│   ├── wf6_7fzc_model_0.cif
│   ├── wf6_7fzc_model_1.cif
│   ├── wf6_7fzc_model_2.cif
│   ├── wf6_7fzc_model_3.cif
│   └── wf6_7fzc_model_4.cif
├── whi_7fzp/
│   └── ...
└── ...
```

The directory name should match your validation ID (lowercase).

## Step 2: Download Reference Structures

Use the helper script to download reference structures:

```bash
python scripts/eval/run_eval_with_ids.py download-refs \
    --ids-file validation_ids.txt \
    --output-dir ./references
```

This will download all unique PDB structures (extracting PDB IDs from your validation IDs) to the `./references` directory.

## Step 3: Run Evaluation

Once you have predictions and references, run the evaluation:

```bash
python scripts/eval/run_eval_with_ids.py evaluate \
    --ids-file validation_ids.txt \
    --predictions-dir ./predictions \
    --references-dir ./references \
    --output-dir ./evaluation_results \
    --format boltz \
    --testset test
```

**Note**: The evaluation script requires Docker with the OpenStructure image. Make sure Docker is running and you have the `openstructure-0.2.8` image.

### Evaluation Output

The evaluation will create:
```
evaluation_results/
├── evals/
│   ├── wf6_7fzc_model_0.json          # Polymer metrics
│   ├── wf6_7fzc_model_0_ligand.json   # Ligand metrics
│   ├── wf6_7fzc_model_1.json
│   └── ...
└── filtered_predictions/               # Symlinks to predictions
```

## Alternative: Manual Evaluation

If you prefer to run the evaluation script directly:

```bash
python scripts/eval/run_evals.py \
    ./predictions \
    ./references \
    ./evaluation_results \
    --format boltz \
    --testset test \
    --mount $(pwd)
```

## Understanding Your Validation IDs

Your validation IDs are in the format: `{ligand_code}_{pdb_id}`

- Example: `WF6_7FZC` → PDB ID is `7FZC`
- Example: `WHI_7FZP` → PDB ID is `7FZP`

The script automatically extracts the PDB ID (the part after the underscore) to download reference structures.

## Troubleshooting

### Missing Predictions

If some predictions are missing, the script will warn you. Make sure:
1. You've run predictions for all validation IDs
2. The directory structure matches expected format
3. You have 5 samples per structure (model_0 through model_4)

### Docker Issues

If Docker is not available or you get permission errors:
- Make sure Docker is running: `sudo systemctl status docker`
- You may need to run with `sudo` or add your user to the docker group
- Check that the `openstructure-0.2.8` image exists: `docker images | grep openstructure`

### Reference Structure Issues

If reference structures fail to download:
- Check your internet connection
- Some PDB IDs might be deprecated or unavailable
- Try downloading uncompressed: add `--uncompressed` flag

## Next Steps: Aggregating Results

After evaluation, you can aggregate results using:

```bash
python scripts/eval/aggregate_evals.py
```

(You may need to modify this script to point to your evaluation results directory)

