#!/usr/bin/env bash
set -euo pipefail

################################################################################
# run.sh  —  Launch your experiment locally or on SLURM with torchrun + DDP
#
# USAGE (local):
#   ./run.sh [-n NAME] [-d DEVICE] <module> [--arg1 val1 ...]
#
#   -n|--name        Custom run name (default: exp_<timestamp>)
#   -d|--device      CUDA_VISIBLE_DEVICES (single‐GPU only; default 0)
#   <module>         Python module path, e.g. train or apps.llm.train
#   [--arg val ...]  Passed through to your module
#
# USAGE (SLURM):
#   ./run.sh --slurm [-n NAME] [--nodes N] [--gpus-per-node G] <module> [--arg1 val1 ...]
#
#   --slurm           Submit via sbatch instead of local python
#   --nodes N         Number of nodes (default: 1)
#   --gpus-per-node G GPUs per node (default: 1)
#   (note: -d/--device is ignored in SLURM mode)
#
# remaining_args is "everything after all launcher flags"—the first of which
# must be your Python module name (no "-m" needed), followed by its flags.
#
################################################################################

### 0. Parse launcher flags
USE_SLURM=0
custom_name=""
device_num=0
nodes=1
gpus_per_node=1
NOGIT=0
remaining_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slurm)   USE_SLURM=1; shift ;;
    -n|--name) custom_name="$2"; shift 2 ;;
    --name=*)  custom_name="${1#*=}"; shift ;;
    -d|--device)
               device_num="$2"; shift 2 ;;
    --device=*)
               device_num="${1#*=}"; shift ;;
    --nodes)   nodes="$2"; shift 2 ;;
    --nodes=*) nodes="${1#*=}"; shift ;;
    --gpus-per-node)
               gpus_per_node="$2"; shift 2 ;;
    --gpus-per-node=*)
               gpus_per_node="${1#*=}"; shift ;;
    --nogit)   NOGIT=1; shift ;;
    *)
      remaining_args+=("$1")
      shift
      ;;
  esac
done

### 1. Build run name & directory
ts=$(date +%Y%m%d_%H%M%S)
if [[ -n "$custom_name" ]]; then
  run_name="${custom_name}_${ts}"
else
  run_name="exp_${ts}"
fi
exp_folder="runs/${run_name}"
mkdir -p "$exp_folder"

### 2. Snapshot code & configs
rsync -avm --prune-empty-dirs \
  --exclude='runs/**' \
  --exclude='wandb/' --exclude='wandb/**' \
  --exclude='__pycache__/' --exclude='__pycache__/**' \
  --exclude='.git/' --exclude='.gitignore' --exclude='.gitmodules' \
  --exclude='.pytest_cache/' --exclude='*.pyc' \
  --include='*/' \
  --include='*.py' --include='*.sh' \
  --include='*.yaml' --include='*.yml' \
  --include='*.log' --include='*.txt' \
  --include='*.md' --include='*.cfg' \
  --include='*.cpp' --include='*.cu' --include='*.h' \
  --exclude='*' \
  ./  "$exp_folder/"

### 3. (Optional) Commit snapshot
if [[ "$NOGIT" -eq 0 ]]; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    git add .
    git commit -m "Run: $run_name"
    git push
  else
    echo "[run.sh] No code changes to commit."
  fi
fi

### 4. Launch
echo "Launching job with name: $run_name in folder: $exp_folder"
cd "$exp_folder"

if [[ "$USE_SLURM" -eq 1 ]]; then
  python submit.py \
    --nodes "$nodes" \
    --ngpus "$gpus_per_node" \
    --partition learn \
    --qos core_shared \
    --account flows \
    --job_name "$run_name" \
    ${remaining_args[@]}

else
  # Local: single‐GPU via torchrun
  export CUDA_VISIBLE_DEVICES=$device_num
  torchrun \
    --nnodes=1 \
    --nproc-per-node=1 \
    -m train ${remaining_args[@]} \
    --job_name "$run_name" \
    > output.log 2>&1

  echo "[run.sh] Local experiment complete. Results in: $exp_folder"
fi

# bash ./run_slurm.sh --slurm --nodes=2 --gpus-per-node=8 -n vae_32 --qos dev --timeout 1080

# bash ./run_slurm.sh --slurm --nodes=32 --gpus-per-node=8 -n vae_32_6e_4_8recon_deep_head --qos core_shared --timeout 1080