#!/bin/bash

# Usage: bash run_v100.sh [-p <partition>] [-n <nodes>] [-g <gpus_per_node>] <config_dir> [extra_args...]
# Example: bash run_v100.sh configs/large_mem --sit
# Example: bash run_v100.sh -p scavenge -n 4 -g 8 configs/large_mem

PARTITION="scavenge"
NODES=16
GPUS_PER_NODE=8
CONFIG_DIR=""
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p)
      PARTITION="$2"; shift 2 ;;
    -n)
      NODES="$2"; shift 2 ;;
    -g|--gpus)
      GPUS_PER_NODE="$2"; shift 2 ;;
    -*)
      # Any other option is an extra argument
      EXTRA_ARGS+=("$1")
      if [[ $# -gt 1 && "${2:0:1}" != "-" ]]; then
        EXTRA_ARGS+=("$2")
        shift 2
      else
        shift 1
      fi
      ;;
    *)
      # First non-option is the config dir
      if [[ -z "$CONFIG_DIR" ]]; then
        CONFIG_DIR="$1"; shift 1
      else
        # Subsequent non-options are considered extra arguments
        EXTRA_ARGS+=("$1")
        shift 1
      fi
      ;;
  esac
done

if [ -z "$CONFIG_DIR" ]; then
    echo "Usage: $0 [-p <partition>] [-n <nodes>] [-g <gpus_per_node>] <config_dir> [extra_args...]"
    exit 1
fi

for config in "$CONFIG_DIR"/*; do
    if [ -f "$config" ]; then
        name=$(basename "$config")
        name_no_ext="${name%.*}"
        bash ./run_slurm.sh --slurm --nodes="$NODES" --gpus-per-node="$GPUS_PER_NODE" --qos normal --timeout 4320 \
            --account=fairusers --partition "$PARTITION" --use_volta32 \
            --config "$config" -n "$name_no_ext" "${EXTRA_ARGS[@]}"
    fi
done

# bash ./run_slurm.sh --slurm --nodes=16 --gpus-per-node=8 --qos normal --timeout 4320 \
#            --account=fairusers --partition scavenge --use_volta32 \
#            --config configs/norm/cfg_1.5_no_norm.yaml -n cfg_1.5_no_norm
# bash run_v100.sh configs/large_model
# bash run_v100.sh configs/lap_only
# bash run_v100.sh toy_configs/cfg --toy
# bash run_v100.sh configs/dit_ablations
# bash run_v100.sh configs/server_dbg
# bash run_v100.sh configs/sit_configs/ablate_noise --sit
# bash run_v100.sh configs/style_ablates/noise_mlp
# bash run_v100.sh configs_new/imagenet64
# bash run_v100.sh configs_new/ablate_noiseclass
# bash run_v100.sh configs_new/sit64 --sit
# bash run_v100.sh configs_new/temp --sit
# ms-python.vscode-pylance-2024.8.1
# bash run_v100.sh configs_new/styleGAN64
# bash run_v100.sh configs_new/cifar
# bash run_v100.sh configs_new/imagenet32
# bash run_v100.sh configs_new/temp
# bash run_v100.sh configs_new/cache256/new_loss_func
# bash run_v100.sh configs_new/cache256/256_scaleup
# bash run_v100.sh configs_new/cache256/256_more_runs
# bash run_v100.sh configs_new/cache256/256_more_runs/continue -p scavenge --num_retries 3
# bash run_v100.sh configs_new/cache256/256_more_runs/continue_quick_warmup -p scavenge --num_retries 3
# bash run_v100.sh configs_new/cache256/styleGAN -p scavenge --num_retries 3
# bash run_v100.sh /private/home/mingyangd/dmy/nn_flow/configs_new/cache256/256_more_runs/cont_temp -p learnfair --num_retries 3
# bash run_v100.sh /private/home/mingyangd/dmy/nn_flow/configs_new/temp -p scavenge --num_retries 3
# bash run_v100.sh /private/home/mingyangd/dmy/nn_flow/configs_new/cache256/styleGAN -p learnfair --num_retries 3