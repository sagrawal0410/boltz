#!/bin/bash

# Usage: bash profile_h100.sh [-p <partition>] [-g <gpus_per_node>] [-q <qos>] [-a <account>] [-N <nodes_spec>] [-h200] <config_file> [extra_args...]
# -N/--nodes-list accepts:
#   - Comma list: 1,2,4,8
#   - Range:      1-16
#   - Range+step: 1-16:2
#
# Example:
#   bash profile_h100.sh configs/large_mem/exp.yaml -N 1,2,4,8
#   bash profile_h100.sh -p devaccel -g 8 -q h100_lowest -a flows -N 2-16:2 configs/large_mem/exp.yaml
#   bash profile_h100.sh -h200 -N 4,8,16 configs/large_mem/exp.yaml

PARTITION="h100"
GPUS_PER_NODE=8
QOS="h100_lowest"
ACCOUNT="flows"
CONFIG_FILE=""
EXTRA_ARGS=()
NODES_LIST=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p)
      PARTITION="$2"; shift 2 ;;
    -g|--gpus)
      GPUS_PER_NODE="$2"; shift 2 ;;
    -q|--qos)
      QOS="$2"; shift 2 ;;
    -a|--account)
      ACCOUNT="$2"; shift 2 ;;
    -N|--nodes-list)
      spec="$2"
      if [[ "$spec" == *","* ]]; then
        IFS=',' read -ra NODES_LIST <<< "$spec"
      else
        IFS=':-' read -ra parts <<< "$spec"
        start="${parts[0]}"; end="${parts[1]}"; step="${parts[2]:-1}"
        if [[ -z "$start" || -z "$end" ]]; then
          echo "Invalid -N/--nodes-list spec: $spec"; exit 1
        fi
        for n in $(seq "$start" "$step" "$end"); do
          NODES_LIST+=("$n")
        done
      fi
      shift 2 ;;
    -h200)
      PARTITION="h200"
      QOS="h200_lowest"
      shift 1 ;;
    -*)
      EXTRA_ARGS+=("$1")
      if [[ $# -gt 1 && "${2:0:1}" != "-" ]]; then
        EXTRA_ARGS+=("$2")
        shift 2
      else
        shift 1
      fi
      ;;
    *)
      if [[ -z "$CONFIG_FILE" ]]; then
        CONFIG_FILE="$1"; shift 1
      else
        EXTRA_ARGS+=("$1")
        shift 1
      fi
      ;;
  esac
done

if [ -z "$CONFIG_FILE" ]; then
  echo "Usage: $0 [-p <partition>] [-g <gpus_per_node>] [-q <qos>] [-a <account>] [-N <nodes_spec>] <config_file> [extra_args...]"
  echo "  -N/--nodes-list examples: 1,2,4,8 | 1-16 | 1-16:2"
  exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: config file not found: $CONFIG_FILE"
  exit 1
fi

# Default to a single run at 16 nodes if no list provided
if [ ${#NODES_LIST[@]} -eq 0 ]; then
  NODES_LIST=(16)
fi

name=$(basename "$CONFIG_FILE")
name_no_ext="${name%.*}"

for nodes in "${NODES_LIST[@]}"; do
  job_name="${name_no_ext}_n${nodes}"
  bash ./run_slurm.sh --slurm --nodes="$nodes" --gpus-per-node="$GPUS_PER_NODE" --qos "$QOS" --timeout 4320 \
    --account="$ACCOUNT" --partition "$PARTITION" \
    --config "$CONFIG_FILE" -n "$job_name" "${EXTRA_ARGS[@]}"
done

# Notes:
# - Accepts a single config file, runs across a list/range of node counts.
# - Job name (-n) is suffixed with _n<N> for each submission.
# - Quick switch to H200: -h200 (partition=h200, qos=h200_lowest).
# bash profile_h100.sh /checkpoint/flows/mingyangd/nn_flow/configs_new/temp/B_resnetL_cfg2.yaml -N 8,16