#!/bin/bash

# Usage: bash run_h100.sh [-p <partition>] [-n <nodes>] [-g <gpus_per_node>] [-q <qos>] [-a <account>] [-h200] <config_dir> [extra_args...]
# Example: bash run_h100.sh configs/large_mem --sit
# Example: bash run_h100.sh -p devaccel -n 4 -g 8 -q h100_lowest -a flows configs/large_mem
# Example: bash run_h100.sh -h200 configs/large_mem

PARTITION="h100"
NODES=16
GPUS_PER_NODE=8
QOS="h100_lowest"
ACCOUNT="flows"
NOGIT_ARG=""
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
    -q|--qos)
      QOS="$2"; shift 2 ;;
    -a|--account)
      ACCOUNT="$2"; shift 2 ;;
    -h200)
      # Quick switch to H200 resources
      PARTITION="h200"
      QOS="h200_lowest"
      shift 1 ;;
    --nogit)
      NOGIT_ARG="--nogit"
      shift 1 ;;
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
    echo "Usage: $0 [-p <partition>] [-n <nodes>] [-g <gpus_per_node>] [-q <qos>] [-a <account>] <config_dir> [extra_args...]"
    exit 1
fi

for config in "$CONFIG_DIR"/*; do
    if [ -f "$config" ]; then
        name=$(basename "$config")
        name_no_ext="${name%.*}"
        bash ./run_slurm.sh --slurm --nodes="$NODES" --gpus-per-node="$GPUS_PER_NODE" --qos "$QOS" --timeout 4320 \
            --account="$ACCOUNT" --partition "$PARTITION" \
            --config "$config" -n "$name_no_ext" "${EXTRA_ARGS[@]}" $NOGIT_ARG
    fi
done

# Notes:
# - Defaults tailored for H100: qos=h100_lowest, account=flows, no --use_volta32 flag.
# - Override with -q/--qos and -a/--account as needed; other args forwarded to run_slurm.sh.
#
# Example invocations:
# bash run_h100.sh configs/large_model --num_retries 3
# bash run_h100.sh -h200 configs_new/cache256/256_L --num_retries 2
# bash run_h100.sh config_mae/MAE_resnet34 --mae 
# bash run_h100.sh configs_new/cache256/256_MAE
# bash run_h100.sh configs_new/cache256/256_MAE/eval_is
# bash run_h100.sh configs_new/cache256/256_MAE/resid_style
# bash run_h100.sh configs_new/temp -h200
# bash run_h100.sh configs_new/temp/resid_style_cont_cfg3

# bash run_h100.sh configs_new/cache256/resid_cont/baselines
# bash run_h100.sh configs_new/cache256/256_MAE/cont_training/cfg_and_sinkhorn
# bash run_h100.sh configs_new/cache256/256_MAE/cont_training/optimizer
# bash run_h100.sh configs_new/temp
# bash run_h100.sh configs_new/cache256/256_MAE/cont_training/other_ckpts
# bash run_h100.sh config_mae/MAE_convnext --mae 
# bash run_h100.sh configs_new/cache256/256_MAE/guidance
# bash run_h100.sh configs_new/cache256/256_MAE/more_runs
# bash run_h100.sh config_mae/MAE_convnext --mae 
# bash run_h100.sh config_mae/scaling --mae --num_retries 2
# bash run_h100.sh configs_new/cache256/256_MAE/more_runs/baseline_diff_MAE -h200
# bash run_h100.sh configs_new/temp -n 32
# bash run_h100.sh configs_new/cache256/256_MAE/ablate_features -n -h200
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks -n 8
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/pretrain_34 -n 16 -q h100_core_shared
# bash run_h100.sh configs_new/temp -n 16 -q h200_flows_high
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/345_transpose -n 16 -q h200_flows_high
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/more_tries -n 16 -q h200_flows_high
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/large -n 16
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/cls_tokens -n 16 -q h100_core_shared
# bash run_h100.sh config_mae/ablate_mae -n 16 --mae
# bash run_h100.sh config_mae/more_maes -n 16 --mae
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/90k_ckpt -n 8 -q h100_core_shared
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/more_mae_runs -n 16 
# bash run_h100.sh config_mae/MAE_resnetGN -n 16 --mae
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/MAE_gn -n 16 
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/MAE_gn/new_gn -n 16 
# bash run_h100.sh config_mae/MAE_resnetGN/cont_cls -n 16 --mae
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/MAE_gn/new_gn_60k -n 16 
# bash run_h100.sh config_mae/MAE_resnetGN/gn_augs -n 16 --mae
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/MAE_gn/new_gn_60k -n 16 
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/MAE_gn/Large -n 16 -q h100_core_shared
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/MAE_gn/cfg_emb -n 16 
# bash run_h100.sh config_mae/MAE_vit/better_coeff -n 16 --mae
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/MAE_gn/cfg_emb/large_model -n 16 

# bash run_h100.sh config_mae/MAE_vit/continue -n 16 --mae
# bash run_h100.sh configs_new/cache256/MAE_vit -n 16 -q h100_core_shared
# bash run_h100.sh config_mae/MAE_resnetGN/cont_cls -n 16 --mae -q h100_core_shared
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/MAE_gn/cfg_emb/runs_1016 -n 16 -q h100_core_shared
# bash run_h100.sh configs_new/temp -n 16 -q h100_core_shared
# bash run_h100.sh config_mae/MAE_resnetGN/gn_large -n 16 --mae -q h200_lowest

# bash run_h100.sh configs_new/cache256/256_MAE/finetune -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/256_MAE/finetune/cls_model -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/MAE_gn/Large_mae -n 16 -q h100_core_shared
# bash run_h100.sh configs_new/cache256/256_MAE/feature_chunks/MAE_gn/Large_mae/more_blocks -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1019Lgen/small_bsz -n 16 
# bash run_h100.sh configs_new/temp -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1020_ev_and_bf -n 16 -q h200_lowest
# bash run_h100.sh config_mae/MAE_resnetGN/gn_shallow -n 16 --mae 
# bash run_h100.sh configs_new/cache256/1021_mmd_and_repul -n 16 -q h200_core_shared
# squeue --me -o "%.18i %.20P %.100j %.8u %.2t %.10M %.6D %R"
# bash run_h100.sh configs_new/cache256/1021_mmd_and_repul/small_lr -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1021_mmd_and_repul/cont_eval -n 16 -q h200_lowest


# bash run_h100.sh configs_new/cache256/1021_mmd_and_repul/no_ratio -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1023_vit_proj_interpatch/vit -n 16 -q h200_lowest (v)
# bash run_h100.sh configs_new/cache256/1023_vit_proj_interpatch/shallow_XL -n 16 -q h200_lowest (v)
# bash run_h100.sh configs_new/cache256/1023_vit_proj_interpatch/finetune_XL -n 16 -q h200_core_shared (v)
# bash run_h100.sh config_mae/MAE_resnetGN/finetune_shallow -n 16 --mae --nogit (v)
# bash run_h100.sh configs_new/cache256/1023_vit_proj_interpatch/proj -n 16 -q h200_lowest --nogit (v)
# bash run_h100.sh config_mae/MAE_vit/1024_hier -n 16 --mae --nogit (v)
# bash run_h100.sh configs_new/cache256/1023_vit_proj_interpatch/hier_vit -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1023_vit_proj_interpatch/shallow_temp -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1026_cfgpow/cfg_range -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1026_cfgpow/test_compile -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1026_cfgpow/compile_bf16_fast -n 16 -q h200_lowest
# bash run_h100.sh config_mae/MAE_resnetGN/skip_conn -n 16 --mae
# bash run_h100.sh configs_new/cache256/1026_cfgpow/best_viz_quality -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1029_best_trains -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1029_best_trains/sanity -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1029_best_trains/speed -n 16 -q h200_core_shared --num_retries 0
# bash run_h100.sh config_mae/MAE_resnetGN/1030_scaling -n 16 --mae
# bash run_h100.sh config_mae/MAE_resnetGN/1030_speed -n 16 --mae -q h100_core_shared
# bash run_h100.sh configs_new/cache256/1030_convnext -n 16 -q h100_lowest
# bash run_h100.sh configs_new/cache256/1030_convnext/h200 -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1030_convnext/h200/more -n 16 -q 0_lowest
# bash run_h100.sh configs_new/cache256/1030_convnext/concats -n 16 -q h200_lowest
# bash run_h100.sh config_mae/MAE_resnetGN/1101_scalext -n 16 --mae
# bash run_h100.sh configs_new/cache256/1103_cnext -n 16 -q h200_lowest
# bash run_h100.sh config_mae/MAE_resnetGN/1104_skip -n 16 --mae
# bash run_h100.sh configs_new/cache256/1103_cnext/less_ood -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1103_cnext/more_skip -n 16 -q h200_lowest
# bash run_h100.sh config_mae/MAE_resnetGN/1109_scales -n 16 --mae -q h200_lowest
# bash run_h100.sh config_mae/MAE_resnetGN/1109_scales/eval_maes -n 16 --mae -q h200_lowest
# bash run_h100.sh configs_new/cache256/1110_scaling -n 16 -q h200_lowest
# bash run_h100.sh config_mae/MAE_resnetGN/1109_scales/eval_maes/fixev35 -n 16 --mae 
# bash run_h100.sh configs_new/cache256/1110_scaling/to_run -n 16 -q h200_lowest
# bash run_h100.sh config_mae/MAE_resnetGN/1112_shallow -n 16 --mae
# bash run_h100.sh configs_new/cache256/1110_scaling/wide_cfg -n 16 -q h200_core_shared
# bash run_h100.sh configs_new/cache256/1110_scaling/shallow_model -n 16 -q h200_lowest
# bash run_h100.sh config_mae/MAE_resnetGN/1112_shallow/run_h200 -n 16 --mae -q h200_lowest
# bash run_h100.sh configs_new/cache256/1110_scaling/size_scaling -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1110_scaling/640S -n 16 -q h200_lowest
# bash run_h100.sh config_mae/MAE_resnetGN/1116_shallower -n 16 --mae
# bash run_h100.sh config_mae/MAE_resnetGN/1116_shallower/small_ker -n 16 --mae
# bash run_h100.sh configs_new/cache256/1117_train -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1117_train/sk -n 16 -q h200_lowest
# bash run_h100.sh config_mae/MAE_resnetGN/1116_shallower/even_shallow -n 16 --mae
# bash run_h100.sh configs_new/cache256/1117_train/sk/temp -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1117_train/shallower -n 16 -q h200_core_shared
# bash run_h100.sh config_mae/MAE_resnetGN/1116_shallower/no_relu -n 16 --mae
# bash run_h100.sh configs_new/cache256/1117_train/pixels -n 16 -q h200_lowest
# bash run_h100.sh config_mae/MAE_resnetGN/1116_shallower/dae -n 16 --mae -q h200_core_shared
# bash run_h100.sh configs_new/cache256/1117_train/pixels/pixel_more -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1117_train/no_relu -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1117_train/pixels/to_run -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1117_train/pixels/pixel32 -n 16 -q h200_lowest
# bash run_h100.sh config_mae/MAE_resnetGN/1122_params -n 16 --mae -q h200_lowest
# bash run_h100.sh configs_new/cache256/1117_train/dae -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1117_train/pixels/pixel32/more_sweeps -n 16 
# bash run_h100.sh config_mae/MAE_resnetGN/1122_params/mae_ablate -n 16 --mae -q h200_lowest
# bash run_h100.sh configs_new/cache256/1117_train/ablate_resnet -n 16 -q h200_lowest
# bash run_h100.sh config_mae/MAE_resnetGN/1122_params/pixel_mae -n 16 --mae -q h200_lowest
# bash run_h100.sh configs_new/cache256/1117_train/pixels/to_run -n 16 -q h200_lowest

# pixel MAE
# bash run_h100.sh config_mae/MAE_resnetGN/1126_pixels -n 16 --mae 
# bash run_h100.sh config_mae/MAE_resnetGN/1126_pixels/mae_psz -n 16 --mae -q h200_core_shared
# bash run_h100.sh configs_both/GN_ablate -n 16 --both -q h200_lowest
# bash run_h100.sh configs_both/debug -n 16 --both -q h200_core_shared --num_retries 1
# bash run_h100.sh configs_both/ablates -n 16 --both -q h200_lowest
# bash run_h100.sh configs_new/cache256/1117_train/split -n 16 -q h200_core_shared
# bash run_h100.sh configs_both/to_run -n 16 --both -q h200_lowest
# bash run_h100.sh configs_both/small_ablate -n 16 --both -q h200_lowest
# bash run_h100.sh config_mae/MAE_resnetGN/1205_lr -n 16 --mae
# bash run_h100.sh configs_both/ablates/latent_vs_cache -n 16 --both -q h200_lowest
# bash run_h100.sh configs_both/pixel_fid -n 16 --both -q h200_lowest
# bash run_h100.sh configs_new/cache256/1210_ablations/gen_size -n 16 -q h200_lowest
# bash run_h100.sh configs_new/cache256/1210_ablations/mem_bank -n 16 -q h200_lowest
# bash run_h100.sh configs_both/640_L -n 32 --both -q h200_core_shared
# bash run_h100.sh configs_new/cache256/1210_ablations/544_L -n 32 -q h200_core_shared
# bash run_h100.sh configs_new/cache256/1210_ablations/two_clips -n 16 -q h200_lowest
# bash run_h100.sh config_mae/MAE_resnetGN/1215_cont_640 -n 16 --mae -q h200_lowest