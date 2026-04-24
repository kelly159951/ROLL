#!/bin/bash
set +x
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

if [ ! -d "./models/Qwen3-8B-Base" ]; then
  echo "Qwen3-8B-Base local model directory not found: ./models/Qwen3-8B-Base"
  echo "Place the model snapshot at ./models/Qwen3-8B-Base."
  exit 1
fi

CONFIG_PATH=$(basename $(dirname $0))
PYTHONPATH="$(pwd):${PYTHONPATH}" python examples/start_rlvr_pipeline.py --config_path $CONFIG_PATH --config_name rlvr_config_4gpu_wandb_entropy_top20
