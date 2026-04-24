#!/bin/bash
set +x
export WANDB_MODE=offline

CONFIG_PATH=$(basename $(dirname $0))
PYTHONPATH="$(pwd):${PYTHONPATH}" python examples/start_rlvr_pipeline.py --config_path $CONFIG_PATH --config_name rlvr_config_4gpu_wandb_baseline
