#!/bin/bash
# This script is used to finetune the pretrained MCAN model.

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2;;
    --task)
      TASK="$2"
      shift 2;;
    --pretrained_model)
      PRETRAINED_MODEL_PATH="$2"
      shift 2;;
    --version)
      VERSION="$2"
      shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

TASK=${TASK:-ok} # task name, one of ['ok', 'aok_val', 'aok_test'], default 'ok'
GPU=${GPU:-0} # GPU id(s) you want to use, default '0'
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"Visual-Question-Answering/outputs/beit_pretrain_1/ckpts/mp_rank_00_model_states.pt"} # path to the pretrained model, default is the result from our experiments
VERSION=${VERSION:-finetuning_${TASK}} # version name, default 'finetuning_for_$TASK'

# run python script
# CUDA_VISIBLE_DEVICES=$GPU \
python main.py \
    --task $TASK \
    --cfg configs/finetune.yml \
    --version $VERSION \
    --pretrained_model $PRETRAINED_MODEL_PATH \
    --gpu $GPU --seed 99 --grad_accu 2 --finetune True --run_mode finetune
