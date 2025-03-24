#!/bin/bash
export WANDB_MODE=offline
gpu=0

run(){
  bs=128
  micro_bs=4
  learning_rate='3e-4'
  num_train_epochs=3
  mode=$1
  rank=$2
  l_num=$3
  seed=$4
  is_train_in_stage=false
  lora_alpha="16"
  target_name='qv'
  lora_dropout=0.05
  lora_bias=none
  cutoff_len=256
  wandb_project=proejct_name
  wandb_run_name=llama-msplora-${target_name}-${mode}-stage-${is_train_in_stage}-r-${rank}-n-${l_num}-alpha-16-seed-${seed}-bs-${bs}-lr-${learning_rate}-len-${cutoff_len}-epochs-${num_train_epochs}
  echo $wandb_run_name
  exp_dir=./llama-lora-${mode}/${wandb_run_name}
  mkdir -p $exp_dir
  
  CUDA_VISIBLE_DEVICES=$gpu python llama_finetune.py \
    --base_model= meta-llama/Llama-2-7b-hf \
    --cutoff_len=$cutoff_len \
    --mode=$mode \
    --seed=$seed \
    --group_by_length \
    --lora_r=$rank \
    --lora_n=$l_num \
    --lora_alpha=$lora_alpha \
    --lora_dropout=$lora_dropout \
    --lora_target_modules='[q_proj,v_proj]' \
    --batch_size=$bs \
    --is_train_in_stage=$is_train_in_stage \
    --micro_batch_size=$micro_bs \
    --num_epochs=$num_train_epochs \
    --learning_rate=$learning_rate \
    --wandb_project=$wandb_project \
    --wandb_run_name=$wandb_run_name \
    --output_dir=${exp_dir}/model
}

seeds=(42)
l_nums=(3)

# run LoRA with rank 64, seed 42
#run 'base' 64 1 42

# run MSPLoRA with rank 64, lora_num 3, seed 42
#run 'msplora' 64 3 42
