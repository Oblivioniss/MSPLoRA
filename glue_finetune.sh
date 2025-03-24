#!/bin/bash

declare -A epochs=(["mrpc"]=30 ["qnli"]=25 ["rte"]=80 ["sst2"]=60 ["stsb"]=40 ["cola"]=80)
declare -A bs=(["mrpc"]=64 ["qnli"]=64 ["rte"]=64 ["sst2"]=64 ["stsb"]=64 ["cola"]=64)
declare -A ml=(["mrpc"]=256 ["qnli"]=256 ["rte"]=512 ["sst2"]=256 ["stsb"]=256 ["cola"]=256)
declare -A lr=(["mrpc"]="4e-4" ["qnli"]="4e-4" ["rte"]="4e-4" ["sst2"]="5e-4" ["stsb"]="4e-4" ["cola"]="4e-4")
declare -A metrics=(["mrpc"]="accuracy" ["qnli"]="accuracy" ["rte"]="accuracy" ["sst2"]="accuracy" ["stsb"]="pearson" ["cola"]="matthews_correlation")
declare -A target=(["all"]="query value key attention.output.dense intermediate.dense output.dense" ["qv"]="query value")

export WANDB_MODE=offline

run(){
  task_name=$1
  learning_rate=${lr[$1]}
  num_train_epochs=${epochs[$1]}
  per_device_train_batch_size=${bs[$1]}
  gradient_accumulation_steps=1
  rank=$2
  l_num=$3
  seed=$6
  lora_alpha="16"
  target_modules=${target[$4]}
  target_name=$4
  mode=$5
  lora_dropout=0.05
  lora_bias=none
  lora_task_type=SEQ_CLS
  wandb_project=msplora-reproduce
  share=false
  is_train_in_stage=false
  wandb_run_name=roberta-lora-${mode}-${target_name}-${task_name}-stage-${is_train_in_stage}-r-${rank}-n-${l_num}-alpha-16-seed-${seed}-bs-${per_device_train_batch_size}-lr-${learning_rate}-epochs-${num_train_epochs}
  
  exp_dir=./roberta_glue_reproduce_${mode}/${wandb_run_name}

  CUDA_VISIBLE_DEVICES=0 python ./run_glue_lora.py \
  --model_name_or_path FacebookAI/roberta-base  \
  --task_name ${task_name} \
  --do_train --do_eval \
  --max_seq_length ${ml[$1]} \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size ${per_device_train_batch_size} \
  --load_best_model_at_end True --metric_for_best_model ${metrics[$1]} \
  --learning_rate ${learning_rate} \
  --num_train_epochs ${num_train_epochs} \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --weight_decay 0.1 \
  --warmup_ratio 0.06 \
  --logging_steps 10 \
  --seed ${seed} --wandb_project ${wandb_project} \
  --lora_alpha ${lora_alpha} --lora_dropout ${lora_dropout} --lora_bias ${lora_bias} \
  --lora_task_type ${lora_task_type} --target_modules ${target_modules} --rank ${rank} \
  --l_num ${l_num} --mode ${mode} \
  --is_train_in_stage ${is_train_in_stage} \
  --output_dir ${exp_dir}/model \
  --logging_dir ${exp_dir}/log \
  --run_name ${wandb_run_name} \
  --overwrite_output_dir
}
task_base=('mrpc' 'qnli' 'rte' 'sst2' 'stsb' 'cola')
seeds=(42)
ranks=(8)

for task in "${task_base[@]}"; do
  for seed in "${seeds[@]}"; do
    for rank in "${ranks[@]}"; do
      echo "Running with task: $task, seed: $seed, rank: $rank, mode: $mode, l_num: $l_num"
      run "$task" "$rank" "$l_num" "qv" "msp" "$seed"
    done
  done
done
