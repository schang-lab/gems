#!/bin/bash

MODEL_NAMES=(
  "Qwen/Qwen3-8B"
  "mistralai/Mistral-7B-Instruct-v0.1"
  "meta-llama/Llama-2-7b-chat-hf"
)
splits=(
  "individual_val0p05_test0p60_evalpartial_0p40"
  "individual_val0p05_test0p60_evalpartial_0p00"
  "question_val0p10_test0p20_evalpartial_0p00"
)
topks=(0 3 8)
seeds=(41 42 43)
dataset_name="dunning_kruger"
N_GPUS=1

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  for split in "${splits[@]}"; do
    for topk in "${topks[@]}"; do
      for seed in "${seeds[@]}"; do
        input="outputs/llm_prompts/${dataset_name}_${split}_seed${seed}_topk_${topk}_test.jsonl"
        echo "Running: ${input}"
        python scripts/llm/run_agentic_cot.py \
          --input_path "$input" \
          --base_model_name_or_path "$MODEL_NAME" \
          --is_chat \
          --tp_size $N_GPUS \
          --gpu-memory-utilization 0.9
      done
    done
  done
done