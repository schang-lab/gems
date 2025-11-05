#!/bin/bash

DATASET_NAME="dunning_kruger" # opinionqa, twin, dunning_kruger
GNN_ARCH="sage" # gat, rgcn, sage
LLM_MODEL_NICKNAME="llama2_7b" # llama2_7b, mistral_7b, qwen3_8b
OUTPUT_EMBEDDING_FILENAME=YOUR_OUTPUT_EMBEDDING_FILENAME # from scripts/graph/train.py
SEED=42 # seed used when splitting dataset. must be identical to the dataset split used during GNN training

SPLIT_FILENAME="question_val0p10_test0p20_evalpartial_0p00" # only used for setting 3 (new questions)

GNN_EMBEDDING_PATH="outputs/gems_embeddings/gems_training_${GNN_ARCH}/${OUTPUT_EMBEDDING_FILENAME}.pth"
LLM_EMBEDDING_PATH="outputs/llm_embeddings/${DATASET_NAME}_option_strings_embedding_${LLM_MODEL_NICKNAME}_layer_all_eos_False.pt"
SPLIT_FILEPATH="outputs/dataset_splits/${DATASET_NAME}_${SPLIT_FILENAME}_seed${SEED}.jsonl"

python scripts/graph/llm_to_gnn_mapping.py \
  --dataset_name="${DATASET_NAME}" \
  --gnn_embedding_path="${GNN_EMBEDDING_PATH}" \
  --llm_embedding_path="${LLM_EMBEDDING_PATH}" \
  --split_filepath="${SPLIT_FILEPATH}" \
  --verbose
  