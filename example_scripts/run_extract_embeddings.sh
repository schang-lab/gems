DATASET="dunning_kruger"

python scripts/preprocessing/run_hidden_extract.py \
 --json data/${DATASET}/${DATASET}_option_strings.json \
 --model meta-llama/Llama-2-7b-hf \
 --out outputs/llm_embeddings \
 --n_workers 1 \
 --layer "all" \
 --extract_position before_eos

python scripts/preprocessing/run_hidden_extract.py \
 --json data/${DATASET}/${DATASET}_option_strings.json \
 --model mistralai/Mistral-7B-v0.1 \
 --out outputs/llm_embeddings \
 --n_workers 1 \
 --layer "all" \
 --extract_position before_eos

python scripts/preprocessing/run_hidden_extract.py \
 --json data/${DATASET}/${DATASET}_option_strings.json \
 --model Qwen/Qwen3-8B-Base \
 --out outputs/llm_embeddings \
 --n_workers 1 \
 --layer "all" \
 --extract_position before_eos