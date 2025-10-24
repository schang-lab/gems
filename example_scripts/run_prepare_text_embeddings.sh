DATASET="dunning_kruger"
EMBEDDING_MODEL="gemini-embedding-001"

python scripts/preprocessing/run_text_embedding.py \
 --json data/${DATASET}/${DATASET}_option_strings.json \
 --model_name ${EMBEDDING_MODEL}