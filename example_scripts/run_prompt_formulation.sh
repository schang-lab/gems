DATASET="dunning_kruger"
TOPKS=(3 0 3)
EMBEDDING_MODEL="gemini-embedding-001"
SEED=42
SPLIT_NAMES=(
  "individual_val0p05_test0p60_evalpartial_0p40"
  "individual_val0p05_test0p60_evalpartial_0p00"
  "question_val0p10_test0p20_evalpartial_0p00"
)

for i in "${!SPLIT_NAMES[@]}"; do
    SPLIT_NAME="${SPLIT_NAMES[$i]}"
    TOPK="${TOPKS[$i]}"
    echo "Processing split: $SPLIT_NAME with top_k=$TOPK"
    python scripts/preprocessing/run_prompt_formulation.py \
        --dataset "$DATASET" \
        --top_k "$TOPK" \
        --text_embedding_path "outputs/text_embeddings/${DATASET}_text_embeddings_${EMBEDDING_MODEL}.pth" \
        --split_path "outputs/dataset_splits/${DATASET}_${SPLIT_NAME}_seed${SEED}.jsonl"
done