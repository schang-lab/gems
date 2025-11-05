DATASET="dunning_kruger"
SEED=42

# setting 1: missing responses (i.e., imputation)
# 60% test, 5% val, 35% train individuals
# additionally, for each individual in val/test, 40% of their responses are provided during train
python scripts/preprocessing/run_dataset_split.py \
 --dataset $DATASET \
 --split_axis individual \
 --test_ratio 0.60 \
 --val_ratio 0.05 \
 --eval_partial_ratio 0.40 \
 --seed ${SEED}

# setting 2: new individuals
# 60% test, 5% val, 35% train individuals
python scripts/preprocessing/run_dataset_split.py \
 --dataset $DATASET \
 --split_axis individual \
 --test_ratio 0.60 \
 --val_ratio 0.05 \
 --eval_partial_ratio 0.00 \
 --seed ${SEED}

# setting 3: new questions
# 20% test, 10% val, 70% train questions
python scripts/preprocessing/run_dataset_split.py \
 --dataset $DATASET \
 --split_axis question \
 --test_ratio 0.20 \
 --val_ratio 0.10 \
 --eval_partial_ratio 0.00 \
 --seed ${SEED}