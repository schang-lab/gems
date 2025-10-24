### Setting 1 (missing responses)
python scripts/graph/train.py \
    dataset.split_filepath="outputs/dataset_splits/opinionqa_question_val0p05_test0p60_evalpartial_0p40_seed42.jsonl" \
    gnn.gnn_arch="gat" \
    gnn.add_self_loops=true \
    split_info.transductive.train.indiv_question=0.5 \
    split_info.inductive.train.indiv=0.0 \
    training.seed=42


### Setting 2 (new individuals)
# Note that in this case, split info (i.e. split between message-passing and supervision edges
# during the training phase) is different
# compared to the setting 1 and setting 3;
# in each training step, 50% of individual nodes are selected, their response edges are masked.
python scripts/graph/train.py \
    dataset.split_filepath="outputs/dataset_splits/opinionqa_question_val0p05_test0p60_evalpartial_0p00_seed42.jsonl" \
    gnn.gnn_arch="gat" \
    gnn.add_self_loops=true \
    split_info.transductive.train.indiv_question=0.0 \
    split_info.inductive.train.indiv=0.5 \


### Setting 3 (new questions)
# Note that after running this script, there will be an GNN output node embedding file
# at outputs/gems_embeddings.
# This file will be used in scripts/graph/llm_to_gnn_mapping.py
# to map LLM representations of new questions to GNN embeddings, and make predictions.
python scripts/graph/train.py \
    dataset.split_filepath="outputs/dataset_splits/opinionqa_question_val0p10_test0p20_evalpartial_0p00_seed42.jsonl" \
    experiment.save_embedding=true \
    split_info.new_question.is_this=true \
    split_info.transductive.train.indiv_question=0.5 \
    split_info.inductive.train.indiv=0.0