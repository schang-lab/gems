from __future__ import annotations
import argparse
import os
import json
import pathlib

import numpy as np

from sibyl.graph.node import NodeCollection
from sibyl.config.dataset_map import DATASET_MAP

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
SAVING_FORMAT = "{dataset}_{split_axis}_val{val_ratio_str}_test{test_ratio_str}_evalpartial_{eval_partial_ratio_str}_seed{seed}.jsonl"

def main():

    parser = argparse.ArgumentParser(description="Run dataset split.")
    parser.add_argument("--dataset", type=str,
                        choices=DATASET_MAP.keys(), required=True,
                        help="Input dataset.")
    parser.add_argument("--split_axis", type=str,
                        choices=["question", "individual"], default="individual",
                        help="Either split by individual (setting 1, 2) or by question (setting 3).")
    parser.add_argument("--test_ratio", type=float,
                        default=0.60,
                        help="Proportion of the test along split_axis.")
    parser.add_argument("--val_ratio", type=float,
                        default=0.05,
                        help="Proportion of the validation along split_axis.")
    parser.add_argument("--eval_partial_ratio", type=float,
                        default=0.00,
                        help="(only for setting 1) Proportion of the eval set incorporated back to train set.")
    parser.add_argument("--seed", type=int,
                        default=42,
                        help="Random seed for split sampling.")
    args = parser.parse_args()
    
    dataset_info = DATASET_MAP[args.dataset]
    individual_dict = NodeCollection(
        source=dataset_info['source'],
        basepath=DATA_DIR / args.dataset,
        qkeypath=dataset_info['qkeypath'],
        qstringpath=dataset_info['qstringpath'],
        valid_traits=dataset_info['valid_traits'],
        use_selective_subgroups=dataset_info['use_selective_subgroups'],
        min_questions_per_indiv=dataset_info['min_questions_per_indiv'],
        dataset_name=args.dataset,
    ).individuals

    val_ratio, test_ratio = args.val_ratio, args.test_ratio
    train_ratio = 1 - val_ratio - test_ratio
    assert train_ratio > 0, "Sum of val_ratio and test_ratio must be less than 1."
    assert 0 <= args.eval_partial_ratio < 1, "eval_partial_ratio must be in [0, 1)."
    
    np.random.seed(args.seed)
    
    train_response, val_response, test_response = {}, {}, {}
    if args.split_axis == "individual":
        indiv_keys = list(individual_dict.keys())
        np.random.shuffle(indiv_keys)
        n_indiv = len(indiv_keys)
        n_val, n_test = int(n_indiv * val_ratio), int(n_indiv * test_ratio)
        test_indivs, val_indivs, train_indivs = (
            indiv_keys[:n_test],
            indiv_keys[n_test:n_test+n_val],
            indiv_keys[n_test+n_val:]
        )
        for indiv in train_indivs:
            train_response[indiv] = individual_dict[indiv].respond_info
        for indiv in val_indivs + test_indivs:
            _tmp = individual_dict[indiv].respond_info
            _qkeys = list(_tmp.keys())
            np.random.shuffle(_qkeys)
            _n_train = int(len(_qkeys) * args.eval_partial_ratio)
            train_response[indiv] = {k: _tmp[k] for k in _qkeys[:_n_train]}
            if indiv in val_indivs:
                val_response[indiv] = {k: _tmp[k] for k in _qkeys[_n_train:]}
            else:
                test_response[indiv] = {k: _tmp[k] for k in _qkeys[_n_train:]}
    
    elif args.split_axis == "question":
        all_qkeys = set()
        for indiv in individual_dict.values():
            all_qkeys.update(indiv.respond_info.keys())
        all_qkeys = list(all_qkeys)
        np.random.shuffle(all_qkeys)
        n_q = len(all_qkeys)
        n_val, n_test = int(n_q * val_ratio), int(n_q * test_ratio)
        test_qkeys, val_qkeys, train_qkeys = (
            set(all_qkeys[:n_test]),
            set(all_qkeys[n_test:n_test+n_val]),
            set(all_qkeys[n_test+n_val:])
        )
        for indiv, indiv_obj in individual_dict.items():
            for qkey, resp in indiv_obj.respond_info.items():
                if qkey in train_qkeys:
                    train_response.setdefault(indiv, {}).update({qkey: resp})
                elif qkey in val_qkeys:
                    val_response.setdefault(indiv, {}).update({qkey: resp})
                elif qkey in test_qkeys:
                    test_response.setdefault(indiv, {}).update({qkey: resp})
                else:
                    raise ValueError("Question key not found in any split sets.")

    saving_format = SAVING_FORMAT.format(
        dataset=args.dataset,
        split_axis=args.split_axis,
        val_ratio_str=f"{val_ratio:.2f}".replace(".", "p"),
        test_ratio_str=f"{test_ratio:.2f}".replace(".", "p"),
        eval_partial_ratio_str=f"{args.eval_partial_ratio:.2f}".replace(".", "p"),
        seed=args.seed
    )
    print(f"--> Saving saming filename: {saving_format}")
    BASE_PATH = ROOT_DIR / "outputs" / "dataset_splits"
    os.makedirs(BASE_PATH, exist_ok=True)
    with open(BASE_PATH / saving_format, 'w') as f:
        json.dump({
            "train": train_response,
            "val": val_response,
            "test": test_response,
        }, f, indent=4)


if __name__ == "__main__":
    main()