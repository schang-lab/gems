import argparse
import json
import pathlib
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Literal

import torch
import numpy as np

from sibyl.graph.node import NodeCollection
from sibyl.config.dataset_map import DATASET_MAP

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def retreive_llm_embeddings(llm_embedding_dict: Dict,
                            choice_keys: List[str],
                            target_layer_idx: int) -> torch.Tensor:
    embeddings_list = [
        llm_embedding_dict[ck][target_layer_idx].to(torch.float32)
        for ck in choice_keys
    ]
    return torch.stack(embeddings_list)


def retrieve_gnn_embeddings(gnn_embedding_dict: Dict,
                            keys: List[str],
                            search_for: Literal['indiv', 'question']) -> torch.Tensor:
    assert search_for in ['indiv', 'question']
    embeddings_list = []
    for ck in keys:
        for wref in gnn_embedding_dict['per_wave'].values():
            lookup = wref['post_gnn'][search_for] # GNN output embedding dictionary
            if ck in lookup:
                embeddings_list.append(lookup[ck])
                break            
    assert len(embeddings_list) == len(keys), "Some keys not found"
    return torch.stack(embeddings_list)


class RidgeCVModel:
    def __init__(
        self,
        coef_np: np.ndarray,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        alpha: float,
        standardize_X: bool,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.coef_np = coef_np          # (d_in, d_out) in numpy (float64)
        self.x_mean = x_mean.reshape(1, -1)  # (1, d_in)
        self.x_std = x_std.reshape(1, -1)    # (1, d_in)
        self.y_mean = y_mean.reshape(1, -1)  # (1, d_out)
        self.alpha = float(alpha)
        self.standardize_X = bool(standardize_X)
        self.dtype = dtype
        self.device = device

        # Torch views for convenient prediction on torch tensors
        self.coef = torch.from_numpy(self.coef_np).to(dtype=self.dtype, device=self.device)

    def _standardize(self, X_np: np.ndarray) -> np.ndarray:
        if self.standardize_X:
            return (X_np - self.x_mean) / self.x_std
        else:
            return (X_np - self.x_mean)  # center only

    def predict(self, X_new: torch.Tensor) -> torch.Tensor:
        """
        Predict Y for new data using stored train statistics.
        """
        X_np = X_new.detach().cpu().numpy().astype(np.float64, copy=False)
        Xp = self._standardize(X_np)            # (n, d_in)
        Yhat_np = Xp @ self.coef_np + self.y_mean  # (n, d_out)
        Yhat = torch.from_numpy(Yhat_np).to(dtype=self.dtype, device=X_new.device)
        return Yhat


def _fit_ridge_svd_centered(Xc: np.ndarray, Yc: np.ndarray, alpha: float) -> np.ndarray:
    """
    Solve ridge for centered targets: Yc ≈ Xproc @ W.
    Returns W (d_in, d_out). Uses SVD for stability.
    """
    # Xc can be standardized or just centered; both are fine.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # Xc = U S V^T
    # Ridge pseudoinverse piece: V diag(S/(S^2 + alpha)) U^T
    denom = (S * S) + float(alpha)
    factors = (S / denom).reshape(-1, 1)               # (r,1)
    W = (Vt.T * factors.T) @ (U.T @ Yc)                # (d_in, d_out)
    return W


def ridge_cv_torch(
    X: torch.Tensor,
    Y: torch.Tensor,
    alphas: Union[Iterable[float], float],
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    standardize_X: bool = True,
) -> Dict[str, Any]:
    """
    Ridge regression with K-fold CV and optional feature standardization.
    - Standardization and centering are computed on each TRAIN fold only.
    - Targets are centered (no scaling), so intercept is unpenalized.
    - If n_splits == 1, no cross-validation is performed and a single model is
      fit using the provided alpha. In that case, `alphas` must be either a
      single float or an iterable containing exactly one value.
    Returns:
        dict with:
            - "model": a fitted RidgeCVModel
            - "alpha": the selected alpha (float)
            - "cv_scores": list of (alpha, mean_mse, [per-fold MSE]); empty if n_splits == 1
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D tensors.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    n = X.shape[0]
    d_in = X.shape[1]
    d_out = Y.shape[1]

    # Convert to NumPy float64 for numeric stability
    Xnp = X.detach().cpu().numpy().astype(np.float64, copy=False)
    Ynp = Y.detach().cpu().numpy().astype(np.float64, copy=False)

    # Helper to coerce `alphas` to a list of floats while supporting the scalar case
    def _coerce_alphas(a_in: Union[Iterable[float], float]) -> List[float]:
        if isinstance(a_in, (float, int, np.floating, np.integer)):
            return [float(a_in)]
        try:
            return [float(a) for a in a_in]  # type: ignore
        except TypeError:
            raise TypeError(
                "alphas must be an iterable of floats, or a single float when n_splits == 1"
            )

    alphas_list: List[float] = _coerce_alphas(alphas)
    if len(alphas_list) == 0:
        raise ValueError("alphas must be a non-empty iterable of nonnegative floats (or a single float)")
    if any(a < 0 for a in alphas_list):
        raise ValueError("alphas must be nonnegative")

    # If n_splits == 1, enforce a single alpha and train once on full data
    if n_splits == 1:
        if len(alphas_list) != 1:
            raise ValueError("When n_splits == 1, `alphas` must contain exactly one value (or be a single float).")
        best_alpha = float(alphas_list[0])

        # Compute full-data stats and fit
        x_mean_full = Xnp.mean(axis=0, keepdims=True)
        if standardize_X:
            x_std_full = Xnp.std(axis=0, ddof=0, keepdims=True)
            x_std_full = np.where(x_std_full == 0.0, 1.0, x_std_full)
            Xp_full = (Xnp - x_mean_full) / x_std_full
        else:
            x_std_full = np.ones_like(x_mean_full)
            Xp_full = (Xnp - x_mean_full)

        y_mean_full = Ynp.mean(axis=0, keepdims=True)
        Yc_full = Ynp - y_mean_full

        W_full = _fit_ridge_svd_centered(Xp_full, Yc_full, best_alpha)

        model = RidgeCVModel(
            coef_np=W_full,
            x_mean=x_mean_full,
            x_std=x_std_full,
            y_mean=y_mean_full,
            alpha=best_alpha,
            standardize_X=standardize_X,
            dtype=X.dtype,
            device=X.device,
        )
        return {
            "model": model,
            "alpha": float(best_alpha),
            "cv_scores": [],  # no CV performed
        }

    # ====== K-fold CV path (n_splits >= 2) ======

    # Prepare folds
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
    folds: List[np.ndarray] = np.array_split(idx, n_splits)

    def prepare_XY_train(
        Xtr: np.ndarray, Ytr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_mean = Xtr.mean(axis=0, keepdims=True)
        if standardize_X:
            x_std = Xtr.std(axis=0, ddof=0, keepdims=True)
            # avoid division by zero on constant features
            x_std = np.where(x_std == 0.0, 1.0, x_std)
            Xp = (Xtr - x_mean) / x_std
        else:
            x_std = np.ones_like(x_mean)
            Xp = (Xtr - x_mean)  # center only

        y_mean = Ytr.mean(axis=0, keepdims=True)
        Yc = Ytr - y_mean
        return Xp, Yc, x_mean, x_std, y_mean

    cv_results: List[Tuple[float, float, List[float]]] = []
    best_alpha: Optional[float] = None
    best_score = np.inf

    # Cross-validation loop
    for a in alphas_list:
        fold_mse: List[float] = []
        for k in range(n_splits):
            val_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(n_splits) if j != k], axis=0)

            Xtr, Ytr = Xnp[tr_idx], Ynp[tr_idx]
            Xva, Yva = Xnp[val_idx], Ynp[val_idx]

            Xp_tr, Yc_tr, x_mean_tr, x_std_tr, y_mean_tr = prepare_XY_train(Xtr, Ytr)
            W = _fit_ridge_svd_centered(Xp_tr, Yc_tr, a)

            # Validation transform using TRAIN stats only
            if standardize_X:
                Xp_va = (Xva - x_mean_tr) / x_std_tr
            else:
                Xp_va = (Xva - x_mean_tr)

            Yhat_va = Xp_va @ W + y_mean_tr
            fold_mse.append(_mse(Yva, Yhat_va))

        mean_mse = float(np.mean(fold_mse))
        cv_results.append((a, mean_mse, fold_mse))
        if mean_mse < best_score:
            best_score = mean_mse
            best_alpha = a

    # Fit on full data with the best alpha and store final train statistics
    assert best_alpha is not None
    x_mean_full = Xnp.mean(axis=0, keepdims=True)
    if standardize_X:
        x_std_full = Xnp.std(axis=0, ddof=0, keepdims=True)
        x_std_full = np.where(x_std_full == 0.0, 1.0, x_std_full)
        Xp_full = (Xnp - x_mean_full) / x_std_full
    else:
        x_std_full = np.ones_like(x_mean_full)
        Xp_full = (Xnp - x_mean_full)

    y_mean_full = Ynp.mean(axis=0, keepdims=True)
    Yc_full = Ynp - y_mean_full
    W_full = _fit_ridge_svd_centered(Xp_full, Yc_full, best_alpha)

    model = RidgeCVModel(
        coef_np=W_full,
        x_mean=x_mean_full,
        x_std=x_std_full,
        y_mean=y_mean_full,
        alpha=best_alpha,
        standardize_X=standardize_X,
        dtype=X.dtype,
        device=X.device,
    )

    return {
        "model": model,
        "alpha": float(best_alpha),
        "cv_scores": cv_results,  # list of (alpha, mean_mse, [per-fold MSE])
    }


def main(
    dataset_name: str,
    gnn_embedding_dict: Dict, llm_embedding_dict: Dict,
    predefined_split: Dict,
) -> None:

    dataset_info = DATASET_MAP[dataset_name]
    global_nodecollection = NodeCollection(
        source=dataset_info['source'],
        basepath=REPO_ROOT / "data" / dataset_name,
        qkeypath=dataset_info['qkeypath'],
        qstringpath=dataset_info['qstringpath'],
        valid_traits=dataset_info['valid_traits'],
        use_selective_subgroups=dataset_info['use_selective_subgroups'],
        min_questions_per_indiv=dataset_info['min_questions_per_indiv'],
        dataset_name=dataset_name,
    )

    # n_layers is the number of layers in transformer-based LLM
    _key = list(llm_embedding_dict.keys())[0]
    n_layers = llm_embedding_dict[_key].shape[0]

    # construct a list of question keys (e.g., "Q1_W1")
    # and choice keys (e.g., "Q1_W1_option_1")
    choices_dict = json.load(open(
        os.path.join(
            REPO_ROOT, "data",
            dataset_name, f"{dataset_name}_option_strings.json"
        ), 'r')
    )
    choice_keys = sorted(list(choices_dict.keys()))
    question_keys = sorted(list(set([ck.split('_option_')[0] for ck in choice_keys])))

    # construct the set of questions appearing in each of train/val/test splits
    per_split_q = {split_name: set() for split_name in predefined_split}
    for split_name in predefined_split: # train/val/test
        for indiv_response in predefined_split[split_name].values():
            response_qkeys = set(indiv_response.keys())
            per_split_q[split_name].update(response_qkeys)
    assert (per_split_q['train'].isdisjoint(per_split_q['val'])
            and per_split_q['train'].isdisjoint(per_split_q['test'])
            and per_split_q['val'].isdisjoint(per_split_q['test'])), (
        "Train/Val/Test splits have overlapping questions!"
    )
    print("Number of questions per split:")
    for split_name in per_split_q:
        print(f"  {split_name}: {len(per_split_q[split_name])}")

    # construct the set of choices appearing in each of train/val/test splits
    per_split_c = {
        split_name: sorted([
            ck for ck in choice_keys
            if any(ck.startswith(qk) for qk in per_split_q[split_name])
        ])
        for split_name in per_split_q
    }
    per_split_q_to_c_map, per_split_c_to_i_map = {}, {}
    for split_name in per_split_c:
        per_split_q_to_c_map[split_name], per_split_c_to_i_map[split_name] = {}, {}
        for _idx, ck in enumerate(per_split_c[split_name]):
            qk = ck.split('_option_')[0]
            per_split_q_to_c_map[split_name].setdefault(qk, []).append(ck)
            per_split_c_to_i_map[split_name][ck] = _idx

    # val(test)_edges: List of (indiv, qkey, true_human_choice)
    eval_edges = {'val': [], 'test': []}
    for indiv, v in global_nodecollection.individuals.items():
        respond_info = v.respond_info
        for _qkey, _choice in respond_info.items():
            if _qkey in per_split_q['val']:
                eval_edges['val'].append((str(int(indiv)), _qkey, int(_choice)))
            elif _qkey in per_split_q['test']:
                eval_edges['test'].append((str(int(indiv)), _qkey, int(_choice)))

    # additionally, prepare the gnn output embeddings for individuals in eval set
    eval_indivs_embeddings = {
        'val' : retrieve_gnn_embeddings(
            gnn_embedding_dict,
            [edge[0] for edge in eval_edges['val']], search_for='indiv'
        ),
        'test' : retrieve_gnn_embeddings(
            gnn_embedding_dict,
            [edge[0] for edge in eval_edges['test']], search_for='indiv'
        )
    }

    # alpha determined by validation edges accuracy, and the range of alpha can be modified    
    best_val_acc_global, test_acc_at_best_val_global = -1, -1
    for alpha in [50.0, 100.0, 200.0, 400.0, 800.0]:
        print(f"--> Evaluating for regularization strength: {alpha} ")
        best_val_acc, test_acc_at_best_val = -1, -1
        best_layer = -1
        for target_layer_idx in range(1, n_layers):
            print(f" --> Evaluating for LLM layer: {target_layer_idx}")
            X_train = retreive_llm_embeddings(
                llm_embedding_dict,
                per_split_c['train'], target_layer_idx
            )
            Y_train = retrieve_gnn_embeddings(
                gnn_embedding_dict,
                per_split_c['train'], search_for='question'
            )
            result = ridge_cv_torch(X_train, Y_train, alphas=[alpha], n_splits=1)
            model = result["model"] # model.coef is (d_llm, d_gnn_output) tensor

            for split_name in ['val', 'test']:
                X_eval = retreive_llm_embeddings(
                    llm_embedding_dict,
                    per_split_c[split_name], target_layer_idx
                )
                Y_eval_pred = model.predict(X_eval) # (n_eval_choice, d_gnn_output)
                eval_scores = eval_indivs_embeddings[split_name] @ Y_eval_pred.T # (n_eval_indiv, n_eval_choice)

                # Evaluate accuracy on validation edges
                total_cnt = 0
                correct_cnt = 0
                for idx, edge in enumerate(eval_edges[split_name]):
                    indiv, qkey, true_val = edge
                    ckeys = per_split_q_to_c_map[split_name][qkey]
                    ckeys_idx = [per_split_c_to_i_map[split_name][ck] for ck in ckeys]
                    scores = eval_scores[idx, ckeys_idx]
                    pred_idx = torch.argmax(scores).item()
                    pred = int(ckeys[pred_idx].split('_option_')[-1])
                    correct_cnt += int(pred == true_val)
                    total_cnt += 1

                accuracy = correct_cnt / total_cnt if total_cnt > 0 else -1
                print(f" --> {split_name} accuracy: {accuracy:.4f}")
                if split_name == "val" and accuracy > best_val_acc:
                    best_val_acc = accuracy
                    best_layer = target_layer_idx
                if split_name == "test" and best_layer == target_layer_idx:
                    test_acc_at_best_val = accuracy
        
        print(f"   --> Best layer for alpha {alpha}: {best_layer}")
        print(f"   --> Best val accuracy for alpha {alpha}: {best_val_acc:.4f}")
        print(f"   --> Test accuracy at best val for alpha {alpha}: {test_acc_at_best_val:.4f}")
        if best_val_acc > best_val_acc_global:
            best_val_acc_global = best_val_acc
            test_acc_at_best_val_global = test_acc_at_best_val

    print(f"==> Overall best val accuracy: {best_val_acc_global:.4f}")
    print(f"==> Overall test accuracy at best val: {test_acc_at_best_val_global:.4f}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",
                        type=str, required=True,
                        help="name of the dataset. e.g., opinionqa, twin, dunning_kruger")
    parser.add_argument("--gnn_embedding_path",
                        type=str, required=True,
                        help="an output from scripts/graph/train.py.")
    parser.add_argument("--llm_embedding_path",
                        type=str, required=True,
                        help="an output from scripts/preprocessing/run_hidden_extract.py.")
    parser.add_argument("--split_filepath",
                        type=str, required=True,
                        help=("jsonl file that specifies dataset splits. Must be consistent with "
                              "sibyl/config/graph/gems_training_config.yaml - dataset - split_filepath."))
    args = parser.parse_args()
    gnn_embedding_dict = torch.load(args.gnn_embedding_path,
                                    map_location='cpu', weights_only=False)
    llm_embedding_dict = torch.load(args.llm_embedding_path,
                                    map_location='cpu', weights_only=False)
    predefined_split = json.load(open(args.split_filepath, 'r'))
    main(dataset_name=args.dataset_name,
         gnn_embedding_dict=gnn_embedding_dict,
         llm_embedding_dict=llm_embedding_dict,
         predefined_split=predefined_split)
