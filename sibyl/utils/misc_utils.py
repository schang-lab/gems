import json
from collections import Counter
from typing import Dict, List, Any, Union, Literal, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, t


def list_normalize(l: List[float]) -> List[float]:

    if np.isclose(sum(l), 0):
        raise ValueError("--> list_normalize: sum of list is 0.")
    return [i / sum(l) for i in l]


def ordinal_emd(
    list_1: List[float],
    list_2: List[float],
    ordinal: List[float],
) -> float:

    assert len(list_1) == len(list_2), "-->ordinal_emd(): list_1 and list_2 should be same length."
    assert len(list_1) == len(ordinal), "-->ordinal_emd(): list_1 and ordinal should be same length."
    if max(ordinal) == min(ordinal):
        return np.nan

    ordinal, list_1, list_2 = zip(*sorted(zip(ordinal, list_1, list_2)))
    non_neg_idx = next((i for i, val in enumerate(ordinal) if val >= 0), 0)
    ordinal = ordinal[non_neg_idx:]
    list_1 = list_normalize(list_1[non_neg_idx:])
    list_2 = list_normalize(list_2[non_neg_idx:])

    cum_dist_1 = np.cumsum(list_1)
    cum_dist_2 = np.cumsum(list_2)
    emd = 0
    for i in range(len(list_1) - 1):
        emd += abs(cum_dist_1[i] - cum_dist_2[i]) * (
            ordinal[i + 1] - ordinal[i]
        )
    return emd / (max(ordinal) - min(ordinal))


def js_divergence(list_1: List[float], list_2: List[float]) -> float:

    assert len(list_1) == len(list_2), "-->js_divergence(): list_1 and list_2 should be same length."
    list_1 = np.array(list_normalize(list_1), dtype=float)
    list_2 = np.array(list_normalize(list_2), dtype=float)
    return jensenshannon(list_1, list_2, base=2)**2


def joint_counts(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x.astype(int)
    y = y.astype(int)
    n_x = x.max() + 1
    n_y = y.max() + 1
    return np.bincount(x * n_y + y, minlength=n_x * n_y)


def normalized_entropy(x: np.ndarray, base: float = 2., norm_off: bool = False) -> float:
    counts = Counter(x)
    counts = np.array(list(counts.values()), dtype=float)
    n_x = counts.shape[0]
    h = entropy(counts, base=base)
    if norm_off:
        return h
    return h / (np.log(n_x) / np.log(base)) if n_x > 1 else 0.0


def normalized_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    norm: Literal['sqrt', 'min', 'max', 'avg'] = 'sqrt',
    *,
    base: float = 2.,
    return_ci: bool = False,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: Optional[np.random.Generator] = None
) -> Tuple[float, Optional[Tuple[float, float]]]:

    assert x.shape == y.shape
    assert norm in ['sqrt', 'min', 'max', 'avg'], f"Unknown norm '{norm}'."

    def row_hash(row: np.ndarray) -> int:
        return row[0] * (y.max() + 1) + row[1]

    H_x = normalized_entropy(x, base=base, norm_off=True)
    H_y = normalized_entropy(y, base=base, norm_off=True)
    H_xy = normalized_entropy(
        np.apply_along_axis(row_hash, 1, np.column_stack((x, y))),
        base=base,
        norm_off=True
    )
    I_xy = H_x + H_y - H_xy

    if norm == 'sqrt':
        nmi = I_xy / np.sqrt(H_x * H_y) if H_x > 0 and H_y > 0 else 0.0
    elif norm == 'avg':
        nmi = (2 * I_xy) / (H_x + H_y) if H_x + H_y else 0.0
    elif norm == 'min':
        nmi = I_xy / min(H_x, H_y) if min(H_x, H_y) else 0.0
    elif norm == 'max':
        nmi = I_xy / max(H_x, H_y) if max(H_x, H_y) else 0.0

    if not return_ci:
        return nmi, None

    rng = random_state or np.random.default_rng()
    stats = np.empty(n_boot, dtype=float)
    idx = np.arange(len(x))
    for b in range(n_boot):
        sample = rng.choice(idx, size=idx.size, replace=True)
        stats[b], _ = normalized_mutual_information(
            x[sample], y[sample],
            norm=norm, base=base, return_ci=False
        )
    alpha = 1.0 - ci
    lo, hi = np.percentile(stats, [100*alpha/2, 100*(1-alpha/2)])
    return nmi, (lo, hi)


def info_metrics(arr_2d: np.ndarray, **kwargs):
    x, y = arr_2d[:, 0], arr_2d[:, 1]
    nmi, ci = normalized_mutual_information(x, y, **kwargs)
    return {
        "H_norm_X": normalized_entropy(x),
        "H_norm_Y": normalized_entropy(y),
        "NMI": nmi,
        "NMI_CI": ci,
    }


def convert_for_json(obj: Any) -> Any:
    if isinstance(obj, np.float64) or isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, list):
        return [convert_for_json(i) for i in obj]
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    return obj


def fetch_statistics(filepath: Union[str, Path],
                     field: str,
                     stattype: str = "mean") -> float:
    lines = []
    with open(filepath, "r") as f:
        for line in f:
            lines.append(json.loads(line))
    fieldvalues = []
    for line in lines:
        fieldvalues.append(line[field])
    
    if stattype == "mean":
        return np.nanmean(fieldvalues)
    elif stattype == "std":
        return np.nanstd(fieldvalues)
    else:
        raise ValueError(f"Unknown stattype: {stattype}.")
    

class StringIndexer:
    def __init__(self, strings):
        self._index = {s: i for i, s in enumerate(strings)}
        self._rindex = {i: s for i, s in enumerate(strings)}
    
    def get_index(self, s) -> int:
        return self._index[s]
    
    def get_string(self, i) -> str:
        return self._rindex[i]
    
    def get_strings(self) -> List[str]:
        return list(self._rindex[i] for i in range(len(self._rindex)))
    
    def __len__(self):
        return len(self._index)
    

class PairIndexer:
    def __init__(self, pairs: List[Tuple[str, str, float]]):
        self._index: Dict[Tuple[str, str], Tuple[int, float]] = {
            tuple(pair[:2]): tuple([i, pair[2]])
            for i, pair in enumerate(pairs)
        }
        self._rindex: Dict[int, Tuple[str, str, float]] = {
            i: pair for i, pair in enumerate(pairs)
        }
    
    def get_index(self, key: Tuple[str, str]) -> Tuple[int, float]:
        return self._index[key]
    
    def get_pair(self, i: int) -> Tuple[str, str, float]:
        return self._rindex[i]
    
    def __len__(self):
        return len(self._index)
    

def rand_indices(n: int, frac: float, seed: int = 42) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    k = int(n * frac)
    return torch.randperm(n, generator=g)[:k]