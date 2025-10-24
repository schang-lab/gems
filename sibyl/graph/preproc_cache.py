from __future__ import annotations
import os
import json
import pathlib
import hashlib
import datetime
import inspect
import pickle
from typing import Any, Dict, Tuple

import torch
from torch_geometric.data import HeteroData as _PyG_HeteroData


CACHE_VERSION = 1  # bump when cache format changes


def _is_heterodata(obj: object) -> bool:
    # Avoid attribute access that triggers PyG's __getattr__/collect
    name = obj.__class__.__name__
    mod  = obj.__class__.__module__
    if name in {"CustomHeteroData", "HeteroData"}:
        return True
    if isinstance(obj, _PyG_HeteroData):
        return True
    # Fallback: many PyG objects live under torch_geometric.*
    return mod.startswith("torch_geometric")


def _file_sig(path: str | os.PathLike) -> Dict[str, Any]:
    p = pathlib.Path(path)
    try:
        s = p.stat()
        return {"path": str(p),
                "size": s.st_size,
                "mtime": int(s.st_mtime)}
    except FileNotFoundError:
        return {"path": str(p), "size": None, "mtime": None}


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compute_prep_cache_key(
        cfg_pynative: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Make a fingerprint that uniquely identifies the expensive 'prep' artifacts.
    We include dataset settings and split settings.
    """
    ds = cfg_pynative["dataset"]
    split_info = cfg_pynative["split_info"]

    qkeypath = ds.get("qkeypath", "")
    qstringpath = ds.get("qstringpath", "")
    frozen_embeddingpath = ds.get("frozen_embeddingpath", None)

    sig = {
        "cache_version": CACHE_VERSION,
        "dataset": {
            "input_wave_number": ds.get("input_wave_number", []),
            "qkeypath": _file_sig(qkeypath) if qkeypath else None,
            "qstringpath": _file_sig(qstringpath) if qstringpath else None,
            "frozen_embeddingpath": _file_sig(frozen_embeddingpath) if frozen_embeddingpath else None,
            "target_layer": ds.get("target_layer", None),
            "treat_as_different": ds.get("treat_as_different", True),
            "valid_traits": ds.get("valid_traits", []),
            "exit_undefined_traits": ds.get("exit_undefined_traits", True),
            "node_types": ds.get("node_types", []),
            "use_22_subgroups": ds.get("use_22_subgroups", False),
        },
        "split_info": split_info,
        "code_hint": "prep_cache_v1",
    }
    digest = hashlib.sha1(
        _stable_json(sig).encode("utf-8")
    ).hexdigest()[:16]
    return digest, sig


def _cpuify(obj):
    # Tensors → CPU
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    # Dict / list / tuple → recurse
    if isinstance(obj, dict):
        return {k: _cpuify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpuify(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpuify(v) for v in obj)
    # PyG hetero graphs → to("cpu") without peeking at x_dict
    if _is_heterodata(obj):
        try:
            return obj.to("cpu")
        except Exception:
            return obj  # if already CPU or to() not available
    # Everything else unchanged
    return obj


def _to_device(obj, device: torch.device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)
    if _is_heterodata(obj):
        try:
            return obj.to(device)
        except Exception:
            return obj
    return obj


def _ensure_dir(p: str | os.PathLike) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def cache_paths(repo_root: pathlib.Path, key: str) -> Tuple[pathlib.Path, pathlib.Path]:
    cache_dir = repo_root / "outputs" / "cache" / "graph_prep"
    _ensure_dir(cache_dir)
    return cache_dir / f"prep_{key}.pt", cache_dir / f"prep_{key}.meta.json"


def try_load_prep_cache(repo_root: pathlib.Path, key: str) -> Dict[str, Any] | None:
    path, meta = cache_paths(repo_root, key)
    if not path.exists():
        return None
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(bundle, dict):  # sanity
            return None
        if bundle.get("_cache_version") != CACHE_VERSION:
            return None
        return bundle
    except Exception:
        return None


def save_prep_cache(repo_root: pathlib.Path, key: str, signature: Dict[str, Any], bundle: Dict[str, Any]) -> None:
    path, meta = cache_paths(repo_root, key)
    bundle = {**bundle, "_cache_version": CACHE_VERSION, "_created_at": datetime.datetime.utcnow().isoformat() + "Z"}
    # always save CPU copy for portability
    cpu_bundle = _cpuify(bundle)
    torch.save(cpu_bundle, path)
    with open(meta, "w", encoding="utf-8") as f:
        json.dump({"key": key, "signature": signature, "created_at": bundle["_created_at"]}, f, indent=2)


def move_prep_to_device(bundle: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Return a shallow-copied bundle whose tensors/graphs are on the given device."""
    out = {}
    for k, v in bundle.items():
        if k.startswith("_cache_") or k in ("_cache_version", "_created_at"):
            continue
        out[k] = _to_device(v, device)
    return out


def show_per_instance_size(filepath: str | os.PathLike) -> None:
    data = torch.load(filepath,
                      map_location="cpu",
                      weights_only=False)
    sizes = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            sizes[k] = v.element_size() * v.numel()
        else:
            sizes[k] = len(pickle.dumps(v))

    for k, size in sizes.items():
        print(f"{k}: {size:,} bytes ({size/1024**2:.2f} MB)")
    print(f"Total size: {sum(sizes.values())/1024**2:.2f} MB")