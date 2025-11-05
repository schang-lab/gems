from __future__ import annotations

import pathlib
from typing import Dict, Optional, Tuple, List

import torch

def load_pickled_modules(ckpt_path: str, device: Optional[str] = None
) -> Tuple[Dict[str, torch.nn.Module], torch.nn.Module, Dict[str, torch.nn.Module]]:
    """
    Load encoders, gnn, decoders that were saved as full module objects.
    Returns (encoders_dict, gnn_module, decoders_dict).
    """
    dev = torch.device(
        device if device is not None 
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gnn = blob["gnn"].to(dev).eval()
    encoders = {}
    for k, m in blob["encoders"].items():
        encoders[k] = (m.to(dev).eval()) if m is not None else None
    decoders = {k: m.to(dev).eval() for k, m in blob["decoders"].items()}
    return encoders, gnn, decoders