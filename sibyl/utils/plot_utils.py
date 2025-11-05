import ast
import re
import pathlib
import datetime
import json
from typing import List, Optional, Union, Tuple, Dict
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def run_pca(
    data: np.ndarray,
    /,
    n_components: int = 2,
    whiten: bool = False,
    random_state: Optional[int] = None,
    *,
    plot: bool = True,
    plot_dims: Tuple[int, int] = (1, 2),
    figsize: Tuple[int, int] = (6, 6),
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    save: bool = False,
    dotsize: int = 5,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    if labels is not None and colors is not None:
        assert len(labels) == len(colors), (
            "run_pca: length of labels and colors must match."
        )

    assert len(plot_dims) == 2 and plot_dims[0] != plot_dims[1], \
        "run_pca: plot_dims must be two distinct PCs (e.g., (1,4))."
    needed_components = max(n_components, plot_dims[0], plot_dims[1])

    _pca_full = PCA(
        whiten=whiten,
        random_state=random_state,
    )
    _pca_full.fit(data)
    evr_all = _pca_full.explained_variance_ratio_
    pca = PCA(
        n_components=needed_components,
        whiten=whiten,
        random_state=random_state,
    )
    components = pca.fit_transform(data)

    if plot:
        # convert to 0-index for numpy slices
        i = plot_dims[0] - 1
        j = plot_dims[1] - 1
        assert i < components.shape[1] and j < components.shape[1], \
            "run_pca: requested plot_dims exceed computed components."

        plt.figure(figsize=figsize)
        plt.scatter(
            components[:, i],
            components[:, j],
            alpha=1.0,
            c=colors if colors is not None else "blue",
            s=dotsize,
        )
        if labels is not None:
            for k, lab in enumerate(labels):
                plt.annotate(lab, (components[k, i], components[k, j]))

        var_ratio = pca.explained_variance_ratio_ * 100
        plt.title(
            f"PCA → PC{plot_dims[0]} ({var_ratio[i]:.1f}%)"
            f" vs PC{plot_dims[1]} ({var_ratio[j]:.1f}%)"
        )
        if x_range is not None:
            plt.xlim(x_range)
        if y_range is not None:
            plt.ylim(y_range)
        plt.xlabel(f"PC{plot_dims[0]}")
        plt.ylabel(f"PC{plot_dims[1]}")
        plt.tight_layout()

        if save:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"pca_plot_{ts}.png"
            plt.savefig(fname, bbox_inches="tight", dpi=1200)
            print(f"--> run_pca: PCA plot saved as {fname}.")
        plt.show()

    return components, evr_all


def _coerce_text(log_text: Union[str, pathlib.Path]) -> str:
    if isinstance(log_text, pathlib.Path) or \
        (isinstance(log_text, str) and log_text.endswith(".log")):
        return pathlib.Path(log_text).read_text()
    if not isinstance(log_text, str):
        raise ValueError("log_text must be a string or pathlib.Path")
    return log_text

def _compute_ema(xs: List[float], alpha: float) -> List[float]:
    if not xs:
        return []
    ema = [xs[0]]
    for x in xs[1:]:
        ema.append(alpha * ema[-1] + (1 - alpha) * x)
    return ema

def _parse_log_records(log_text: Union[str, pathlib.Path]):
    """
    Return list of records: dict(
        ts: datetime,
        epoch: int,
        step: Optional[int],
        metric: str,
        value: float
    )        
    Supports both step-level lines like:
        '... Epoch 000 | train_step17 loss 1.02'
    and epoch-level lines like:
        '... Epoch 012 | val loss 0.123'
    """
    text = _coerce_text(log_text)

    _FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    # Timestamp + Epoch + train_step + metric + value
    STEP_RE = re.compile(
        rf"^\s*(?P<ts>\d{{4}}-\d{{2}}-\d{{2}}\s+\d{{2}}:\d{{2}}:\d{{2}},\d{{3}}).*?"
        rf"Epoch\s+(?P<epoch>\d+)\s*\|\s*train_step(?P<step>\d+)\s+"
        rf"(?P<metric>.+?)\s+(?P<value>{_FLOAT})\s*$"
    )
    # Timestamp + Epoch + metric + value  (no step; e.g., 'val loss', 'train loss')
    EPOCH_RE = re.compile(
        rf"^\s*(?P<ts>\d{{4}}-\d{{2}}-\d{{2}}\s+\d{{2}}:\d{{2}}:\d{{2}},\d{{3}}).*?"
        rf"Epoch\s+(?P<epoch>\d+)\s*\|\s*(?P<metric>.+?)\s+(?P<value>{_FLOAT})\s*$"
    )

    records = []
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if m:
            ts = datetime.datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
            records.append({
                "ts": ts,
                "epoch": int(m.group("epoch")),
                "step": int(m.group("step")),
                "metric": m.group("metric").strip(),
                "value": float(m.group("value")),
            })
            continue
        m = EPOCH_RE.search(line)
        if m:
            # Guard: don't double-count train_step lines; STEP_RE already caught those
            if "train_step" in m.group("metric"):
                continue
            ts = datetime.datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
            records.append({
                "ts": ts,
                "epoch": int(m.group("epoch")),
                "step": None,
                "metric": m.group("metric").strip(),
                "value": float(m.group("value")),
            })
    return records

def _attach_step_indices(records):
    """
    Compute global_step and epoch_float for step-level records.
    """
    step_end_ts: Dict[Tuple[int,int], datetime.datetime] = {}
    for r in records:
        if r["step"] is not None:
            key = (r["epoch"], r["step"])
            ts = r["ts"]
            step_end_ts[key] = max(step_end_ts.get(key, ts), ts)

    ordered = sorted(step_end_ts.items(), key=lambda kv: (kv[1], kv[0][0], kv[0][1]))
    global_index = {key: i for i, (key, _) in enumerate(ordered)}

    per_epoch_sorted = defaultdict(list)
    for (ep, st), ts in step_end_ts.items():
        per_epoch_sorted[ep].append((ts, st))
    step_rank = {}
    steps_in_epoch = {}
    for ep, lst in per_epoch_sorted.items():
        lst.sort()
        step_rank[ep] = {st: i for i, (ts, st) in enumerate(lst)}
        steps_in_epoch[ep] = len(lst)

    for r in records:
        r["global_step"] = None
        r["epoch_float"] = None
        if r["step"] is not None:
            key = (r["epoch"], r["step"])
            r["global_step"] = global_index[key]
            n = steps_in_epoch[r["epoch"]]
            if n and n > 0:
                pos = step_rank[r["epoch"]][r["step"]] / float(n)
                r["epoch_float"] = r["epoch"] + pos
    return records

def _infer_steps_per_epoch(step_recs) -> Optional[int]:
    """
    Infer steps/epoch from observed step IDs.
    """
    if not step_recs:
        return None
    by_epoch = defaultdict(set)
    max_step_id = -1
    for r in step_recs:
        st = r["step"]
        by_epoch[r["epoch"]].add(st)
        if st > max_step_id:
            max_step_id = st
    spe_from_max = max_step_id + 1 if max_step_id >= 0 else None
    counts = [len(s) for s in by_epoch.values()]
    if counts:
        try:
            from statistics import mode
            spe_mode = mode(counts)
        except Exception:
            spe_mode = max(counts)
        return max(spe_from_max or 0, spe_mode)
    return spe_from_max

def _parse_and_pretty_print_config(filepath: str):
    """
    Reads the first line from a log file, extracts the config dict,
    and pretty-prints it in JSON format.
    """
    with open(filepath, "r") as f:
        first_line = f.readline().strip()

    # Find the substring after "config:"
    marker = "config:"
    if marker not in first_line:
        raise ValueError("No config found in first line")

    config_str = first_line.split(marker, 1)[1].strip()
    
    # Safely evaluate the Python dict string
    try:
        config_dict = ast.literal_eval(config_str)
    except Exception as e:
        raise ValueError(f"Failed to parse config: {e}")

    # Pretty-print JSON
    pretty_json = json.dumps(config_dict, indent=4)
    print(pretty_json)
    return config_dict


def plot_training_curve(
    log_text: Union[str, pathlib.Path],
    metrics: Optional[List[str]] = None,   # e.g., ["loss", "val loss"]
    *,
    x_axis: str = "global_step",           # keep "global_step" for step axis
    decay_rate: float = 0.9,
    plot_original: bool = True,
    raw_data_alpha: float = 0.12,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 6),
    combine_epoch_metrics_on_step: bool = False, # overlay epoch metrics on step axis
    epoch_anchor: str = "end",
    use_log_scale: bool = False,
    print_config: bool = False,
    # "start"-> n*spe, "mid"-> n*spe+spe/2, "end"->(n+1)*spe-1, "boundary"->(n+1)*spe
) -> None:
    """
    Plot step-level series on a step x-axis, and optionally overlay epoch-level metrics
    (e.g., validation) at multiples of steps_per_epoch.
    Example (train vs val loss on step axis):
        plot_training_curve(..., metrics=["loss", "val loss"],
                            x_axis="global_step",
                            combine_epoch_metrics_on_step=True,
                            steps_per_epoch=50,
                            epoch_anchor="start")
    """
    if print_config:
        _parse_and_pretty_print_config(log_text)
    records = _attach_step_indices(_parse_log_records(log_text))

    step_recs = [r for r in records if r["step"] is not None]
    epoch_recs = [r for r in records if r["step"] is None]

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    if metrics is None:
        uniq = sorted({norm(r["metric"]): r["metric"] for r in (step_recs or epoch_recs)}.values())
        metrics_to_plot = uniq
    else:
        metrics_to_plot = metrics

    use_ema = 0 < decay_rate < 1
    plt.figure(figsize=figsize)

    if step_recs:
        # --- Step-level series ---
        for label in metrics_to_plot:
            sel = [r for r in step_recs if norm(r["metric"]) == norm(label)]
            if not sel:
                continue
            if x_axis != "global_step":
                raise ValueError("For a pure step axis, call with x_axis='global_step'.")
            xs = [r["global_step"] for r in sel if r["global_step"] is not None]
            ys = [r["value"] for r in sel if r["global_step"] is not None]
            pairs = sorted(zip(xs, ys), key=lambda t: t[0])
            xs, ys = map(list, zip(*pairs)) if pairs else ([], [])
            if not xs:
                continue
            if plot_original:
                plt.plot(xs, ys, marker="o", markersize=3, alpha=raw_data_alpha, label=label)
            if use_ema:
                ema = _compute_ema(ys, decay_rate)
                plt.plot(xs, ema, linestyle="--", label=f"{label} EMA (α={decay_rate})")

        # --- Overlay epoch-level metrics at n * steps_per_epoch (or other anchor) ---
        if combine_epoch_metrics_on_step and epoch_recs:
            spe = _infer_steps_per_epoch(step_recs)
            if not spe or spe <= 0:
                raise ValueError("steps_per_epoch could not be inferred; please pass steps_per_epoch explicitly.")
            anchor = epoch_anchor.strip().lower()
            def anchor_pos(n: int) -> float:
                if anchor == "start":
                    return n * spe
                elif anchor == "mid":
                    return n * spe + spe / 2.0
                elif anchor == "end":
                    return (n + 1) * spe - 1
                elif anchor in {"boundary", "next"}:
                    return (n + 1) * spe
                else:
                    raise ValueError("epoch_anchor must be one of: start|mid|end|boundary")

            for label in metrics_to_plot:
                sel_e = [r for r in epoch_recs if norm(r["metric"]) == norm(label)]
                if not sel_e:
                    continue
                xs_e = [anchor_pos(r["epoch"]) for r in sel_e]
                ys_e = [r["value"] for r in sel_e]
                pairs = sorted(zip(xs_e, ys_e), key=lambda t: t[0])
                xs_e, ys_e = map(list, zip(*pairs))
                # Use distinct markers for epoch-level series
                plt.plot(xs_e, ys_e, linestyle="-", label=f"{label} (epoch)")

        plt.xlabel("Step")

    else:
        # No step-level records; fallback to epoch plotting (not your case, but keeps backward compatibility)
        by_metric: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for r in epoch_recs:
            by_metric[r["metric"]].append((r["epoch"], r["value"]))
        for label in metrics_to_plot:
            pairs = sorted(by_metric.get(label, []), key=lambda t: t[0])
            if not pairs:
                continue
            es, ys = map(list, zip(*pairs))
            if plot_original:
                plt.plot(es, ys, marker="o", markersize=3, alpha=raw_data_alpha, label=label)
            if use_ema:
                ema = _compute_ema(ys, decay_rate)
                plt.plot(es, ema, linestyle="--", label=f"{label} EMA (α={decay_rate})")
        plt.xlabel("Epoch")

    ylab = metrics_to_plot[0].title() if len(metrics_to_plot) == 1 else "Value"
    plt.ylabel(ylab)
    title = "Training Curves" + (f" (EMA α={decay_rate})" if use_ema else "")
    plt.title(title)
    if ymin is not None or ymax is not None:
        plt.ylim(ymin, ymax)
    if xmin is not None or xmax is not None:
        plt.xlim(xmin, xmax)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if use_log_scale:
        plt.xscale("log")
    plt.show()