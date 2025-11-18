#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_zucchet_style.py
---------------------
Produce two publication-ready figures from your saved results:
  • FigAB: Alignment vs depth (A) stacked with Loss vs depth (B).
  • FigEF: Alignment vs epoch (E) stacked with Loss vs epoch (F).

Inputs (per method, per configuration) are numpy files saved by your trainer:
  memory_{M}_hidden_{H}_layers_{L}_act_{ACT}/cosine_similarity/*.npy
  memory_{M}_hidden_{H}_layers_{L}_act_{ACT}/loss_trajectory/*.npy

Each cosine_similarity .npy holds a Python list of length = epochs,
where each item is a dict like:
  {'overall': float, 'layers': {'layer_0': {...}, 'layer_1': {...}, ...}}
We’ll consume the 'overall' series (default) OR a specific per-layer entry.

-----------
Examples
-----------

# 1) Panels A&B (ONLINE only; depths 1..4), Panels E&F (ONLINE,TBPTT,FBPTT at L=4)
python plot_zucchet_style.py \
  --root_ONLINE /path/to/results/toy_ONLINE \
  --root_TBPTT /path/to/results/toy_TBPTT \
  --root_FBPTT /path/to/results/toy_FBPTT \
  --memory 2 --hidden 64 --activation full_glu --mixing full \
  --ab_layers 1,2,3,4 \
  --ef_layers 4 \
  --methods ONLINE,TBPTT,FBPTT \
  --max_epochs 200 \
  --agg_epochs 5 \
  --cos_source overall \
  --outdir ./figs --fill

# 2) Same but use alignment from one specific layer (e.g., layer 1) instead of 'overall'
python plot_zucchet_style.py ... --cos_source layer:1

-----------
Notes
-----------
• For FBPTT: cosine alignment to itself is drawn as a flat 1.0 line.
• Error bars: mean ± s.e.m. across seeds.
• If seeds differ among methods, the script aligns on the intersection of seeds
  within each (method, depth) group automatically.
• If --agg_epochs > 1, we average non-overlapping windows of epochs both for
  alignment and loss to reduce noise and compress the x-axis.
"""

import argparse
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ------------------------- utils -------------------------

def _set_epoch_ticks(ax, xs, n=5):
    """Force exactly n integer ticks evenly spanning xs[0]..xs[-1]."""
    if xs is None or len(xs) == 0: 
        return
    lo, hi = int(0), int(xs[-1])
    ticks = np.linspace(lo, hi, num=n)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t)}" for t in ticks])

def _add_headroom(ax, frac=0.06, hard_top=None):
    """Expand y-limits upward by a fraction of the current data span."""
    y0, y1 = ax.get_ylim()
    span = max(1e-8, y1 - y0)
    top = y1 + frac * span
    if hard_top is not None:
        top = min(top, hard_top)
    ax.set_ylim(y0, top)

def _legend_one_row(ax, labels, handles, where="top"):
    """
    Place a single-row legend in the extra headroom at top (or bottom).
    - where='top' -> at y = 1.005 inside axes, spanning width
    """
    ncol = len(labels)
    ax.legend(
        handles, labels,
        frameon=False,
        loc="lower center" if where == "top" else "upper center",
        bbox_to_anchor=(0.5, 1.0 if where == "top" else -0.02),
        ncol=ncol, handlelength=2.0, columnspacing=1.0, borderpad=0.1, labelspacing=0.6,
    )

def _panel_letter(ax, letter: str, dx=0.0, dy=0.02, fontsize=None):
    # use current rcParams if fontsize not given
    if fontsize is None:
        base_size = plt.rcParams.get("axes.labelsize", 10)
        # Handle string values like 'medium', 'large', etc.
        if isinstance(base_size, str):
            fontsize = 12  # default fallback for string values
        else:
            fontsize = float(base_size) + 2
    ax.text(0.0 + dx, 1.0 + dy, letter, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=fontsize, fontweight="bold")


def _nice_axes(ax, x_int_ticks=True, ygrid=True):
    if ygrid:
        ax.grid(True, axis="y", alpha=0.25)
    if x_int_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _parse_seed_from_filename(fname: str) -> Optional[str]:
    # looks for "_seed_<SEED>.npy"
    m = re.search(r"_seed_([A-Za-z0-9\-]+)\.npy$", fname)
    return m.group(1) if m else None

def _window_average(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or y.size == 0:
        return y
    n = len(y)
    k = (n + w - 1) // w
    out = []
    for i in range(k):
        s, e = i * w, min((i + 1) * w, n)
        out.append(np.nanmean(y[s:e]))
    return np.array(out, dtype=float)

def _mean_sem(series_list: List[np.ndarray], T: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if not series_list:
        return np.array([]), np.array([])
    # trim/pad to common length T (or max length)
    if T is None:
        T = max(len(s) for s in series_list)
    M = []
    for s in series_list:
        if len(s) < T:
            pad = np.full(T, np.nan); pad[:len(s)] = s; M.append(pad)
        else:
            M.append(s[:T])
    M = np.vstack(M)
    mean = np.nanmean(M, axis=0)
    n_eff = np.sum(~np.isnan(M), axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        sem = np.nanstd(M, axis=0, ddof=1) / np.sqrt(np.maximum(n_eff, 1))
    sem[np.isnan(sem)] = 0.0
    return mean, sem

def _sig(memory: int, hidden: int, layers: int, activation: str) -> str:
    return f"memory_{memory}_hidden_{hidden}_layers_{layers}_act_{activation}"

def _glob_metric_files(root: Path, sig: str, subdir: str) -> Dict[str, Path]:
    """
    Returns {seed: path} for .npy under <root>/<sig>/<subdir>/
    where filename contains '_seed_<seed>.npy'.
    """
    d = root / sig / subdir
    out = {}
    if not d.exists():
        return out
    for p in d.glob("*.npy"):
        sd = _parse_seed_from_filename(p.name)
        if sd:
            out[sd] = p
    return out

def _load_cosine_series(path: Path, cos_source: str, max_epochs: int) -> np.ndarray:
    """
    cos_source:
      - "overall": uses item['overall']
      - "layer:k": uses item['layers'][f'layer_{k}']['all']
    """
    with open(path, "rb") as f:
        arr = np.load(f, allow_pickle=True)
    series = []
    if cos_source == "overall":
        for t, item in enumerate(arr):
            if t >= max_epochs: break
            series.append(float(item.get('overall', np.nan)))
    elif cos_source.startswith("layer:"):
        k = int(cos_source.split(":")[1])
        key = f"layer_{k}"
        for t, item in enumerate(arr):
            if t >= max_epochs: break
            lay = item.get('layers', {}).get(key, {})
            v = lay.get('all', np.nan)
            series.append(float(v) if np.isscalar(v) else np.nan)
    else:
        raise ValueError("--cos_source must be 'overall' or 'layer:<k>'")
    return np.array(series, dtype=float)

def _load_loss_series(path: Path, max_epochs: int) -> np.ndarray:
    with open(path, "rb") as f:
        arr = np.load(f, allow_pickle=True)
    # your trainer already saves per-epoch losses; just truncate
    return np.asarray(arr)[:max_epochs].astype(float)

def _align_epoch_axes(means: List[np.ndarray], sems: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Pad each mean/sem to the same length with NaNs; return also the epoch index (1..T)."""
    T = max((len(m) for m in means if len(m) > 0), default=0)
    if T == 0:
        return means, sems, np.array([])
    def pad(x):
        if len(x) == T: return x
        out = np.full(T, np.nan); out[:len(x)] = x; return out
    means = [pad(m) for m in means]
    sems  = [pad(s) for s in sems]
    epochs = np.arange(1, T + 1)
    return means, sems, epochs

# ------------------------- loading per (method, layers) -------------------------

def load_alignment_for_method_depth(
    root: Path, memory: int, hidden: int, layers: int, activation: str,
    cos_source: str, max_epochs: int
) -> Dict[str, np.ndarray]:
    """
    Returns per-seed time series of cosine for a single (method, depth).
    { seed: np.ndarray[T] }
    """
    sig = _sig(memory, hidden, layers, activation)
    cos_files = _glob_metric_files(root, sig, "cosine_similarity")
    out = {}
    for sd, p in cos_files.items():
        try:
            out[sd] = _load_cosine_series(p, cos_source, max_epochs)
        except Exception:
            pass
    return out

def load_loss_for_method_depth(
    root: Path, memory: int, hidden: int, layers: int, activation: str,
    max_epochs: int
) -> Dict[str, np.ndarray]:
    sig = _sig(memory, hidden, layers, activation)
    loss_files = _glob_metric_files(root, sig, "loss_trajectory")
    out = {}
    for sd, p in loss_files.items():
        try:
            out[sd] = _load_loss_series(p, max_epochs)
        except Exception:
            pass
    return out

# ------------------------- plotting -------------------------

def plot_fig_AB(
    outpath: Path,
    ab_method: str,
    ab_layers: List[int],
    roots: Dict[str, Path],
    memory: int, hidden: int, activation: str,
    mixing: str,
    max_epochs: int,
    cos_source: str,
    fill: bool,
    agg_epochs: int,
):
    """
    Panel A: alignment-vs-epoch, multiple lines (one per depth) for ab_method.
    Panel B: loss-vs-epoch, multiple lines (one per depth) for ab_method.
    """
    if ab_method not in roots:
        raise SystemExit(f"--ab_method '{ab_method}' missing. Provide --root_{ab_method}.")

    root = roots[ab_method]
    depths = []
    # per-depth series (mean & sem) for alignment and loss
    align_mu_list, align_se_list = [], []
    loss_mu_list,  loss_se_list  = [], []
    # collect lengths so we can align x-axes after windowing
    align_lengths, loss_lengths = [], []

    for L in ab_layers:
        # load per-seed time series
        cos_map  = load_alignment_for_method_depth(root, memory, hidden, L, activation, cos_source, max_epochs)
        loss_map = load_loss_for_method_depth(root,        memory, hidden, L, activation, max_epochs)

        # aggregate across seeds (epoch-wise)
        a_mean, a_sem = _mean_sem([v for v in cos_map.values()  if len(v) > 0])
        l_mean, l_sem = _mean_sem([v for v in loss_map.values() if len(v) > 0])

        if a_mean.size == 0 and l_mean.size == 0:
            # no data for this depth
            continue

        # windowing (non-overlapping)
        W = max(1, int(agg_epochs))
        a_mean_w = _window_average(a_mean, W)
        a_sem_w  = _window_average(a_sem,  W)  # approx OK for viz
        l_mean_w = _window_average(l_mean, W)
        l_sem_w  = _window_average(l_sem,  W)

        align_mu_list.append(a_mean_w); align_se_list.append(a_sem_w)
        loss_mu_list.append(l_mean_w);  loss_se_list.append(l_sem_w)
        align_lengths.append(len(a_mean_w))
        loss_lengths.append(len(l_mean_w))
        depths.append(L)

    if not depths:
        raise SystemExit("No data for FigAB. Check roots/config or ab_layers.")

    # align x across depths
    T_align = max(align_lengths) if align_lengths else 0
    T_loss  = max(loss_lengths)  if loss_lengths  else 0

    def _pad_list(arrs, T):
        out = []
        for a in arrs:
            if len(a) < T:
                pad = np.full(T, np.nan); pad[:len(a)] = a; out.append(pad)
            else:
                out.append(a[:T])
        return out

    align_mu_list = _pad_list(align_mu_list, T_align)
    align_se_list = _pad_list(align_se_list, T_align)
    loss_mu_list  = _pad_list(loss_mu_list,  T_loss)
    loss_se_list  = _pad_list(loss_se_list,  T_loss)

    epochs_A = np.arange(1, T_align + 1) * max(1, int(agg_epochs))
    epochs_B = np.arange(1, T_loss  + 1) * max(1, int(agg_epochs))

    # ---- Plot
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(8.8, 6.6), sharex=False, gridspec_kw={"height_ratios":[3,2]})
    # color cycle per depth
    cmap = plt.cm.get_cmap('tab10', max(1, len(depths)))

    # Panel A: alignment vs epoch (one line per depth)
    for i, (L, mu, se) in enumerate(zip(depths, align_mu_list, align_se_list)):
        c = cmap(i)
        axA.plot(epochs_A, mu, lw=2.2, color=c, label=f"L={L}")
        if fill:
            axA.fill_between(epochs_A, mu - se, mu + se, color=c, alpha=0.18)
    axA.set_ylim(0.94, 1.001)
    axA.set_ylabel("Cosine similarity")
    axA.set_title(f"(A) Alignment vs epoch  |  method={ab_method}  |  M={memory}, H={hidden}, act={activation}, mix={mixing}, src={cos_source}")
    axA.grid(True, axis="y", alpha=0.25)
    axA.legend(frameon=False, loc="lower right")
    axA.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Panel B: loss vs epoch (one line per depth)
    for i, (L, mu, se) in enumerate(zip(depths, loss_mu_list, loss_se_list)):
        c = cmap(i)
        axB.plot(epochs_B, mu, lw=2.0, color=c, label=f"L={L} loss")
        if fill:
            axB.fill_between(epochs_B, mu - se, mu + se, color=c, alpha=0.18)
    axB.set_yscale("log")
    axB.set_xlabel("Epoch")
    axB.set_ylabel("Training loss")
    axB.set_title("(B) Performance vs epoch")
    axB.grid(True, axis="y", alpha=0.25)
    axB.legend(frameon=False, loc="upper right")
    axB.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"[FigAB] Saved: {outpath}")

def plot_fig_EF(
    outpath: Path,
    methods: List[str],
    roots: Dict[str, Path],
    memory: int, hidden: int, layers_ef: int, activation: str,
    mixing: str,
    max_epochs: int,
    agg_epochs: int,
    cos_source: str,
    fill: bool
):
    """
    Panel E: alignment vs epoch (mean±sem), methods overlayed.
    Panel F: loss vs epoch (mean±sem), methods overlayed.
    """
    colors = {
        "ONLINE": "black",
        "TBPTT": "tab:orange",
        "FBPTT": "tab:green"
    }

    # collect series per method
    align_means, align_sems, names_E, epochs_E = [], [], [], None
    loss_means,  loss_sems,  names_F, epochs_F = [], [], [], None

    for m in methods:
        if m not in roots:
            print(f"[FigEF] Warning: method {m} has no root; skipping.")
            continue

        cos_map  = load_alignment_for_method_depth(roots[m], memory, hidden, layers_ef, activation, cos_source, max_epochs)
        loss_map = load_loss_for_method_depth(roots[m], memory, hidden, layers_ef, activation, max_epochs)

        # For FBPTT, alignment-to-self is 1.0 (use length = min length across seeds or fallback to loss length)
        if m == "FBPTT":
            lengths = []
            if loss_map:
                lengths += [len(v) for v in loss_map.values()]
            if cos_map:
                lengths += [len(v) for v in cos_map.values()]
            T = max(lengths) if lengths else max_epochs
            cos_series_list = [np.ones(min(T, max_epochs), dtype=float)]
        else:
            # intersect seeds with cosine data (cos similarity requires paired FBPTT during training generation)
            cos_series_list = [cos_map[s] for s in sorted(cos_map.keys()) if len(cos_map[s]) > 0]

        loss_series_list = [loss_map[s] for s in sorted(loss_map.keys()) if len(loss_map[s]) > 0]

        # aggregate across seeds (epoch-wise)
        a_mean, a_sem = _mean_sem(cos_series_list)
        l_mean, l_sem = _mean_sem(loss_series_list)

        # epoch windowing
        W = max(1, agg_epochs)
        a_mean_w = _window_average(a_mean, W)
        a_sem_w  = _window_average(a_sem,  W)  # approximate; good enough for viz
        l_mean_w = _window_average(l_mean, W)
        l_sem_w  = _window_average(l_sem,  W)

        # align epoch axes later across methods
        align_means.append(a_mean_w); align_sems.append(a_sem_w); names_E.append(m)
        loss_means.append(l_mean_w);  loss_sems.append(l_sem_w);  names_F.append(m)

    # align x for panel E
    align_means, align_sems, xs_E = _align_epoch_axes(align_means, align_sems)
    # align x for panel F
    loss_means,  loss_sems,  xs_F = _align_epoch_axes(loss_means,  loss_sems)

    if xs_E.size == 0 and xs_F.size == 0:
        raise SystemExit("No data for FigEF. Check roots/config.")

    # convert xs to window-end epochs
    W = max(1, agg_epochs)
    if xs_E.size > 0: xs_E = xs_E * W
    if xs_F.size > 0: xs_F = xs_F * W

    # ---- Plot
    fig, (axE, axF) = plt.subplots(2, 1, figsize=(8.8, 6.6), sharex=False, gridspec_kw={"height_ratios":[3,2]})

    # Panel E: alignment vs epoch
    for m, mu, se in zip(names_E, align_means, align_sems):
        c = colors.get(m, None)
        axE.plot(xs_E, mu, lw=2.0, label=m, color=c)
        if fill:
            axE.fill_between(xs_E, mu - se, mu + se, alpha=0.18, color=c)
    axE.set_ylim(0.0, 1.01)
    axE.set_ylabel("Cosine similarity")
    axE.set_title(f"(E) Alignment vs epoch  |  L={layers_ef}  |  M={memory}, H={hidden}, act={activation}, mix={mixing}, src={cos_source}")
    axE.grid(True, axis="y", alpha=0.25)
    axE.legend(frameon=False, loc="lower right")
    axE.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Panel F: loss vs epoch
    for m, mu, se in zip(names_F, loss_means, loss_sems):
        c = colors.get(m, None)
        axF.plot(xs_F, mu, lw=2.2, label=f"{m} loss", color=c)
        if fill:
            axF.fill_between(xs_F, mu - se, mu + se, alpha=0.18, color=c)
    axF.set_yscale("log")
    axF.set_xlabel("Epoch")
    axF.set_ylabel("Training loss")
    axF.set_title("(F) Performance vs epoch")
    axF.grid(True, axis="y", alpha=0.25)
    axF.legend(frameon=False, loc="upper right")
    axF.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"[FigEF] Saved: {outpath}")

def plot_fig_ADEF_composite(
    outpath: Path,
    ab_method: str,
    ab_layers: List[int],
    methods: List[str],
    ef_layers: int,
    roots: Dict[str, Path],
    memory: int, hidden: int, activation: str, mixing: str,
    max_epochs: int, agg_epochs: int, cos_source: str,
    fill: bool,
    fig2_fontsize: float = 10.0,   # <— new
):
    """
    2x2 composite:
      A: alignment vs epoch, curves by depth (method=ab_method)
      B: loss vs epoch, curves by depth (method=ab_method)
      C: alignment vs epoch, curves by method (depth=ef_layers)
      D: loss vs epoch, curves by method (depth=ef_layers)
    Now with: 5 epoch ticks, single-row legends in headroom, and per-figure font control.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # collect A/B (per-depth)
    W = max(1, int(agg_epochs))
    cmap_depths = plt.cm.get_cmap('viridis', max(1, len(ab_layers)))
    depth_labels = [f"L={L}" for L in ab_layers]

    AB_align, AB_align_sem, AB_loss, AB_loss_sem = [], [], [], []

    if ab_method not in roots:
        raise SystemExit(f"[Fig A/B] missing root for method={ab_method}")

    for L in ab_layers:
        cos_map  = load_alignment_for_method_depth(roots[ab_method], memory, hidden, L, activation, cos_source, max_epochs)
        loss_map = load_loss_for_method_depth(roots[ab_method],        memory, hidden, L, activation, max_epochs)
        a_mean, a_sem = _mean_sem([v for v in cos_map.values()  if len(v) > 0])
        l_mean, l_sem = _mean_sem([v for v in loss_map.values() if len(v) > 0])
        AB_align.append(_window_average(a_mean, W)); AB_align_sem.append(_window_average(a_sem, W))
        AB_loss.append(_window_average(l_mean, W));  AB_loss_sem.append(_window_average(l_sem, W))

    AB_align, AB_align_sem, xs_A = _align_epoch_axes(AB_align, AB_align_sem)
    AB_loss,  AB_loss_sem,  xs_B = _align_epoch_axes(AB_loss,  AB_loss_sem)
    if xs_A.size > 0: xs_A = xs_A * W
    if xs_B.size > 0: xs_B = xs_B * W

    # collect C/D (per-method at fixed depth)
    method_colors = {"ONLINE":"black", "TBPTT":"tab:orange", "FBPTT":"tab:green"}
    CD_align, CD_align_sem, names_E = [], [], []
    CD_loss,  CD_loss_sem,  names_F = [], [], []

    for m in methods:
        if m not in roots: 
            continue
        cos_map  = load_alignment_for_method_depth(roots[m], memory, hidden, ef_layers, activation, cos_source, max_epochs)
        loss_map = load_loss_for_method_depth(roots[m],        memory, hidden, ef_layers, activation, max_epochs)
        if m == "FBPTT":
            lengths = [len(v) for v in list(cos_map.values()) + list(loss_map.values())]
            T = max(lengths) if lengths else max_epochs
            cos_list = [np.ones(min(T, max_epochs), dtype=float)]
        else:
            cos_list = [v for v in cos_map.values() if len(v) > 0]
        loss_list = [v for v in loss_map.values() if len(v) > 0]

        a_mean, a_sem = _mean_sem(cos_list); l_mean, l_sem = _mean_sem(loss_list)
        CD_align.append(_window_average(a_mean, W)); CD_align_sem.append(_window_average(a_sem, W)); names_E.append(m)
        CD_loss.append(_window_average(l_mean, W));  CD_loss_sem.append(_window_average(l_sem, W));  names_F.append(m)

    CD_align, CD_align_sem, xs_C = _align_epoch_axes(CD_align, CD_align_sem)
    CD_loss,  CD_loss_sem,  xs_D = _align_epoch_axes(CD_loss,  CD_loss_sem)
    if xs_C.size > 0: xs_C = xs_C * W
    if xs_D.size > 0: xs_D = xs_D * W

    # ---- FIGURE (font control is local to this fig only)
    with plt.rc_context({
        "font.size":        fig2_fontsize,
        "axes.titlesize":   fig2_fontsize + 1,
        "axes.labelsize":   fig2_fontsize,
        "legend.fontsize":  fig2_fontsize,
        "xtick.labelsize":  fig2_fontsize,
        "ytick.labelsize":  fig2_fontsize,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.3), sharex=False, sharey=False)
        axA, axC = axes[0,0], axes[0,1]
        axB, axD = axes[1,0], axes[1,1]

        # A — alignment by depth
        depth_handles = []
        for i, (mu, se) in enumerate(zip(AB_align, AB_align_sem)):
            c = cmap_depths(i)
            h, = axA.plot(xs_A, mu, lw=2.0, color=c, label=depth_labels[i])
            depth_handles.append(h)
            if fill: axA.fill_between(xs_A, mu - se, mu + se, color=c, alpha=0.18)
        axA.set_ylabel("Cos similarity")
        _nice_axes(axA, x_int_ticks=False)        # <— don’t override our ticks
        _set_epoch_ticks(axA, xs_A, n=5)          # <— apply ticks LAST
        axA.tick_params(axis="x", labelbottom=True)
        _panel_letter(axA, "A")
        _legend_one_row(axA, depth_labels, depth_handles, where="top")

        # B — loss by depth
        depth_handles_B = []
        for i, (mu, se) in enumerate(zip(AB_loss, AB_loss_sem)):
            c = cmap_depths(i)
            h, = axB.plot(xs_B, mu, lw=2.0, color=c, label=depth_labels[i])
            depth_handles_B.append(h)
            if fill: axB.fill_between(xs_B, mu - se, mu + se, color=c, alpha=0.18)
        axB.set_yscale("log"); axB.set_xlabel("Epoch"); axB.set_ylabel("Training loss")
        _nice_axes(axB, x_int_ticks=False)
        _set_epoch_ticks(axB, xs_B, n=5)
        axB.tick_params(axis="x", labelbottom=True)
        _panel_letter(axB, "B")
        _legend_one_row(axB, depth_labels, depth_handles_B, where="top")

        # C — alignment by method
        method_handles = []
        for m, mu, se in zip(names_E, CD_align, CD_align_sem):
            c = method_colors.get(m, None)
            h, = axC.plot(xs_C, mu, lw=2.0, color=c, label=m if m != "ONLINE" else "Ours")
            method_handles.append(h)
            if fill: axC.fill_between(xs_C, mu - se, mu + se, alpha=0.18, color=c)
        axC.set_ylabel("Cos similarity")
        _nice_axes(axC, x_int_ticks=False)
        _set_epoch_ticks(axC, xs_C, n=5)
        axC.tick_params(axis="x", labelbottom=True)
        _panel_letter(axC, "C")
        _legend_one_row(axC, [h.get_label() for h in method_handles], method_handles, where="top")

        # D — loss by method
        method_handles_D = []
        for m, mu, se in zip(names_F, CD_loss, CD_loss_sem):
            c = method_colors.get(m, None)
            h, = axD.plot(xs_D, mu, lw=2.0, color=c, label=m if m != "ONLINE" else "Ours")
            method_handles_D.append(h)
            if fill: axD.fill_between(xs_D, mu - se, mu + se, alpha=0.18, color=c)
        axD.set_yscale("log"); axD.set_xlabel("Epoch"); axD.set_ylabel("Training loss")
        _nice_axes(axD, x_int_ticks=False)
        _set_epoch_ticks(axD, xs_D, n=5)
        axD.tick_params(axis="x", labelbottom=True)
        _panel_letter(axD, "D")
        _legend_one_row(axD, [h.get_label() for h in method_handles_D], method_handles_D, where="top")

        plt.tight_layout(pad=0.8, w_pad=0.9, h_pad=0.9)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"[Composite 2x2] Saved: {outpath}")

def plot_fig_ADEF_tall4(
    outpath: Path,
    ab_method: str, ab_layers: List[int],
    methods: List[str], ef_layers: int,
    roots: Dict[str, Path],
    memory: int, hidden: int, activation: str, mixing: str,
    max_epochs: int, agg_epochs: int, cos_source: str, fill: bool,
):
    """
    Create a tall 4x1 figure:
      A: alignment vs epoch, one curve per depth (method=ab_method)
      B: loss vs epoch, one curve per depth (method=ab_method)
      C: alignment vs epoch, one curve per method (depth=ef_layers)
      D: loss vs epoch, one curve per method (depth=ef_layers)
    """
    import matplotlib.pyplot as plt

    # ---------- collect A/B (per-depth, one method) ----------
    W = max(1, int(agg_epochs))
    cmap_depths = plt.cm.get_cmap('viridis', max(1, len(ab_layers)))
    depth_labels = [f"L={L}" for L in ab_layers]

    AB_align, AB_align_sem = [], []
    AB_loss,  AB_loss_sem = [], []

    if ab_method not in roots:
        raise SystemExit(f"[Fig A/B] missing root for method={ab_method}")

    for i, L in enumerate(ab_layers):
        cos_map  = load_alignment_for_method_depth(roots[ab_method], memory, hidden, L, activation, cos_source, max_epochs)
        loss_map = load_loss_for_method_depth(roots[ab_method], memory, hidden, L, activation, max_epochs)
        a_mean, a_sem = _mean_sem([v for v in cos_map.values()  if len(v) > 0])
        l_mean, l_sem = _mean_sem([v for v in loss_map.values() if len(v) > 0])

        a_mean_w = _window_average(a_mean, W); a_sem_w = _window_average(a_sem, W)
        l_mean_w = _window_average(l_mean, W); l_sem_w = _window_average(l_sem, W)

        AB_align.append(a_mean_w); AB_align_sem.append(a_sem_w)
        AB_loss.append(l_mean_w);  AB_loss_sem.append(l_sem_w)

    AB_align, AB_align_sem, xs_A = _align_epoch_axes(AB_align, AB_align_sem)
    AB_loss,  AB_loss_sem,  xs_B = _align_epoch_axes(AB_loss,  AB_loss_sem)
    if xs_A.size > 0: xs_A = xs_A * W
    if xs_B.size > 0: xs_B = xs_B * W

    # ---------- collect C/D (per-method, one depth) ----------
    method_colors = {"ONLINE":"black", "TBPTT":"tab:orange", "FBPTT":"tab:green"}
    CD_align, CD_align_sem, names_E = [], [], []
    CD_loss,  CD_loss_sem,  names_F = [], [], []

    for m in methods:
        if m not in roots:
            print(f"[Fig C/D] Warning: method {m} missing root; skip.")
            continue
        cos_map  = load_alignment_for_method_depth(roots[m], memory, hidden, ef_layers, activation, cos_source, max_epochs)
        loss_map = load_loss_for_method_depth(roots[m], memory, hidden, ef_layers, activation, max_epochs)

        if m == "FBPTT":
            lengths = []
            if loss_map: lengths += [len(v) for v in loss_map.values()]
            if cos_map:  lengths += [len(v) for v in cos_map.values()]
            T = max(lengths) if lengths else max_epochs
            cos_series_list = [np.ones(min(T, max_epochs), dtype=float)]
        else:
            cos_series_list = [v for v in cos_map.values() if len(v) > 0]

        loss_series_list = [v for v in loss_map.values() if len(v) > 0]

        a_mean, a_sem = _mean_sem(cos_series_list)
        l_mean, l_sem = _mean_sem(loss_series_list)

        a_mean_w = _window_average(a_mean, W); a_sem_w = _window_average(a_sem, W)
        l_mean_w = _window_average(l_mean, W); l_sem_w = _window_average(l_sem, W)

        CD_align.append(a_mean_w); CD_align_sem.append(a_sem_w); names_E.append(m)
        CD_loss.append(l_mean_w);  CD_loss_sem.append(l_sem_w);  names_F.append(m)

    CD_align, CD_align_sem, xs_E = _align_epoch_axes(CD_align, CD_align_sem)
    CD_loss,  CD_loss_sem,  xs_F = _align_epoch_axes(CD_loss,  CD_loss_sem)
    if xs_E.size > 0: xs_E = xs_E * W
    if xs_F.size > 0: xs_F = xs_F * W

    # ---------- figure layout: 4x1 tall ----------
    fig, axes = plt.subplots(4, 1, figsize=(7.4, 9.5), sharex=False)
    axA, axB, axC, axD = axes[0], axes[1], axes[2], axes[3]

    # --- Panel A (alignment vs epoch by depth)
    for i, (mu, se) in enumerate(zip(AB_align, AB_align_sem)):
        c = cmap_depths(i)
        axA.plot(xs_A, mu, lw=2.0, color=c, label=depth_labels[i])
        if fill: axA.fill_between(xs_A, mu - se, mu + se, color=c, alpha=0.18)
    axA.set_ylim(0.94, 1.001)
    axA.set_ylabel("Cos similarity")
    _nice_axes(axA)
    _panel_letter(axA, "A")
    axA.legend(frameon=False, loc="lower right")

    # --- Panel B (loss vs epoch by depth)
    for i, (mu, se) in enumerate(zip(AB_loss, AB_loss_sem)):
        c = cmap_depths(i)
        axB.plot(xs_B, mu, lw=2.0, color=c, label=depth_labels[i])
        if fill: axB.fill_between(xs_B, mu - se, mu + se, color=c, alpha=0.18)
    axB.set_yscale("log")
    axB.set_ylabel("Training loss")
    _nice_axes(axB)
    _panel_letter(axB, "B")
    axB.legend(frameon=False, loc="upper right")

    # --- Panel C (alignment vs epoch by method)
    for m, mu, se in zip(names_E, CD_align, CD_align_sem):
        c = method_colors.get(m, None)
        axC.plot(xs_E, mu, lw=2.0, color=c, label=m)
        if fill: axC.fill_between(xs_E, mu - se, mu + se, alpha=0.18, color=c)
    axC.set_ylim(0.0, 1.0)
    axC.set_ylabel("Cos similarity")
    _nice_axes(axC)
    _panel_letter(axC, "C")
    axC.legend(frameon=False, loc="lower right")

    # --- Panel D (loss vs epoch by method)
    for m, mu, se in zip(names_F, CD_loss, CD_loss_sem):
        c = method_colors.get(m, None)
        axD.plot(xs_F, mu, lw=2.0, color=c, label=f"{m}")
        if fill: axD.fill_between(xs_F, mu - se, mu + se, alpha=0.18, color=c)
    axD.set_yscale("log")
    axD.set_xlabel("Epoch")
    axD.set_ylabel("Training loss")
    _nice_axes(axD)
    _panel_letter(axD, "D")
    axD.legend(frameon=False, loc="upper right")

    # tighten & save
    plt.tight_layout(pad=1.0)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"[Tall 4x1] Saved: {outpath}")


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    # roots by method (provide only the ones you have)
    ap.add_argument("--root_ONLINE", type=Path)
    ap.add_argument("--root_TBPTT",  type=Path)
    ap.add_argument("--root_FBPTT",  type=Path)

    # config shared
    ap.add_argument("--memory", type=int, required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--activation", type=str, default="full_glu")
    ap.add_argument("--mixing", type=str, default="full")

    # AB panels
    ap.add_argument("--ab_layers", type=str, required=True,
                    help="Comma-separated list of depths for A/B (e.g., '1,2,3,4').")
    ap.add_argument("--ab_method", type=str, default="ONLINE",
                    help="Method for A/B panels. Default ONLINE. (Usually ONLINE in the paper.)")

    # EF panels
    ap.add_argument("--ef_layers", type=int, required=True,
                    help="Depth to use for E/F panels (e.g., 4).")
    ap.add_argument("--methods", type=str, default="ONLINE,TBPTT,FBPTT",
                    help="Comma-separated methods to plot in E/F (subset of ONLINE,TBPTT,FBPTT).")

    # plotting / processing
    ap.add_argument("--max_epochs", type=int, default=1000)
    ap.add_argument("--agg_epochs", type=int, default=1,
                    help="Window size for epoch aggregation (>=1).")
    ap.add_argument("--cos_source", type=str, default="overall",
                    help="'overall' (default) or 'layer:<k>' to use per-layer 'all' cosine.")
    ap.add_argument("--outdir", type=Path, default=Path("./figs"))
    ap.add_argument("--prefix", type=str, default="")
    ap.add_argument("--fill", action="store_true", default=False)
    ap.add_argument("--composite_prefix", type=str, default="Fig2_like",
                    help="Filename prefix for the composite figure.")
    ap.add_argument("--fig2_fontsize", type=float, default=10.0,
                help="Base font size for the 2x2 Fig2-like composite only.")

    args = ap.parse_args()

    # roots dict
    roots = {}
    if args.root_ONLINE: roots["ONLINE"] = args.root_ONLINE
    if args.root_TBPTT:  roots["TBPTT"]  = args.root_TBPTT
    if args.root_FBPTT:  roots["FBPTT"]  = args.root_FBPTT

    # parse layers list for A/B
    try:
        ab_layers = [int(x) for x in args.ab_layers.split(",") if x.strip() != ""]
        ab_layers = sorted(set(ab_layers))
    except Exception:
        raise SystemExit("--ab_layers must be a comma-separated list of integers, e.g., '1,2,3,4'.")

    # outdir
    args.outdir.mkdir(parents=True, exist_ok=True)

    # file names
    base = args.prefix or f"M{args.memory}_H{args.hidden}_act{args.activation}_mix{args.mixing}_T{args.max_epochs}_W{args.agg_epochs}_src{args.cos_source.replace(':','-')}"

    # --- FigAB ---
    fig_ab_path = args.outdir / f"{base}_FigAB_method-{args.ab_method}_depths-{'-'.join(map(str,ab_layers))}.png"
    plot_fig_AB(
        outpath=fig_ab_path,
        ab_method=args.ab_method,
        ab_layers=ab_layers,
        roots=roots,
        memory=args.memory, hidden=args.hidden, activation=args.activation,
        mixing=args.mixing,
        max_epochs=args.max_epochs,
        agg_epochs=args.agg_epochs,
        cos_source=args.cos_source,
        fill=args.fill
    )

    # --- FigEF ---
    fig_ef_path = args.outdir / f"{base}_FigEF_L{args.ef_layers}_methods-{'-'.join(args.methods.split(','))}.png"
    methods = [m.strip() for m in args.methods.split(",") if m.strip() != ""]
    plot_fig_EF(
        outpath=fig_ef_path,
        methods=methods,
        roots=roots,
        memory=args.memory, hidden=args.hidden, layers_ef=args.ef_layers, activation=args.activation,
        mixing=args.mixing,
        max_epochs=args.max_epochs,
        agg_epochs=args.agg_epochs,
        cos_source=args.cos_source,
        fill=args.fill
    )

    # --- Composite 2x2 ---
    comp_name = f"{args.composite_prefix}_M{args.memory}_H{args.hidden}_Lef{args.ef_layers}_act{args.activation}_mix{args.mixing}_T{args.max_epochs}_W{args.agg_epochs}_src{args.cos_source.replace(':','-')}.png"
    comp_path = args.outdir / comp_name
    plot_fig_ADEF_composite(
        outpath=comp_path,
        ab_method=args.ab_method,
        ab_layers=ab_layers,
        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
        ef_layers=args.ef_layers,
        roots=roots,
        memory=args.memory, hidden=args.hidden, activation=args.activation, mixing=args.mixing,
        max_epochs=args.max_epochs, agg_epochs=args.agg_epochs, cos_source=args.cos_source,
        fill=args.fill,
        fig2_fontsize=args.fig2_fontsize,
    )
    # --- Tall 4x1  ---
    tall_name = f"{args.composite_prefix}_TALL4_M{args.memory}_H{args.hidden}_Lef{args.ef_layers}_act{args.activation}_mix{args.mixing}_T{args.max_epochs}_W{args.agg_epochs}_src{args.cos_source.replace(':','-')}.png"
    tall_path = args.outdir / tall_name
    plot_fig_ADEF_tall4(
        outpath=tall_path,
        ab_method=args.ab_method, ab_layers=ab_layers,
        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
        ef_layers=args.ef_layers,
        roots=roots,
        memory=args.memory, hidden=args.hidden, activation=args.activation, mixing=args.mixing,
        max_epochs=args.max_epochs, agg_epochs=args.agg_epochs, cos_source=args.cos_source,
        fill=args.fill,
    )


if __name__ == "__main__":
    main()
