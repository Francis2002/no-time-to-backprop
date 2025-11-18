#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_tgap_and_grad_plots.py

Builds:
  1) Truncation-gap curve: MSE vs Memory for FBPTT / ONLINE / TBPTT
  2) Gradient cosine-similarity plots:
       - per layer
       - per parameter group (lambda, gamma, B)
       - forward-mode only (lambda+gamma+B)
       - all params (if available)

Assumptions about your on-disk layout (robust to small variations):
  results_root/
    toy_FBPTT/ ... or ... /toy/...
    toy_TBPTT/
    toy_ONLINE/
    ...
  and inside each configuration:
    .../loss_trajectory/<flag>.npy
    .../gamma_evolution/<flag>.npy         [optional]
    .../lambda_evolution/<flag>.npy        [optional]
    .../all_grads/<flag>.npy               [optional]
  plus one or more global_results.csv files.

Usage example:
  python make_tgap_and_grad_plots.py \
      --results-root results_from_gpu \
      --outdir figs \
      --dataset toy \
      --architecture ZUC \
      --activation full_glu \
      --mixing full \
      --layers 1 2 4 \
      --hidden 32 64 128 \
      --memories 1 2 3 4 5
"""

import argparse
import json
import math
import os
os.environ["MPLBACKEND"] = "Agg"  # must be set before importing pyplot
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import csv

# -------------------------
# Utilities
# -------------------------

def find_files(root: Path, name: str) -> List[Path]:
    """Recursively find all files named `name` under `root`."""
    return [p for p in root.rglob(name) if p.is_file()]

def sem(x: np.ndarray, axis=0):
    n = np.sum(~np.isnan(x), axis=axis)
    return np.nanstd(x, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))

def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def str_to_bool(value):
    """Convert string representations to boolean values."""
    if isinstance(value, str):
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            return False
        else:
            try:
                return bool(int(value))
            except ValueError:
                return False
    return bool(value)

def canonical_key(row: Dict[str, Any]) -> Tuple:
    """Configuration identity ignoring method/seed (must match across methods)."""
    return (
        row.get("dataset", "toy"),
        row["architecture"],
        row["activation"],
        row["mixing"],
        str_to_bool(row["prenorm"]),
        str_to_bool(row["postnorm"]),
        str_to_bool(row["encoder"]),
        str_to_bool(row["layer_output"]),
        str_to_bool(row["extra_skip"]),
        row["decoder"],
        str_to_bool(row["non_linearity_in_recurrence"]),
        int(row["num_layers"]),
        int(row["state_size"]) if "state_size" in row else int(row["num_hidden"]),
        int(row["memory"]),
        int(row["batch_size"]),
    )

def row_to_flag(row: Dict[str, Any]) -> str:
    """Rebuild the same flag string you used for saving trajectories."""
    return (
        f"prenorm_{row['prenorm']}_postnorm_{row['postnorm']}"
        f"_encoder_{row['encoder']}_layerout_{row['layer_output']}"
        f"_extraskip_{row['extra_skip']}_decoder_{row['decoder']}"
        f"_mixing_{row['mixing']}_nonlinrec_{row['non_linearity_in_recurrence']}"
        f"_batch_size_{row['batch_size']}_seed_{row['seed']}"
    )

def general_subfolder(row: Dict[str, Any]) -> str:
    """Rebuild the config subfolder name you used."""
    # state_size was your 'hidden' (d_hidden)
    state_size = row.get("state_size", row.get("num_hidden"))
    return f"memory_{row['memory']}_hidden_{state_size}_layers_{row['num_layers']}_act_{row['activation']}"

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def pick_metric(row: Dict[str, Any]) -> float:
    """Prefer average of final 100 steps (stable); fallback to final."""
    v = row.get("average_final_100_loss", None)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        v = row.get("final_loss", None)
    return safe_float(v)

def cosine_similarity(a: np.ndarray, b: np.ndarray, eps=1e-12) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))

def flatten_list(xs):
    return [y for x in xs for y in x]

# -------------------------
# Data ingestion
# -------------------------

def load_all_results(results_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Path]]:
    """
    Load all rows from every 'global_results.csv' under results_root.
    Also build a map from (method, row) -> run root path to fetch trajectories and grads.
    """
    csv_files = find_files(results_root, "global_results.csv")
    rows = []
    row_path_index = {}  # key: (method, seed, canonical_key) -> base path
    for csv_path in csv_files:
        base_dir = csv_path.parent
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Fill dataset if missing (toy by default)
                row.setdefault("dataset", "toy")
                # Normalize some fields to strings
                for k in ["prenorm","postnorm","encoder","layer_output","extra_skip","non_linearity_in_recurrence"]:
                    if k in row and isinstance(row[k], bool):
                        row[k] = "1" if row[k] else "0"
                rows.append(row)

                key = canonical_key(row)
                idx = (row["method"], int(row.get("seed", -1)), key)
                row_path_index[idx] = base_dir
    return rows, row_path_index

def find_blob_paths(base_dir: Path, row: Dict[str, Any]) -> Dict[str, Path]:
    """
    From a base_dir that contained the CSV, reconstruct paths to this run's blobs.
    """
    sub = general_subfolder(row)
    flag = row_to_flag(row)

    # We don't know if the method is encoded in the parent name or not;
    # search relatively in case the structure differs.
    # Build candidates and use the first that exists.
    rel_candidates = [
        base_dir / sub,                                      # same folder
        base_dir / ".." / sub,                               # one up
        base_dir.parent / sub,                               # same parent
    ]
    chosen_root = None
    for c in rel_candidates:
        if c.exists():
            chosen_root = c
            break
    if chosen_root is None:
        # Fall back to direct search of the loss_trajectory file
        # to recover the true parent; otherwise return missing paths.
        return {
            "loss": None,
            "gamma": None,
            "lambda": None,
            "grads": None,
        }

    return {
        "loss":   chosen_root / "loss_trajectory" / f"{flag}.npy",
        "gamma":  chosen_root / "gamma_evolution" / f"{flag}.npy",
        "lambda": chosen_root / "lambda_evolution" / f"{flag}.npy",
        "grads":  chosen_root / "all_grads" / f"{flag}.npy",
    }

# -------------------------
# Aggregation / selection
# -------------------------

def filter_rows(rows: List[Dict[str, Any]], dataset, architecture, activation, mixing,
                layers, hiddens, memories) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("dataset","toy") != dataset:
            continue
        if r.get("architecture") != architecture:
            continue
        if r.get("activation") != activation:
            continue
        if r.get("mixing") != mixing:
            continue
        if int(r["num_layers"]) not in layers:
            continue
        h = int(r.get("state_size", r.get("num_hidden")))
        if h not in hiddens:
            continue
        if int(r["memory"]) not in memories:
            continue
        out.append(r)
    return out

def group_by_key(rows: List[Dict[str, Any]]) -> Dict[Tuple, Dict[str, List[Dict[str, Any]]]]:
    """
    { canonical_key -> { method -> [rows (seeds)] } }
    """
    g: Dict[Tuple, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g[canonical_key(r)][r["method"]].append(r)
    return g

def intersect_seedsets(rows_a: List[Dict], rows_b: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Keep only rows whose seeds appear in both lists (matched comparison)."""
    sa = {int(r["seed"]) for r in rows_a if "seed" in r}
    sb = {int(r["seed"]) for r in rows_b if "seed" in r}
    common = sa & sb
    if not common:
        return [], []
    a = [r for r in rows_a if int(r["seed"]) in common]
    b = [r for r in rows_b if int(r["seed"]) in common]
    return a, b

# -------------------------
# Plot: Truncation gap
# -------------------------

def plot_tgap(groups, methods_order, outpath: Path, ylog=True):
    """
    Build MSE vs Memory plot.
    groups: dict keyed by (layers, hidden, memory) or similar—here we simply loop rows.
    """
    # Collect per memory losses per method
    # Each "group" is a canonical config (dataset, arch, act, mixing, flags...) including layers/hidden/memory.
    # We aggregate over all (layers, hidden) provided by the user—feel free to lock them before calling.
    per_method = defaultdict(lambda: defaultdict(list))  # method -> memory -> [losses across (seeds x capacities)]
    for key, bucket in groups.items():
        # key unpack for readability
        *_, num_layers, hidden, memory = key
        for method, rows in bucket.items():
            for r in rows:
                m = int(r["memory"])
                per_method[method][m].append(pick_metric(r))

    # Sort memories
    memories = sorted(set(flatten_list([list(d.keys()) for d in per_method.values()])))
    plt.figure(figsize=(6,4.2))
    for method in methods_order:
        if method not in per_method:
            continue
        ys = []
        es = []
        for m in memories:
            vals = np.array(per_method[method].get(m, []), dtype=float)
            ys.append(np.nanmean(vals) if vals.size else np.nan)
            es.append(sem(vals) if vals.size else np.nan)
        ys = np.array(ys, dtype=float)
        es = np.array(es, dtype=float)

        # plot mean with errorbars; keep style simple (no forced colors)
        plt.errorbar(memories, ys, yerr=es, marker="", linewidth=2, capsize=3, label=method)

    if ylog:
        plt.yscale("log")
    plt.xlabel("Memory")
    plt.ylabel("MSE")
    # Extract configuration info for title
    if groups:
        # Get a sample key to extract layers and hidden size
        sample_key = next(iter(groups.keys()))
        *_, num_layers, hidden, memory = sample_key
        plt.title(f"Truncation gap - {num_layers} layers, {hidden} hidden")
    else:
        plt.title("Truncation gap")
    plt.grid(True, which="both", axis="y", alpha=0.25)
    plt.xticks(memories)  # Force x-axis to show only integer memory values
    plt.legend(title="Method")
    ensure_outdir(outpath.parent)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# -------------------------
# Gradient extraction helpers
# -------------------------

def load_grad_blob(path: Path):
    """Load a saved grads .npy (list[dict[layer_i]{...}]) produced by your runner."""
    if (path is None) or (not path.exists()):
        return None
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None

def extract_param_arrays(grad_layer_dict: Dict[str, Any], group: str) -> List[np.ndarray]:
    """
    grad_layer_dict is something like:
      {'seq': {'nu': [...], 'theta': [...], 'gamma_log': [...], 'B_re': [...], 'B_im': [...]}}
      or exactly the structure you constructed under grads_for_debug.

    group ∈ {'lambda', 'gamma', 'B', 'forward', 'all'}:
      - 'lambda' → {nu, theta} backpropagated to diag_lambda via VJP (already done in your code when updating)
                   Here we cosine on *incoming grads* per param tensor (nu, theta).
      - 'gamma' → gamma_log
      - 'B'     → B_re, B_im
      - 'forward' → lambda + gamma + B
      - 'all'   → everything we can find under 'seq' (+ optionally encoder/decoder if present)
    """
    seq = grad_layer_dict.get("seq", grad_layer_dict)  # sometimes already inside
    arrays = []
    if group in ("lambda", "forward", "all"):
        if "nu" in seq:      arrays.append(np.asarray(seq["nu"]))
        if "theta" in seq:   arrays.append(np.asarray(seq["theta"]))
    if group in ("gamma", "forward", "all"):
        if "gamma_log" in seq: arrays.append(np.asarray(seq["gamma_log"]))
    if group in ("B", "forward", "all"):
        if "B_re" in seq:    arrays.append(np.asarray(seq["B_re"]))
        if "B_im" in seq:    arrays.append(np.asarray(seq["B_im"]))
    # 'all' could include more (e.g., encoder), but typically your dump is seq-only.
    return arrays

def concat_flat(arrays: List[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate([a.reshape(-1) for a in arrays]).astype(np.float64)

def grads_at_index(grad_blob: np.ndarray, idx: int) -> Dict[int, Dict[str, Any]]:
    """
    grad_blob is a list-like over steps; each element is:
      { 'layer_0': {...}, 'layer_1': {...}, ... }
    Return that structure at `idx` as a simpler {layer_idx -> dict}.
    """
    if grad_blob is None or len(grad_blob) == 0:
        return {}
    i = idx if idx >= 0 else (len(grad_blob) + idx)
    i = max(0, min(i, len(grad_blob) - 1))
    entry = grad_blob[i].item() if isinstance(grad_blob[i], np.ndarray) else grad_blob[i]
    # normalize keys
    out = {}
    for k, v in entry.items():
        if k.startswith("layer_"):
            out[int(k.split("_")[1])] = v
        else:
            out[k] = v
    return out

# -------------------------
# Plot: Gradient cos-sim
# -------------------------

def plot_grad_cossim_per_layer(cossim_table: Dict[str, np.ndarray], outpath: Path, title: str):
    """
    cossim_table: method -> array[L] of mean±sem per layer packed as (means, sems)
                  stored as dict with keys method+"_mean", method+"_sem"
    """
    plt.figure(figsize=(6.2, 4.0))
    layers = np.arange(cossim_table["L"])
    for method in cossim_table["methods"]:
        mu = cossim_table[method+"_mean"]
        se = cossim_table[method+"_sem"]
        plt.errorbar(layers, mu, yerr=se, marker="o", linewidth=2, capsize=3, label=method)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Layer index")
    plt.ylabel("Cosine similarity to FBPTT")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(title="Method")
    ensure_outdir(outpath.parent)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_single_bar(means: Dict[str, float], sems: Dict[str, float], outpath: Path, title: str):
    methods = list(means.keys())
    vals = [means[m] for m in methods]
    errs = [sems[m] for m in methods]
    x = np.arange(len(methods))
    plt.figure(figsize=(4.8, 4.2))
    plt.bar(x, vals, yerr=errs, capsize=4)
    plt.xticks(x, methods)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Cosine similarity to FBPTT")
    plt.title(title)
    ensure_outdir(outpath.parent)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def build_grad_cossim(
    groups,
    row_path_index,
    grad_index=-1,
    param_group="forward",   # 'lambda' | 'gamma' | 'B' | 'forward' | 'all'
    restrict_layers=None     # e.g., [0,1,2,3] or None for all
):
    """
    Compute cosine similarity to FBPTT per layer and overall for ONLINE and TBPTT.
    Returns:
      per_layer: dict suitable for plot_grad_cossim_per_layer
      overall_forward: (means, sems) per method
    """
    methods_ref = "FBPTT"
    methods_cmp = ["ONLINE", "TBPTT"]

    # Accumulators:
    per_layer_scores = {m: [] for m in methods_cmp}  # list over seeds/configs -> np.array[num_layers]
    overall_scores = {m: [] for m in methods_cmp}    # list over seeds/configs -> scalar

    any_layers = None

    for key, bucket in groups.items():
        if methods_ref not in bucket:
            continue
        # We need matching seeds for each comparator
        for m in methods_cmp:
            if m not in bucket:
                continue
            rows_ref, rows_cmp = intersect_seedsets(bucket[methods_ref], bucket[m])
            if not rows_ref:
                continue

            # iterate common seeds
            for r_ref in rows_ref:
                s = int(r_ref["seed"])
                # find the matching comparator row (same seed)
                rc = [rr for rr in rows_cmp if int(rr["seed"]) == s]
                if not rc:
                    continue
                r_cmp = rc[0]

                # locate grad blobs
                ref_paths = find_blob_paths(row_path_index[(methods_ref, s, canonical_key(r_ref))], r_ref)
                cmp_paths = find_blob_paths(row_path_index[(m, s, canonical_key(r_cmp))], r_cmp)
                g_ref = load_grad_blob(ref_paths["grads"])
                g_cmp = load_grad_blob(cmp_paths["grads"])
                if g_ref is None or g_cmp is None:
                    # skip if missing
                    continue

                # extract per-step dicts
                step_ref = grads_at_index(g_ref, grad_index)
                step_cmp = grads_at_index(g_cmp, grad_index)

                # common layer indices
                layer_ids = sorted([k for k in step_ref.keys() if isinstance(k, int) and k in step_cmp])
                if restrict_layers is not None:
                    layer_ids = [l for l in layer_ids if l in restrict_layers]
                if not layer_ids:
                    continue

                # per-layer cos-sim
                per_layer = []
                for li in layer_ids:
                    arr_ref = concat_flat(extract_param_arrays(step_ref[li], param_group))
                    arr_cmp = concat_flat(extract_param_arrays(step_cmp[li], param_group))
                    if arr_ref.size == 0 or arr_cmp.size == 0:
                        per_layer.append(np.nan)
                    else:
                        per_layer.append(cosine_similarity(arr_cmp, arr_ref))
                per_layer = np.array(per_layer, dtype=float)
                per_layer_scores[m].append(per_layer)

                # overall
                all_ref = concat_flat([concat_flat(extract_param_arrays(step_ref[li], param_group)) for li in layer_ids])
                all_cmp = concat_flat([concat_flat(extract_param_arrays(step_cmp[li], param_group)) for li in layer_ids])
                overall_scores[m].append(cosine_similarity(all_cmp, all_ref))

                any_layers = layer_ids

    # Aggregate
    out_per_layer = {"methods": methods_cmp, "L": (len(any_layers) if any_layers else 0)}
    for m in methods_cmp:
        if len(per_layer_scores[m]) == 0:
            out_per_layer[m+"_mean"] = np.array([])
            out_per_layer[m+"_sem"]  = np.array([])
        else:
            M = np.stack(per_layer_scores[m], axis=0)  # [runs, L]
            out_per_layer[m+"_mean"] = np.nanmean(M, axis=0)
            out_per_layer[m+"_sem"]  = sem(M, axis=0)

    out_overall_mean = {m: (float(np.nanmean(overall_scores[m])) if overall_scores[m] else np.nan) for m in methods_cmp}
    out_overall_sem  = {m: (float(sem(np.array(overall_scores[m])) if overall_scores[m] else np.nan)) for m in methods_cmp}

    return out_per_layer, out_overall_mean, out_overall_sem

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="figs")
    ap.add_argument("--dataset", type=str, default="toy")
    ap.add_argument("--architecture", type=str, default="ZUC")
    ap.add_argument("--activation", type=str, default="full_glu")
    ap.add_argument("--mixing", type=str, default="full")
    ap.add_argument("--layers", type=int, nargs="+", default=[1,2,4])
    ap.add_argument("--hidden", type=int, nargs="+", default=[32,64,128])
    ap.add_argument("--memories", type=int, nargs="+", default=[1,2,3,4,5])
    ap.add_argument("--grad-index", type=int, default=-1, help="Which saved grad snapshot to use (-1 = last)")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    # 1) Load everything
    rows, row_path_index = load_all_results(results_root)

    # 2) Filter to your desired slice (dataset/arch/activation/mixing/layers/hiddens/M)
    filt = filter_rows(
        rows,
        dataset=args.dataset,
        architecture=args.architecture,
        activation=args.activation,
        mixing=args.mixing,
        layers=set(args.layers),
        hiddens=set(args.hidden),
        memories=set(args.memories),
    )

    if not filt:
        print("[!] No rows matched your filters. Check --results-root and filters.")
        return

    # 3) Organize by canonical configuration
    grouped = group_by_key(filt)

    # 4) Truncation gap (MSE vs Memory) with mean±sem across seeds & capacities
    # Extract configuration info for filename
    layers_str = "_".join(map(str, sorted(args.layers)))
    hidden_str = "_".join(map(str, sorted(args.hidden)))
    tgap_path = outdir / f"{args.dataset}_tgap_MSE_vs_memory_layers_{layers_str}_hidden_{hidden_str}.png"
    plot_tgap(grouped, methods_order=["FBPTT", "ONLINE", "TBPTT"], outpath=tgap_path, ylog=True)
    print(f"[+] Saved truncation-gap plot → {tgap_path}")

    # 5) Gradient cos-sim panels (if grads exist)
    #    a) Per layer (forward-mode params)
    per_layer, ov_mu, ov_se = build_grad_cossim(
        grouped, row_path_index, grad_index=args.grad_index, param_group="forward", restrict_layers=None
    )
    if per_layer["L"] == 0:
        print("[i] No gradient blobs were found; skipping cosine-similarity plots.")
        return

    plot_grad_cossim_per_layer(
        per_layer,
        outpath=outdir / "grad_cossim_per_layer.png",
        title="Gradient alignment per layer (forward-mode params)",
    )
    print(f"[+] Saved per-layer cos-sim → {outdir/'grad_cossim_per_layer.png'}")

    #    b) Forward-mode overall (bar)
    plot_single_bar(
        ov_mu, ov_se, outdir / "grad_cossim_forward_mode.png",
        title="Overall gradient alignment (λ, γ, B)"
    )
    print(f"[+] Saved forward-mode overall cos-sim → {outdir/'grad_cossim_forward_mode.png'}")

    #    c) Per param group
    for group_name, fname, title in [
        ("lambda", "grad_cossim_lambda.png", "Gradient alignment (λ)"),
        ("gamma",  "grad_cossim_gamma.png",  "Gradient alignment (γ)"),
        ("B",      "grad_cossim_B.png",      "Gradient alignment (B)"),
    ]:
        per_layer_g, ov_mu_g, ov_se_g = build_grad_cossim(
            grouped, row_path_index, grad_index=args.grad_index, param_group=group_name
        )
        if per_layer_g["L"] > 0:
            plot_grad_cossim_per_layer(per_layer_g, outpath=outdir / fname, title=title)
            print(f"[+] Saved {group_name} per-layer cos-sim → {outdir/fname}")
            # also an overall bar for each group
            plot_single_bar(ov_mu_g, ov_se_g, outdir / f"overall_{fname}", title=title + " (overall)")
            print(f"[+] Saved overall {group_name} cos-sim → {outdir/('overall_'+fname)}")

    #    d) All params (if present)
    per_layer_all, ov_mu_all, ov_se_all = build_grad_cossim(
        grouped, row_path_index, grad_index=args.grad_index, param_group="all"
    )
    if per_layer_all["L"] > 0:
        plot_grad_cossim_per_layer(
            per_layer_all, outpath=outdir / "grad_cossim_all_params.png",
            title="Gradient alignment per layer (all params)"
        )
        plot_single_bar(
            ov_mu_all, ov_se_all, outdir / "grad_cossim_all_params_overall.png",
            title="Overall gradient alignment (all params)"
        )
        print(f"[+] Saved all-params cos-sim plots in {outdir}")

if __name__ == "__main__":
    main()
