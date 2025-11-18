#!/usr/bin/env python3
"""
plot_cross_coupling.py
----------------------
Generate "truncation-gap-like" plots to showcase the importance of cross-coupling
when results for mixing=full and mixing=none live in **separate CSV files**.

Creates 6 individual plots:
  - FBPTT @ L in {1,2,4}  (two lines: mixing=full vs mixing=none)
  - Ours(ONLINE) @ L in {1,2,4} (two lines: mixing=full vs mixing=none)
Plus composite figures:
  - 2x3 grid (rows=methods FBPTT/Ours, cols=layers 1,2,4)
  - 3x2 grid (rows=layers 1,2,4, cols=methods FBPTT/Ours)

Columns expected in each CSV (same schema as your truncation-gap files):
  method, architecture, memory, state_size, num_layers, activation, seed,
  and one of {average_final_100_loss, final_loss, best_loss}.
We will inject a 'mixing' column ourselves as {'full','none'} based on which CSV it came from.

Usage example
=============
python plot_cross_coupling.py \
  --fbptt_full_csv   /path/to/toy_FBPTT/global_results_mixing_full.csv \
  --fbptt_none_csv   /path/to/toy_FBPTT/global_results_mixing_none.csv \
  --online_full_csv  /path/to/toy_ONLINE/global_results_mixing_full.csv \
  --online_none_csv  /path/to/toy_ONLINE/global_results_mixing_none.csv \
  --hidden 64 \
  --layers 1 2 4 \
  --memories 1 2 3 4 5 \
  --architecture ZUC \
  --activation full_glu \
  --outdir ./figs_cross \
  --prefix crossmix

Notes
=====
- Y axis defaults to log10 scale (MSE). Disable with --no-ylog10.
- "ONLINE" will be labeled "Ours" in titles/legends.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _pick_metric(df: pd.DataFrame, override: str) -> str:
    if override != "auto":
        return override
    if "average_final_100_loss" in df.columns:
        return "average_final_100_loss"
    if "final_loss" in df.columns:
        return "final_loss"
    return "best_loss"


def _load_concat_add_mixing(full_csv: Path, none_csv: Path) -> pd.DataFrame:
    dff = pd.read_csv(full_csv)
    dfn = pd.read_csv(none_csv)
    dff = dff.copy(); dff["mixing"] = "full"
    dfn = dfn.copy(); dfn["mixing"] = "none"
    # Normalize a few columns
    for df in (dff, dfn):
        if "method" in df.columns:
            df["method"] = df["method"].astype(str).str.upper()
        if "activation" in df.columns:
            df["activation"] = df["activation"].astype(str)
        if "architecture" in df.columns:
            df["architecture"] = df["architecture"].astype(str)
    return pd.concat([dff, dfn], ignore_index=True)


def _filter(df: pd.DataFrame,
            method_name: str,
            architecture: str,
            activation: str,
            hidden: int,
            layers: int,
            memories: List[int]) -> pd.DataFrame:
    mask = (
        (df["method"] == method_name) &
        (df["architecture"] == architecture) &
        (df["activation"] == activation) &
        (df["state_size"] == hidden) &
        (df["num_layers"] == layers) &
        (df["memory"].isin(memories)) &
        (df["mixing"].isin(["full", "none"]))
    )
    return df.loc[mask].copy()


def _agg_seed_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty:
        return df
    g = (
        df.groupby(["mixing", "memory"])[metric]
          .agg(mean="mean",
               sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0,
               n="count")
          .reset_index()
    )
    return g.sort_values(["mixing", "memory"])


def _plot_single_panel(ax: plt.Axes,
                       agg: pd.DataFrame,
                       memories: List[int],
                       title: str,
                       ylog10: bool,
                       metric_label: str):
    styles = {
        "full": {"label": "mixing = full", "color": None, "marker": "o"},
        "none": {"label": "mixing = none", "color": None, "marker": "s"},
    }

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
    styles["full"]["color"] = color_cycle[0]
    styles["none"]["color"] = color_cycle[1]

    for mixing_key in ["full", "none"]:
        sub = agg[agg["mixing"] == mixing_key]
        if sub.empty:
            continue
        x = sub["memory"].to_numpy()
        y = sub["mean"].to_numpy()
        e = sub["sem"].to_numpy()
        ax.errorbar(
            x, y, yerr=e, capsize=3, linewidth=2, marker=styles[mixing_key]["marker"],
            color=styles[mixing_key]["color"], label=styles[mixing_key]["label"], alpha=0.95
        )

    ax.set_xticks(memories)
    ax.set_xlabel("Memory")
    ax.set_ylabel(metric_label)
    if ylog10:
        ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", linestyle="-", alpha=0.25)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=True, fontsize=9, loc="best")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbptt_full_csv", type=Path, required=True)
    ap.add_argument("--fbptt_none_csv", type=Path, required=True)
    ap.add_argument("--online_full_csv", type=Path, required=True)
    ap.add_argument("--online_none_csv", type=Path, required=True)

    ap.add_argument("--architecture", type=str, default="ZUC")
    ap.add_argument("--activation", type=str, default="full_glu")
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--memories", type=int, nargs="+", default=[1, 2, 3, 4, 5])

    ap.add_argument("--metric", type=str, default="auto",
                    help="One of {auto, average_final_100_loss, final_loss, best_loss}.")
    ap.add_argument("--no-ylog10", action="store_true")
    ap.add_argument("--outdir", type=Path, default=Path("./figs_cross"))
    ap.add_argument("--prefix", type=str, default="crossmix")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load and stitch per-method data with mixing labels
    df_fbptt = _load_concat_add_mixing(args.fbptt_full_csv, args.fbptt_none_csv)
    df_online = _load_concat_add_mixing(args.online_full_csv, args.online_none_csv)

    # Pick metric
    metric = _pick_metric(pd.concat([df_fbptt, df_online], ignore_index=True), args.metric)
    metric_label = {"average_final_100_loss": "MSE (avg last 100)",
                    "final_loss": "MSE (final)",
                    "best_loss": "MSE (best)"}[metric]
    ylog10 = not args.no_ylog10

    # Build each of the 6 panels and save individually
    panels: Dict[Tuple[str, int], Path] = {}
    for method_name, label, df_all in [
        ("FBPTT", "FBPTT", df_fbptt),
        ("ONLINE", "Ours", df_online)
    ]:
        for L in args.layers:
            df = _filter(df_all, method_name, args.architecture,
                         args.activation, args.hidden, L, args.memories)
            if df.empty:
                raise SystemExit(f"No rows after filtering for {method_name} @ L={L}. "
                                 f"Check CSV paths and filters.")
            agg = _agg_seed_stats(df, metric)

            fig, ax = plt.subplots(figsize=(4.0, 3.2))
            title = f"{label} — Layers={L}, Hidden={args.hidden}"
            _plot_single_panel(ax, agg, args.memories, title, ylog10, metric_label)

            outname = f"{args.prefix}_{method_name.lower()}_L{L}_H{args.hidden}.png"
            outpath = args.outdir / outname
            fig.tight_layout()
            fig.savefig(outpath, dpi=300, bbox_inches="tight")
            plt.close(fig)

            panels[(method_name, L)] = outpath
            print(f"[Saved] {outpath}")

    # Composite: 2x3 (rows=methods FBPTT/Ours, cols=layers sorted)
    sorted_layers = sorted(args.layers)
    fig23, axes23 = plt.subplots(2, 3, figsize=(12, 6.5), sharex=False, sharey=False)
    for i, (method_name, label, df_all) in enumerate([("FBPTT", "FBPTT", df_fbptt),
                                                      ("ONLINE", "Ours", df_online)]):
        for j, L in enumerate(sorted_layers):
            ax = axes23[i, j]
            df = _filter(df_all, method_name, args.architecture,
                         args.activation, args.hidden, L, args.memories)
            agg = _agg_seed_stats(df, metric)
            title = f"{label} — L={L}"
            _plot_single_panel(ax, agg, args.memories, title, ylog10, metric_label)
    fig23.tight_layout()
    grid23 = args.outdir / f"{args.prefix}_grid_2x3.png"
    fig23.savefig(grid23, dpi=300, bbox_inches="tight")
    plt.close(fig23)
    print(f"[Saved] {grid23}")

    # Composite: 3x2 (rows=layers, cols=methods)
    fig32, axes32 = plt.subplots(3, 2, figsize=(10, 10.5), sharex=False, sharey=False)
    for i, L in enumerate(sorted_layers):
        for j, (method_name, label, df_all) in enumerate([("FBPTT", "FBPTT", df_fbptt),
                                                          ("ONLINE", "Ours", df_online)]):
            ax = axes32[i, j]
            df = _filter(df_all, method_name, args.architecture,
                         args.activation, args.hidden, L, args.memories)
            agg = _agg_seed_stats(df, metric)
            title = f"{label} — L={L}"
            _plot_single_panel(ax, agg, args.memories, title, ylog10, metric_label)
    fig32.tight_layout()
    grid32 = args.outdir / f"{args.prefix}_grid_3x2.png"
    fig32.savefig(grid32, dpi=300, bbox_inches="tight")
    plt.close(fig32)
    print(f"[Saved] {grid32}")


if __name__ == "__main__":
    main()
