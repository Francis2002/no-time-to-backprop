#!/usr/bin/env python3
"""
plot_truncation_gap.py
----------------------
Builds a "truncation-gap" curve figure from your three CSVs (FBPTT, TBPTT, ONLINE)
produced by run_toy_task.py.

What it does
============
- Loads per-run rows from each CSV.
- Filters to architecture==ZUC (your ZucchetCell) and dataset==toy (implicit in your runs).
- Lets you choose EITHER a set of hidden sizes (state_size) OR a set of number of layers,
  and plots 2 color groups (FBPTT/ONLINE) with shade variations for each chosen value.
- X-axis = memory horizon (integer ticks); Y-axis = MSE (log scale by default).
- For each memory value, the figure shows mean ± s.e.m. across seeds.
- The output filename auto-encodes whether you varied hidden or layers and the chosen values.

Usage
=====
Example: vary hidden sizes (fix layers):
    python plot_truncation_gap.py \
      --fbptt_csv /.../results_from_gpu/toy_FBPTT/global_results.csv \
      --online_csv /.../results_from_gpu/toy_ONLINE/global_results.csv \
      --vary hidden --hidden 32 64 128 --layers 1 \
      --memories 1 2 3 4 5 --mixing full --activation full_glu \
      --outdir ./figs

Example: vary number of layers (fix hidden size):
    python plot_truncation_gap.py \
      --fbptt_csv ... --online_csv ... \
      --vary layers --layers 1 2 4 --hidden 64 \
      --memories 1 2 3 4 5 --mixing full --activation full_glu \
      --outdir ./figs

Notes
=====
- The script expects columns at least: method, architecture, memory, state_size, num_layers,
  activation, mixing, seed, and one of {average_final_100_loss, final_loss, best_loss}.
- Memory ticks are integers only.
"""

import argparse
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def _pick_metric(df: pd.DataFrame) -> str:
    if "average_final_100_loss" in df.columns:
        return "average_final_100_loss"
    if "final_loss" in df.columns:
        return "final_loss"
    return "best_loss"


def _filter_df(df: pd.DataFrame,
               architecture: str,
               mixings: List[str],
               activation: str,
               layers: List[int],
               hidden: int,
               memories: List[int]) -> pd.DataFrame:
    mask = (
        (df["architecture"] == architecture) &
        (df["mixing"].isin(mixings)) &
        (df["activation"] == activation) &
        (df["memory"].isin(memories)) &
        (df["num_layers"].isin(layers)) &
        (df["state_size"] == hidden)
    )
    return df.loc[mask].copy()


def _agg_over_seeds(df: pd.DataFrame, metric: str, group_keys: list) -> pd.DataFrame:
    # mean ± s.e.m. (over seeds)
    return (
        df.groupby(group_keys)[metric]
          .agg(mean="mean",
               sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0,
               n="count")
          .reset_index()
    )


def _shades(cmap_name: str, n: int):
    cmap = get_cmap(cmap_name)
    if n == 1:
        return [cmap(0.7)]
    xs = np.linspace(0.35, 0.85, n)
    return [cmap(x) for x in xs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fbptt_csv", type=Path, required=True)
    p.add_argument("--online_csv", type=Path, required=True)

    p.add_argument("--architecture", type=str, default="ZUC")
    p.add_argument("--activation", type=str, default="full_glu")

    # Fixed hidden size as requested
    p.add_argument("--hidden", type=int, default=32,
                   help="Fixed hidden/state size. Default: 32")
    # Multiple now explicitly refers to num_layers values
    p.add_argument("--layers", type=int, nargs="+", required=True,
                   help="List of num_layers values to create one subplot per value.")

    p.add_argument("--memories", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--ylog10", action="store_true", default=True)
    p.add_argument("--title", type=str, default="Truncation Gap: rotational vs none")
    p.add_argument("--outdir", type=Path, default=Path("./figs"))
    p.add_argument("--outfile", type=str, default="")
    p.add_argument("--metric", type=str, default="auto",
                   help="One of {auto, average_final_100_loss, final_loss, best_loss}.")

    args = p.parse_args()
    

    args.outdir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for path, method in [
        (args.fbptt_csv, "FBPTT"),
        (args.online_csv, "ONLINE"),
    ]:
        df = pd.read_csv(path)
        df["method"] = df["method"].astype(str).str.upper()
        df = df[df["method"] == method]
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    metric = _pick_metric(df_all) if args.metric == "auto" else args.metric

    # We need rows for both mixings: rotational and none
    df_all = _filter_df(
        df_all,
        architecture=args.architecture,
        mixings=["rotational", "none"],
        activation=args.activation,
        layers=args.layers,
        hidden=args.hidden,
        memories=args.memories,
    )
    if df_all.empty:
        raise SystemExit("No rows after filtering. Check CSV paths and filters.")

    # Aggregate over seeds. Group by method, memory, mixing, and num_layers
    agg = _agg_over_seeds(df_all, metric, ["method", "memory", "mixing", "num_layers"])

    # Build K×1 grid, each row fixes num_layers (the 'multiple')
    layers_sorted = sorted(args.layers)
    K = len(layers_sorted)
    fig, axes = plt.subplots(K, 1, figsize=(9.0, 4.0*K), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    # Define colors/styles for combinations of (method, mixing)
    combo_styles = {
        ("ONLINE", "rotational"): {"color": "tab:gray", "linestyle": "-", "label": "ONLINE rotational"},
        ("ONLINE", "none"):       {"color": "dimgray",  "linestyle": "--", "label": "ONLINE none"},
        ("FBPTT",  "rotational"): {"color": "tab:blue", "linestyle": "-", "label": "FBPTT rotational"},
        ("FBPTT",  "none"):       {"color": "steelblue","linestyle": "--", "label": "FBPTT none"},
    }

    for row_idx, nl in enumerate(layers_sorted):
        ax = axes[row_idx]
        sub = agg[agg["num_layers"] == nl]

        handles = []
        labels = []

        for method in ["ONLINE", "FBPTT"]:
            for mixing in ["rotational", "none"]:
                sm = sub[(sub["method"] == method) & (sub["mixing"] == mixing)].sort_values("memory")
                if sm.empty:
                    continue
                x = sm["memory"].to_numpy()
                y = sm["mean"].to_numpy()
                e = sm["sem"].to_numpy()
                style = combo_styles[(method, mixing)]
                h = ax.errorbar(
                    x, y, yerr=e, linewidth=2, capsize=3,
                    color=style["color"], linestyle=style["linestyle"],
                    alpha=0.95, marker=""
                )
                handles.append(h)
                labels.append(style["label"])

                # Annotate each point with its y-value (loss)
                # Annotate points; push first/last more to the sides to avoid overlap with lines
                n_pts = len(x)
                for idx, (xi, yi) in enumerate(zip(x, y)):
                    # Place annotation above the point for 'none', below for 'rotational'
                    is_none = (mixing == "none")
                    base_offset_y = 3 if is_none else -3
                    base_va = "bottom" if is_none else "top"

                    # Side overflow for endpoints: left for first, right for last
                    if idx == 0:
                        offset = (-2.5, base_offset_y)
                        ha = "right"
                    elif idx == n_pts - 1:
                        offset = (2.5, base_offset_y)
                        ha = "left"
                    else:
                        offset = (2.5, base_offset_y)
                        ha = "left"

                    ax.annotate(
                        f"{yi:.2e}",
                        xy=(xi, yi),
                        xytext=offset,
                        textcoords="offset points",
                        fontsize=7,
                        color=style["color"],
                        ha=ha,
                        va=base_va,
                        alpha=0.9,
                    )

        ax.legend(handles, labels, frameon=True, ncol=2, fontsize=8, loc='lower right')
        ax.set_title(f"Layers = {nl} | Hidden = {args.hidden}", fontsize=12)
        ax.set_ylabel("MSE", fontsize=11)
        ax.grid(True, which="both", axis="y", linestyle="-", alpha=0.25)
        if args.ylog10:
            ax.set_yscale("log")
            # Start y-axis at 10^0 and below
            ax.set_ylim(top=1)
            ax.set_ylim(bottom=1e-3)

    axes[-1].set_xlabel("Memory", fontsize=12)
    axes[-1].set_xticks(args.memories)

    fig.tight_layout()

    # Output name encodes layers list and fixed hidden, plus activation
    if args.outfile:
        grid_name = args.outfile
    else:
        layers_str = "-".join(str(v) for v in layers_sorted)
        grid_name = (
            f"tgap_mixings_grid{K}x1_layers-{layers_str}_H{args.hidden}_{args.activation}.png"
        )
    grid_path = args.outdir / grid_name
    fig.savefig(grid_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {grid_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
