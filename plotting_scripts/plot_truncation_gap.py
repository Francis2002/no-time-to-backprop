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
  and plots 3 color groups (FBPTT/TBPTT/ONLINE) with shade variations for each chosen value.
- X-axis = memory horizon (integer ticks); Y-axis = MSE (log scale by default).
- For each memory value, the figure shows mean ± s.e.m. across seeds.
- The output filename auto-encodes whether you varied hidden or layers and the chosen values.

Usage
=====
Example: vary hidden sizes (fix layers):
    python plot_truncation_gap.py \
      --fbptt_csv /.../results_from_gpu/toy_FBPTT/global_results.csv \
      --tbptt_csv /.../results_from_gpu/toy_TBPTT/global_results.csv \
      --online_csv /.../results_from_gpu/toy_ONLINE/global_results.csv \
      --vary hidden --hidden 32 64 128 --layers 1 \
      --memories 1 2 3 4 5 --mixing full --activation full_glu \
      --outdir ./figs

Example: vary number of layers (fix hidden size):
    python plot_truncation_gap.py \
      --fbptt_csv ... --tbptt_csv ... --online_csv ... \
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
               mixing: str,
               activation: str,
               layers: List[int],
               hidden: List[int],
               memories: List[int]) -> pd.DataFrame:
    mask = (
        (df["architecture"] == architecture) &
        (df["mixing"] == mixing) &
        (df["activation"] == activation) &
        (df["memory"].isin(memories)) &
        (df["num_layers"].isin(layers)) &
        (df["state_size"].isin(hidden))
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
    p.add_argument("--tbptt_csv", type=Path, required=True)
    p.add_argument("--online_csv", type=Path, required=True)

    p.add_argument("--architecture", type=str, default="ZUC")
    p.add_argument("--activation", type=str, default="full_glu")
    p.add_argument("--mixing", type=str, default="full")

    p.add_argument("--vary", choices=["hidden", "layers"], required=True,
                   help="Vary hidden sizes or number of layers in the figure (the other is fixed).")

    p.add_argument("--hidden", type=int, nargs="+", required=True,
                   help="If varying layers: one value. If varying hidden: many values.")
    p.add_argument("--layers", type=int, nargs="+", required=True,
                   help="If varying hidden: one value. If varying layers: many values.")

    p.add_argument("--memories", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--ylog10", action="store_true", default=True)
    p.add_argument("--title", type=str, default="Truncation Gap")
    p.add_argument("--outdir", type=Path, default=Path("./figs"))
    p.add_argument("--outfile", type=str, default="")
    p.add_argument("--metric", type=str, default="auto",
                   help="One of {auto, average_final_100_loss, final_loss, best_loss}.")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for path, method in [
        (args.fbptt_csv, "FBPTT"),
        (args.tbptt_csv, "TBPTT"),
        (args.online_csv, "ONLINE"),
    ]:
        df = pd.read_csv(path)
        df["method"] = df["method"].astype(str).str.upper()
        df = df[df["method"] == method]
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    metric = _pick_metric(df_all) if args.metric == "auto" else args.metric

    df_all = _filter_df(
        df_all,
        architecture=args.architecture,
        mixing=args.mixing,
        activation=args.activation,
        layers=args.layers,
        hidden=args.hidden,
        memories=args.memories,
    )
    if df_all.empty:
        raise SystemExit("No rows after filtering. Check CSV paths and filters.")

    vary_field = "state_size" if args.vary == "hidden" else "num_layers"
    agg = _agg_over_seeds(df_all, metric, ["method", "memory", vary_field])

    plt.figure(figsize=(8.8, 6.0))
    ax = plt.gca()

    palettes = {
        "FBPTT": "Blues",
        "TBPTT": "Oranges",
        "ONLINE": "Greys",
    }
    var_vals = sorted(agg[vary_field].unique())

    legend_labels = []
    legend_handles = []
    
    for method in ["TBPTT", "ONLINE", "FBPTT"]:
      sub = agg[agg["method"] == method]
      if sub.empty:
        continue
      shades = _shades(palettes[method], len(var_vals))
      
      # Create a dummy handle for the method group header
      method_handle = plt.Line2D([0], [0], color='black', linewidth=0, alpha=0)
      legend_handles.append(method_handle)
      legend_labels.append(f"{method if method != 'ONLINE' else 'Ours'}:")

      for i, vv in enumerate(var_vals):
        ss = sub[sub[vary_field] == vv].sort_values("memory")
        x = ss["memory"].to_numpy()
        y = ss["mean"].to_numpy()
        e = ss["sem"].to_numpy()

        # plot mean with errorbars
        line = plt.errorbar(x, y, yerr=e, marker="", linewidth=2, capsize=3, 
                   label=f"  {vary_field}={vv}", color=shades[i], alpha=0.95)
        legend_handles.append(line)
        legend_labels.append(f"  {vary_field}={vv}")

        #(line,) = ax.plot(x, y, marker="o", linewidth=2.0,
            #                  label=f"{method} {vary_field}={vv}", alpha=0.95,
            #                  color=shades[i])
            #ax.fill_between(x, np.maximum(y - e, 1e-12), y + e, alpha=0.15, color=shades[i])

    ax.set_xlabel("Memory", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_xticks(args.memories)
    if args.ylog10:
      ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", linestyle="-", alpha=0.25)
    ax.set_title(args.title, fontsize=13)
    
    # Place legend outside plot area to avoid overlap
    # ax.legend(legend_handles, legend_labels, frameon=True, ncol=1, fontsize=9, 
    #       bbox_to_anchor=(1.05, 1), loc='upper left')

    if args.outfile:
      outname = args.outfile
    else:
      if args.vary == "hidden":
        vals = "-".join(str(v) for v in sorted(args.hidden))
        outname = f"tgap_vary-hidden_{vals}_L{args.layers[0]}_{args.activation}_{args.mixing}.png"
      else:
        vals = "-".join(str(v) for v in sorted(args.layers))
        outname = f"tgap_vary-layers_{vals}_H{args.hidden[0]}_{args.activation}_{args.mixing}.png"

    outpath = args.outdir / outname
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
