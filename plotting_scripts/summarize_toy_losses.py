#!/usr/bin/env python3
"""
summarize_toy_losses.py
-----------------------
Summarize toy-task losses as mean ± s.e.m. across seeds for FBPTT, TBPTT, and ONLINE.

You give:
  - paths to the three CSVs (FBPTT, TBPTT, ONLINE)
  - a fixed memory horizon (e.g., M=3)
  - a set of num_layers (e.g., 1 2 4)
  - a set of hidden sizes (state_size) (e.g., 32 64 128)

The script filters rows and then either:
  - produces one column per (layers, hidden) combination, or
  - if --aggregate_over_hidden is on, it averages over the provided hidden sizes and gives one column per 'layers' only.

By default it uses `average_final_100_loss` if present, else `final_loss`, else `best_loss`.

Usage examples
--------------
# Fixed hidden size, multiple layers → columns like: LRU(L=1,H=64), LRU(L=2,H=64), ...
python summarize_toy_losses.py \
  --fbptt_csv /path/to/results_from_gpu/toy_FBPTT/global_results.csv \
  --tbptt_csv /path/to/results_from_gpu/toy_TBPTT/global_results.csv \
  --online_csv /path/to/results_from_gpu/toy_ONLINE/global_results.csv \
  --memory 3 --layers 1 2 4 --hidden 64 \
  --activation full_glu --mixing full

# Multiple hidden sizes, aggregate over hidden → one column per layer
python summarize_toy_losses.py \
  --fbptt_csv ... --tbptt_csv ... --online_csv ... \
  --memory 3 --layers 1 2 4 --hidden 32 64 128 \
  --aggregate_over_hidden \
  --activation full_glu --mixing full \
  --latex_out rows.tex

Output
------
- A pretty console table
- Optionally: LaTeX rows (three lines: FBPTT, TBPTT, ONLINE) to paste into your table.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product

METHOD_ORDER = ["FBPTT", "TBPTT", "ONLINE"]

def pick_metric(df: pd.DataFrame) -> str:
    if "average_final_100_loss" in df.columns:
        return "average_final_100_loss"
    if "final_loss" in df.columns:
        return "final_loss"
    return "best_loss"

def load_and_filter(csv_path: Path,
                    method_name: str,
                    memory: int,
                    layers: list[int],
                    hidden: list[int],
                    architecture: str,
                    activation: str,
                    mixing: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize method field (your CSV writer already stores 'method')
    df["method"] = df["method"].astype(str).str.upper()

    mask = (
        (df["method"] == method_name) &
        (df["architecture"] == architecture) &
        (df["memory"] == memory) &
        (df["num_layers"].isin(layers)) &
        (df["state_size"].isin(hidden)) &
        (df["activation"] == activation) &
        (df["mixing"] == mixing)
    )
    out = df.loc[mask].copy()
    if out.empty:
        raise SystemExit(f"[{method_name}] No rows after filtering. Check CSV and filters.")
    return out

def mean_sem(series: pd.Series) -> tuple[float, float, int]:
    vals = series.to_numpy(dtype=float)
    n = len(vals)
    m = float(np.mean(vals)) if n > 0 else np.nan
    s = float(np.std(vals, ddof=1)/np.sqrt(n)) if n > 1 else 0.0
    return m, s, n

def fmt_val(mean: float, sem: float, fmt: str) -> str:
    # fmt can be like '%.3e' or '%.4f'
    return f"{fmt % mean} ± {fmt % sem}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbptt_csv", type=Path, required=True)
    ap.add_argument("--tbptt_csv", type=Path, required=True)
    ap.add_argument("--online_csv", type=Path, required=True)

    ap.add_argument("--memory", type=int, required=True, help="Fixed memory horizon M")
    ap.add_argument("--layers", type=int, nargs="+", required=True, help="Set of num_layers (e.g. 1 2 4)")
    ap.add_argument("--hidden", type=int, nargs="+", required=True, help="Set of hidden sizes (state_size)")

    ap.add_argument("--architecture", type=str, default="ZUC")
    ap.add_argument("--activation", type=str, default="full_glu")
    ap.add_argument("--mixing", type=str, default="full")

    ap.add_argument("--aggregate_over_hidden", action="store_true",
                    help="If set, average results across the provided hidden sizes; columns become layers only.")
    ap.add_argument("--metric", type=str, default="auto",
                    help="Which metric column to summarize: {auto, average_final_100_loss, final_loss, best_loss}")
    ap.add_argument("--fmt", type=str, default="%.3e", help="Numeric format for mean and sem (default: %.3e)")
    ap.add_argument("--latex_out", type=Path, default=None, help="If set, write LaTeX rows to this file.")
    args = ap.parse_args()

    dfs = {}
    for method, path in zip(METHOD_ORDER, [args.fbptt_csv, args.tbptt_csv, args.online_csv]):
        dfs[method] = load_and_filter(path, method, args.memory, args.layers, args.hidden,
                                      args.architecture, args.activation, args.mixing)

    # Decide metric
    metric = args.metric
    if metric == "auto":
        # pick from union of columns
        union_cols = set().union(*[set(df.columns) for df in dfs.values()])
        if "average_final_100_loss" in union_cols:
            metric = "average_final_100_loss"
        elif "final_loss" in union_cols:
            metric = "final_loss"
        else:
            metric = "best_loss"

    # Build column keys
    if args.aggregate_over_hidden:
        columns = [(L, None) for L in sorted(set(args.layers))]
        col_labels = [f"LRU ({L}L)" for (L, _) in columns]
    else:
        columns = [(L, H) for (L, H) in product(sorted(set(args.layers)),
                                               sorted(set(args.hidden)))]
        col_labels = [f"LRU (L={L}, H={H})" for (L, H) in columns]

    # Aggregate
    table = {method: [] for method in METHOD_ORDER}
    counts = {method: [] for method in METHOD_ORDER}  # n per cell (for sanity)
    for method in METHOD_ORDER:
        df = dfs[method]
        for (L, H) in columns:
            sub = df[df["num_layers"] == L]
            if H is not None:
                sub = sub[sub["state_size"] == H]
            # mean±sem across seeds (and across hidden if aggregated)
            m, s, n = mean_sem(sub[metric])
            table[method].append(fmt_val(m, s, args.fmt))
            counts[method].append(n)

    # Pretty print
    header = ["Method"] + col_labels
    row_widths = [max(len(h), 6) for h in header]
    def pad(s, w): return s + " " * (w - len(s))
    print("\n=== Toy MSE (mean ± s.e.m.) ===")
    print("Memory M =", args.memory,
          "| Activation:", args.activation,
          "| Mixing:", args.mixing,
          "| Arch:", args.architecture)
    print("-" * (sum(row_widths) + 3*len(row_widths)))
    # header
    line = " | ".join(pad(h, w) for h, w in zip(header, row_widths))
    print(line)
    print("-" * (sum(row_widths) + 3*len(row_widths)))
    # rows
    for method in METHOD_ORDER:
        cells = [method] + table[method]
        line = " | ".join(pad(c, w) for c, w in zip(cells, row_widths))
        print(line)
    print("-" * (sum(row_widths) + 3*len(row_widths)))
    print("Counts (N runs per cell):")
    for method in METHOD_ORDER:
        cells = [method] + [str(n) for n in counts[method]]
        line = " | ".join(pad(c, w) for c, w in zip(cells, row_widths))
        print(line)
    print()

    # Optional: write LaTeX rows (three lines to paste under your tabular)
    if args.latex_out is not None:
        with open(args.latex_out, "w") as f:
            for method in METHOD_ORDER:
                row = method
                for val in table[method]:
                    # Convert "a ± b" to "a \\pm b"
                    val_tex = val.replace(" ± ", r" $\pm$ ")
                    row += " & " + val_tex
                row += r" \\"
                f.write(row + "\n")
        print(f"[LaTeX] Wrote rows to: {args.latex_out}")
        # Example output:
        # FBPTT & 1.234e-04 $\pm$ 2.1e-05 & ... \\
        # TBPTT & ...
        # ONLINE & ...
        # Paste these beneath your \midrule in the tabular body.
if __name__ == "__main__":
    main()
