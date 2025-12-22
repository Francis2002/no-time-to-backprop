#!/usr/bin/env python3
"""
Build an apples-to-apples ROC AUC table for node classification on Wikipedia/Reddit/MOOC.

Inputs:
  1) A CSV produced by run_toy_task.py (default: results_for_rotational_vs_none/<dataset>_<METHOD>/global_results.csv)
     It must contain:
       - dataset
       - method
       - architecture
       - mixing
       - best_val_roc_auc
       - test_roc_auc_at_best_val_epoch

  2) A baseline JSON (optional) with reported numbers from a single paper (recommended: Deep Graph Sprints):
       {
         "source": "...",
         "metric": "roc_auc",
         "values": {"TGN": {"wikipedia": 0.705, ...}, ...}
       }

Usage:
  python make_auc_table.py --csv path/to/global_results.csv --baselines baseline_results.json --out_md table.md --out_tex table.tex
"""
import argparse
import json
from pathlib import Path

import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str)
    ap.add_argument("--baselines", default="", type=str)
    ap.add_argument("--out_md", default="auc_table.md", type=str)
    ap.add_argument("--out_tex", default="auc_table.tex", type=str)
    ap.add_argument("--datasets", default="wikipedia,reddit,mooc", type=str)
    ap.add_argument("--metric_col", default="test_roc_auc_at_best_val_epoch", type=str)
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    df = pd.read_csv(args.csv)
    if "dataset" not in df.columns:
        raise ValueError("CSV must include a 'dataset' column (patched run_toy_task_patched.py adds it).")

    # Keep only relevant datasets and valid metric
    df = df[df["dataset"].isin(datasets)].copy()
    df = df[df[args.metric_col].notna()].copy()

    # Pick best run per (dataset, method, architecture, mixing) by validation AUC, then by test-at-best-val AUC
    group_cols = ["dataset", "method", "architecture", "mixing"]
    sort_cols = ["best_val_roc_auc", args.metric_col]
    df_best = (df.sort_values(sort_cols, ascending=[False, False])
                 .groupby(group_cols, as_index=False)
                 .head(1)
                 .copy())

    # Build a wide table for "our methods"
    df_best["model"] = df_best["method"] + "/" + df_best["architecture"] + "/" + df_best["mixing"]
    wide = df_best.pivot_table(index="model", columns="dataset", values=args.metric_col, aggfunc="first")
    wide = wide.reindex(columns=datasets)

    # Add baselines if provided
    if args.baselines:
        b = json.loads(Path(args.baselines).read_text())
        for model_name, vals in (b.get("values") or {}).items():
            row = {d: vals.get(d) for d in datasets}
            wide.loc[model_name] = row

    # Format
    wide = wide.sort_index()
    wide_md = wide.copy()
    for c in wide_md.columns:
        wide_md[c] = wide_md[c].map(lambda x: "" if pd.isna(x) else f"{float(x)*100:.2f}")
    md = wide_md.to_markdown()
    Path(args.out_md).write_text(md)

    # LaTeX
    wide_tex = wide.copy()
    for c in wide_tex.columns:
        wide_tex[c] = wide_tex[c].map(lambda x: "" if pd.isna(x) else f"{float(x)*100:.2f}")
    tex = wide_tex.to_latex(escape=False)
    Path(args.out_tex).write_text(tex)

    print(f"Wrote: {args.out_md} and {args.out_tex}")

if __name__ == "__main__":
    main()
