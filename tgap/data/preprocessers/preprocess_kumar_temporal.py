import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Tuple, Dict

"""
Preprocessor for temporal datasets (Wikipedia, MOOC, Reddit) from:

Kumar, S., Zhang, X., & Leskovec, J. (2019).
Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks.
In Proceedings of the 25th ACM SIGKDD International Conference on
Knowledge Discovery & Data Mining (KDD '19).

The CSV files for these datasets are expected to have the following format
(without a header row):
<user_id>,<item_id>,<timestamp>,<state_label>,<feature_1>,<feature_2>,...,<feature_N>
"""

# -----------------------------------------------------------------------------
# Helper Functions (copied from preprocess_bitcoin.py)
# -----------------------------------------------------------------------------

def _map_ids(a: np.ndarray) -> Tuple[np.ndarray, Dict[int,int]]:
    """Map raw integer IDs to contiguous [0..N-1]."""
    uniq = np.unique(a)
    m = {int(k): i for i, k in enumerate(uniq)}
    return np.vectorize(lambda z: m[int(z)])(a), m

def _find_nodes_with_less_than_k_events(a: np.ndarray, k: int) -> np.ndarray:
    """Find nodes with less than k events."""
    unique, counts = np.unique(a, return_counts=True)
    return unique[counts < k]

def _time_feats(ts: np.ndarray):
    # ts is UNIX seconds (int64)
    # hour-of-day & day-of-week as sin/cos
    # Ensure ts is int64 for modulo operations
    ts_int = ts.astype(np.int64)
    hod = (ts_int % 86400) / 3600.0                # [0,24)
    dow = ((ts_int // 86400) + 4) % 7              # Thursday 1970-01-01 => 4
    hod_sin = np.sin(2*np.pi * hod/24.0)
    hod_cos = np.cos(2*np.pi * hod/24.0)
    dow_sin = np.sin(2*np.pi * dow/7.0)
    dow_cos = np.cos(2*np.pi * dow/7.0)
    return np.stack([hod_sin, hod_cos, dow_sin, dow_cos], axis=1).astype(np.float32)

def _dt_endpoints(src: np.ndarray, dst: np.ndarray, ts: np.ndarray, num_nodes: int):
    # Ensure ts is int64 for correct diffs
    ts_int = ts.astype(np.int64)
    last_t = np.full((num_nodes,), -1, dtype=np.int64)
    dt_src = np.empty_like(ts, dtype=np.int64)
    dt_dst = np.empty_like(ts, dtype=np.int64)

    for i in range(ts_int.shape[0]):
        u, v, t = int(src[i]), int(dst[i]), ts_int[i]
        lus, lvs = last_t[u], last_t[v]
        
        dt_src[i] = 0 if lus < 0 else (t - lus)
        dt_dst[i] = 0 if lvs < 0 else (t - lvs)
        
        # Ensure deltas are non-negative (can happen with simultaneous events)
        if dt_src[i] < 0: dt_src[i] = 0
        if dt_dst[i] < 0: dt_dst[i] = 0

        last_t[u] = t
        last_t[v] = t
        
    # log1p scale
    return np.log1p(dt_src).astype(np.float32), np.log1p(dt_dst).astype(np.float32)

# -----------------------------------------------------------------------------
# Main Preprocessing Function
# -----------------------------------------------------------------------------

def preprocess_temporal_csv(
    csv_path: str,
    out_npz: str,
    dataset_name: str,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    drop_hod_dow: bool = False,
):
    """
    Preprocess temporal CSVs (Wikipedia, MOOC, Reddit) from the TGN paper.
    
    CSV format:
    <user_id>,<item_id>,<timestamp>,<state_label>,<feature_1>,...,<feature_N>
    
    Produces a strictly-chronological stream with per-event safe features.
    """
    
    # Check if output file already exists
    if Path(out_npz).exists():
        print(f"[*] Output file {out_npz} already exists. Using existing file.")
        return

    print(f"[*] Loading data from {csv_path}...")
    # Load with header=None as the number of feature columns is variable
    # and the provided file may or may not have a header line.
    try:
        df = pd.read_csv(csv_path, skiprows=1, header=None)
    except Exception as e:
        print(f"[!] Error loading CSV: {e}")
        print("Please ensure the CSV file exists and has no header row.")
        return

    # Extract columns
    # Col 0: user_id (source)
    # Col 1: item_id (destination)
    # Col 2: timestamp
    # Col 3: state_label (target)
    # Col 4...: event features
    src = df[0].to_numpy()
    dst = df[1].to_numpy()
    ts = df[2].to_numpy() # Will be cast to int64 later
    target = df[3].astype(np.int32).to_numpy()
    provided_feat = df.iloc[:, 4:].to_numpy(dtype=np.float32)
    
    print(f"[*] Loaded {src.shape[0]} events.")
    print(f"[*] Found {provided_feat.shape[1]} provided features.")

    # Sort by time
    print("[*] Sorting events by timestamp...")
    order = np.argsort(ts, kind="mergesort")
    src, dst, ts, target = src[order], dst[order], ts[order], target[order]
    provided_feat = provided_feat[order]
    
    # Cast timestamp to int64 *after* sorting (as floats)
    ts = ts.astype(np.int64)

    # Map node ids to contiguous
    print("[*] Mapping node IDs...")
    all_nodes = np.concatenate([src, dst])
    all_nodes_mapped, mapping = _map_ids(all_nodes)
    num_nodes = len(mapping)
    src = all_nodes_mapped[:src.shape[0]]
    dst = all_nodes_mapped[src.shape[0]:]

    # Find nodes with less than k events
    k = 2
    low_activity_nodes = _find_nodes_with_less_than_k_events(all_nodes_mapped, k)
    if len(low_activity_nodes) > 0:
        print(f"[!] Warning: found {len(low_activity_nodes)} nodes with only {k-1} {'event' if k == 2 else 'events'}. Their hidden states will be all 0s.")

    # Create features
    # 1. Provided features (from CSV)
    # 2. Temporal features (dt_src, dt_dst)
    # 3. Time features (hour/day)
    print("[*] Generating temporal features...")
    if not drop_hod_dow:
        hod_dow = _time_feats(ts)                          # [N, 4]
    dt_s, dt_d = _dt_endpoints(src, dst, ts, num_nodes) # [N], [N]

    if not drop_hod_dow:
        print(f"    Provided features dim: {provided_feat.shape[1]}")
        print(f"    Temporal dt features dim: 2")
        print(f"    Time hod/dow features dim: 4")

        # Concatenate all features
        feat = np.concatenate([
            provided_feat, 
            dt_s[:,None], 
            dt_d[:,None], 
            hod_dow
        ], axis=1).astype(np.float32)
    else:
        print(f"    Provided features dim: {provided_feat.shape[1]}")
        print(f"    Temporal dt features dim: 2")
        print(f"    (Dropped hod/dow features)")

        # Concatenate provided + dt features only
        feat = np.concatenate([
            provided_feat, 
            dt_s[:,None], 
            dt_d[:,None], 
        ], axis=1).astype(np.float32)

    # Labels (already extracted as 'target')
    # Reshape to [N, 1] if it's not already
    if target.ndim == 1:
        target = target[:, None]

    # Split data
    N = src.shape[0]
    i_tr = int((1.0 - val_ratio - test_ratio) * N)
    i_va = int((1.0 - test_ratio) * N)
    idx_train = np.arange(0, i_tr, dtype=np.int32)
    idx_val   = np.arange(i_tr, i_va, dtype=np.int32)
    idx_test  = np.arange(i_va, N, dtype=np.int32)

    # Save
    print(f"[*] Saving processed data to {out_npz}...")
    np.savez_compressed(
        out_npz,
        src=src.astype(np.int32),
        dst=dst.astype(np.int32),
        feat=feat,
        target=target.astype(np.int32),
        t=ts.astype(np.int64),
        num_nodes=np.array([num_nodes], dtype=np.int32),
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        meta=np.array([dataset_name.encode('utf-8')], dtype=object),
    )
    print(f"[*] Saved {out_npz}  |  N={N}  nodes={num_nodes}  feat_dim={feat.shape[1]}")
    print(f"    train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")