# preprocess_bitcoin.py
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

from scipy.fftpack import dst

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
    hod = (ts % 86400) / 3600.0                   # [0,24)
    dow = ((ts // 86400) + 4) % 7                 # Thursday 1970-01-01 => align Mon=0 if desired
    hod_sin = np.sin(2*np.pi * hod/24.0)
    hod_cos = np.cos(2*np.pi * hod/24.0)
    dow_sin = np.sin(2*np.pi * dow/7.0)
    dow_cos = np.cos(2*np.pi * dow/7.0)
    return np.stack([hod_sin, hod_cos, dow_sin, dow_cos], axis=1).astype(np.float32)

def _dt_endpoints(src: np.ndarray, dst: np.ndarray, ts: np.ndarray, num_nodes: int):
    last_t = np.full((num_nodes,), -1, dtype=np.int64)
    dt_src = np.empty_like(ts)
    dt_dst = np.empty_like(ts)
    for i in range(ts.shape[0]):
        u, v, t = int(src[i]), int(dst[i]), int(ts[i])
        lus, lvs = last_t[u], last_t[v]
        dt_src[i] = 0 if lus < 0 else max(0, t - lus)
        dt_dst[i] = 0 if lvs < 0 else max(0, t - lvs)
        last_t[u] = t
        last_t[v] = t
    # log1p scale
    return np.log1p(dt_src).astype(np.float32), np.log1p(dt_dst).astype(np.float32)

def preprocess_bitcoin_csv(
    csv_path: str,
    out_npz: str,
    *,
    use_sign_label: bool = False,     # True -> label in {0,1} for rating>0; False -> regression label in [-10..+10]
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    SNAP Bitcoin OTC or Alpha CSV with header: SOURCE,TARGET,RATING,TIME
    Produces a strictly-chronological stream with per-event safe features.
    """
    import pandas as pd

    # Check if output file already exists
    if Path(out_npz).exists():
        print(f"[*] Output file {out_npz} already exists. Using existing file.")
        return

    df = pd.read_csv(csv_path)
    # Standardize column names
    cols = {c.lower(): c for c in df.columns}
    source = df[cols.get("source", "SOURCE")].to_numpy()
    target = df[cols.get("target", "TARGET")].to_numpy()
    rating = df[cols.get("rating", "RATING")].astype(np.float32).to_numpy()
    ts     = df[cols.get("time", "TIME")].astype(np.int64).to_numpy()

    # Sort by time
    order = np.argsort(ts, kind="mergesort")
    source, target, rating, ts = source[order], target[order], rating[order], ts[order]

    # Map node ids to contiguous
    all_nodes = np.concatenate([source, target])
    all_nodes_mapped, mapping = _map_ids(all_nodes)
    num_nodes = len(mapping)
    src = all_nodes_mapped[:source.shape[0]]
    dst = all_nodes_mapped[source.shape[0]:]

    # Find nodes with less than k events
    k = 2
    low_activity_nodes = _find_nodes_with_less_than_k_events(all_nodes, k)
    if len(low_activity_nodes) > 0:
        print(f"[!] Warning: found {len(low_activity_nodes)} nodes with only {k-1} {'event' if k == 2 else 'events'}. Their hidden states will be all 0s.")

    # Features: [dt_src, dt_dst, hod_sin, hod_cos, dow_sin, dow_cos]
    hod_dow = _time_feats(ts)                     # [N,4]
    dt_s, dt_d = _dt_endpoints(src, dst, ts, num_nodes)
    feat = np.concatenate([dt_s[:,None], dt_d[:,None], hod_dow], axis=1).astype(np.float32)

    # Labels
    if use_sign_label:
        target = (rating > 0.0).astype(np.int32)[:, None]   # [N,1]
    else:
        target = rating[:, None].astype(np.float32)         # [N,1]

    N = src.shape[0]
    i_tr = int((1.0 - val_ratio - test_ratio) * N)
    i_va = int((1.0 - test_ratio) * N)
    idx_train = np.arange(0, i_tr, dtype=np.int32)
    idx_val   = np.arange(i_tr, i_va, dtype=np.int32)
    idx_test  = np.arange(i_va, N, dtype=np.int32)

    np.savez_compressed(
        out_npz,
        src=src.astype(np.int32),
        dst=dst.astype(np.int32),
        feat=feat,
        target=target,
        t=ts.astype(np.int64),
        num_nodes=np.array([num_nodes], dtype=np.int32),
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        meta=np.array([b"bitcoin"], dtype=object),
    )
    print(f"[*] Saved {out_npz}  |  N={N}  nodes={num_nodes}  feat_dim={feat.shape[1]}")
