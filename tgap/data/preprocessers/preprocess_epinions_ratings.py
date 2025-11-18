# preprocess_epinions_ratings.py
import numpy as np, zipfile, os
from typing import Tuple
from pathlib import Path

def _load_epinions_txt_from_zip(zip_path: str) -> np.ndarray:
    """
    Returns array of shape [N,6] with columns:
      user, product, category, rating, helpfulness, timestamp
    """
    with zipfile.ZipFile(zip_path) as zf:
        # find the txt inside the zip (there should be a single ratings file)
        members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        assert len(members) >= 1, "No .txt file found in the zip"
        with zf.open(members[0]) as f:
            # columns are space or tab separated floats/ints; use numpy loadtxt
            arr = np.loadtxt(f)
    return arr

def _map_to_contiguous(ids: np.ndarray):
    uniq, inv = np.unique(ids, return_inverse=True)
    mapping = {int(uniq[i]): int(i) for i in range(len(uniq))}
    return inv.astype(np.int32), mapping

def _time_sincos(ts: np.ndarray):
    hod = (ts % 86400) / 3600.0
    dow = ((ts // 86400) + 4) % 7
    return np.stack([
        np.sin(2*np.pi*hod/24.0),
        np.cos(2*np.pi*hod/24.0),
        np.sin(2*np.pi*dow/7.0),
        np.cos(2*np.pi*dow/7.0),
    ], axis=1).astype(np.float32)

def _dt_per_user(ts: np.ndarray, users: np.ndarray):
    last = {}
    dt = np.empty_like(ts)
    for i in range(ts.shape[0]):
        u = int(users[i]); t = int(ts[i])
        dt[i] = 0 if u not in last else max(0, t - last[u])
        last[u] = t
    return np.log1p(dt).astype(np.float32)

def preprocess_epinions_ratings_zip(
    zip_path: str,
    out_npz: str,
    *,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    min_events_user: int = 5,
    min_events_item: int = 5,
    as_regression: bool = True,   # True: float target; False: 1..5 integer class
):
    """
    Epinions ratings with timestamps (Tang/MSU). Produces a strictly-chronological stream:
      src=user_id, dst=item_id (offset by num_users), feat=[dt_user, hod_sin, hod_cos, dow_sin, dow_cos], target, t
    """

    # Check if output file already exists
    if Path(out_npz).exists():
        print(f"[*] Output file {out_npz} already exists. Using existing file.")
        return
    
    arr = _load_epinions_txt_from_zip(zip_path)
    users = arr[:, 0].astype(np.int64)
    items = arr[:, 1].astype(np.int64)
    # categories = arr[:, 2].astype(np.int64)  # optional
    ratings = arr[:, 3].astype(np.float32)
    # helpful = arr[:, 4]                      # DO NOT USE as input (may leak)
    ts = arr[:, 5].astype(np.int64)

    # Sort by time (stable)
    order = np.argsort(ts, kind="mergesort")
    users, items, ratings, ts = users[order], items[order], ratings[order], ts[order]

    # Filter sparse users/items (optional but improves stability)
    if min_events_user > 0:
        _, inv_u, cnt_u = np.unique(users, return_inverse=True, return_counts=True)
        keep_u = (cnt_u[inv_u] >= min_events_user)
    else:
        keep_u = np.ones_like(users, dtype=bool)
    if min_events_item > 0:
        _, inv_i, cnt_i = np.unique(items, return_inverse=True, return_counts=True)
        keep_i = (cnt_i[inv_i] >= min_events_item)
    else:
        keep_i = np.ones_like(items, dtype=bool)
    keep = keep_u & keep_i
    users, items, ratings, ts = users[keep], items[keep], ratings[keep], ts[keep]

    # Map to contiguous IDs (bipartite disjoint)
    u_idx, u_map = _map_to_contiguous(users)
    i_idx, i_map = _map_to_contiguous(items)
    num_users = len(u_map)
    num_items = len(i_map)
    src = u_idx
    dst = (i_idx + num_users).astype(np.int32)
    num_nodes = int(num_users + num_items)

    # Features (causal)
    dt_user = _dt_per_user(ts, u_idx)
    cal = _time_sincos(ts)
    feat = np.concatenate([dt_user[:,None], cal], axis=1).astype(np.float32)

    # Target
    if as_regression:
        target = ratings[:, None].astype(np.float32)
    else:
        # classes 1..5  ->  0..4
        target = (ratings.astype(np.int32) - 1)[:, None]

    # Chronological split
    N = src.shape[0]
    i_tr = int(round((1.0 - val_ratio - test_ratio) * N))
    i_va = int(round((1.0 - test_ratio) * N))
    idx_train = np.arange(0, i_tr, dtype=np.int32)
    idx_val   = np.arange(i_tr, i_va, dtype=np.int32)
    idx_test  = np.arange(i_va, N, dtype=np.int32)

    np.savez_compressed(
        out_npz,
        src=src.astype(np.int32),
        dst=dst.astype(np.int32),
        feat=feat,
        target=target,
        t=ts,
        num_nodes=np.array([num_nodes], dtype=np.int32),
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        meta=np.array([b"epinions_ratings"], dtype=object),
    )
    print(f"[*] Saved {out_npz} | N={N} users={num_users} items={num_items} nodes={num_nodes} feat_dim={feat.shape[1]}")
