# sampler_stream.py
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional, Dict, Any


def _ensure_2d(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x)
    if x.ndim == 0:
        return x[None, None]
    if x.ndim == 1:
        return x[:, None]
    return x

def _build_tbatch_minibatches_np(
    src: np.ndarray,
    dst: np.ndarray,
    perm: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    JODIE t-Batch assignment (linear time), then chunk into fixed-size minibatches.

    Returns:
      batches_idx: int32 [num_minibatches, B]  (indices into original stream)
      batches_msk: float32 [num_minibatches, B] (1.0 real, 0.0 padding)
    """
    B = int(batch_size)
    assert B > 0
    N = int(perm.shape[0])

    # infer number of nodes from ids present (robust even if num_nodes passed separately)
    n_nodes = int(max(src.max(), dst.max())) + 1
    last = np.zeros((n_nodes,), dtype=np.int32)

    buckets: list[list[int]] = []  # buckets[k-1] stores batch k
    # Iterate in temporal order given by perm
    for idx in perm.tolist():
        u = int(src[idx]); v = int(dst[idx])
        k = int(max(last[u], last[v]) + 1)
        last[u] = k
        last[v] = k
        while len(buckets) < k:
            buckets.append([])
        buckets[k-1].append(int(idx))

    minibatches: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for edges in buckets:
        if len(edges) == 0:
            continue
        edges = np.asarray(edges, dtype=np.int32)
        # chunk
        for start in range(0, len(edges), B):
            chunk = edges[start:start+B]
            m = np.ones((chunk.shape[0],), dtype=np.float32)
            if chunk.shape[0] < B:
                pad = B - chunk.shape[0]
                # pad indices with a valid index (we'll mask them out anyway)
                chunk = np.concatenate([chunk, np.full((pad,), chunk[0], dtype=np.int32)], axis=0)
                m = np.concatenate([m, np.zeros((pad,), dtype=np.float32)], axis=0)
            minibatches.append(chunk)
            masks.append(m)

    if len(minibatches) == 0:
        return np.zeros((0, B), dtype=np.int32), np.zeros((0, B), dtype=np.float32)

    return np.stack(minibatches, axis=0), np.stack(masks, axis=0)


def _build_rearranged_batches_np(
    src: np.ndarray,
    dst: np.ndarray,
    perm: np.ndarray,
    batch_size: int,
    *,
    window_mult: int = 10,
) -> np.ndarray:
    """
    Greedy, window-limited rearrangement.

    We maintain a FIFO buffer of upcoming edges (indices into the stream).
    For each batch:
      - pass 1: greedily pick edges whose endpoints are not yet used in this batch
      - pass 2: if batch not full, fill remaining slots with oldest leftover edges
                (collisions allowed) to keep throughput high.

    This reduces collisions without producing tiny batches.

    Returns:
      batches: int32 array of shape [num_batches, batch_size]
    """
    assert batch_size > 0
    assert window_mult >= 1

    B = batch_size
    N = perm.shape[0]
    N_eff = (N // B) * B
    perm = perm[:N_eff]

    W = window_mult * B
    buf: list[int] = []
    ptr = 0
    batches: list[list[int]] = []

    while True:
        # refill buffer up to W
        while ptr < N_eff and len(buf) < W:
            buf.append(int(perm[ptr]))
            ptr += 1

        if len(buf) < B:
            break  # cannot form another full batch

        used = set()
        batch: list[int] = []

        # pass 1: try to pick collision-free edges from buffer
        k = 0
        while len(batch) < B and k < len(buf):
            i = buf[k]
            u = int(src[i]); v = int(dst[i])
            if (u not in used) and (v not in used):
                batch.append(i)
                used.add(u); used.add(v)
                buf.pop(k)     # remove chosen edge
            else:
                k += 1

        # pass 2: fill remainder with oldest edges (allow collisions)
        while len(batch) < B:
            batch.append(buf.pop(0))

        batches.append(batch)

    return np.asarray(batches, dtype=np.int32)


def get_stream_sampler_from_arrays(
    src: jnp.ndarray,                 # [N] int32
    dst: jnp.ndarray,                 # [N] int32
    feat: jnp.ndarray,                # [N, Df] or [N]
    target: jnp.ndarray,              # [N, Dt] or [N]
    *,
    num_nodes: int,
    loop: bool = True,
    shuffle_each_epoch: bool = True,
    batch_size: Optional[int] = None,
    batching_strategy: str = "none",  # "none" | "rearranged"
    window_mult: int = 10,
) -> Tuple:
    """
    Returns (init, step, num_nodes, num_steps, feature_dim, target_dim).
    Each step yields:
      - streaming: (src_i, dst_i, feat_i, target_i)
      - batched:   (src_B, dst_B, feat_B, target_B)
    """

    src = jnp.asarray(src, dtype=jnp.int32)
    dst = jnp.asarray(dst, dtype=jnp.int32)
    feat = _ensure_2d(feat)
    target = _ensure_2d(target)

    N = int(src.shape[0])
    Df = int(feat.shape[1])
    Dt = int(target.shape[1])

    if batch_size is None:
        num_steps = N
        # perm is stored in state; can reshuffle each epoch in do_wrap
        def init(rng):
            perm = jnp.arange(N, dtype=jnp.int32)
            return ((perm, jnp.int32(0)), rng)

        @jax.jit
        def step(state, _=None):
            (perm, idx), rng = state
            i = perm[idx]
            ev = (src[i], dst[i], feat[i], target[i])

            idx_next = idx + jnp.int32(1)
            wrapped = idx_next >= N

            def do_wrap(args):
                _perm, _rng = args
                if shuffle_each_epoch:
                    _rng, prng = jax.random.split(_rng)
                    _perm = jax.random.permutation(prng, N)
                return (_perm, jnp.int32(0)), _rng

            def no_wrap(args):
                _perm, _rng = args
                return (_perm, idx_next), _rng

            (perm_next, idx_next), rng_next = jax.lax.cond(
                wrapped & jnp.array(loop),
                do_wrap, no_wrap, operand=(perm, rng)
            )
            return ((perm_next, idx_next), rng_next), ev

        return init, step, num_nodes, num_steps, Df, Dt

    # batched
    B = int(batch_size)
    N_eff = (N // B) * B
    num_batches = N_eff // B

    if batching_strategy not in ("none", "rearranged", "tbatch"):
        raise ValueError(f"Unknown batching_strategy={batching_strategy}")
    
    sentinel = np.int32(num_nodes)  # assumes real nodes are [0..num_nodes-1]

    # Precompute the epoch order once, capture in closure.
    # For temporal graphs you likely want shuffle_each_epoch=False,
    # but we support shuffling batch order each epoch anyway.
    if batching_strategy == "none":
        base = jnp.arange(N_eff, dtype=jnp.int32)  # drop remainder explicitly
        # base is a 1D perm
        def init(rng):
            return ((base, jnp.int32(0)), rng)

        @jax.jit
        def step(state, _=None):
            (perm, idx), rng = state

            sl = jax.lax.dynamic_slice(perm, (idx,), (B,))   # [B]
            ev = (src[sl], dst[sl], feat[sl, :], target[sl, :])

            idx_next = idx + jnp.int32(B)
            wrapped = idx_next >= N_eff

            def do_wrap(args):
                _perm, _rng = args
                if shuffle_each_epoch:
                    _rng, prng = jax.random.split(_rng)
                    _perm = jax.random.permutation(prng, N_eff)
                return (_perm, jnp.int32(0)), _rng

            def no_wrap(args):
                _perm, _rng = args
                return (_perm, idx_next), _rng

            (perm_next, idx_next), rng_next = jax.lax.cond(
                wrapped & jnp.array(loop),
                do_wrap, no_wrap, operand=(perm, rng)
            )
            return ((perm_next, idx_next), rng_next), ev

        return init, step, num_nodes, num_batches, Df, Dt

    if batching_strategy == "rearranged":
        # Build batches ONCE in Python/NumPy to avoid per-epoch overhead.
        src_np = np.asarray(jax.device_get(src))
        dst_np = np.asarray(jax.device_get(dst))
        perm_np = np.arange(N_eff, dtype=np.int32)

        batches_np = _build_rearranged_batches_np(
            src_np, dst_np, perm_np, B, window_mult=window_mult
        )  # [num_batches, B]
        batches = jnp.asarray(batches_np)  # captured in closure

        def init(rng):
            return ((batches, jnp.int32(0)), rng)

        @jax.jit
        def step(state, _=None):
            (batches_local, bidx), rng = state

            sl = batches_local[bidx]  # [B]
            ev = (src[sl], dst[sl], feat[sl, :], target[sl, :])

            bidx_next = bidx + jnp.int32(1)
            wrapped = bidx_next >= batches_local.shape[0]

            def do_wrap(args):
                _batches, _rng = args
                if shuffle_each_epoch:
                    _rng, prng = jax.random.split(_rng)
                    order = jax.random.permutation(prng, _batches.shape[0])
                    _batches = _batches[order]
                return (_batches, jnp.int32(0)), _rng

            def no_wrap(args):
                _batches, _rng = args
                return (_batches, bidx_next), _rng

            (batches_next, bidx_next), rng_next = jax.lax.cond(
                wrapped & jnp.array(loop),
                do_wrap, no_wrap, operand=(batches_local, rng)
            )

            return ((batches_next, bidx_next), rng_next), ev

        return init, step, num_nodes, int(batches_np.shape[0]), Df, Dt

    if batching_strategy == "tbatch":
        # Build tbatch minibatches ONCE in Python/NumPy
        src_np = np.asarray(jax.device_get(src))
        dst_np = np.asarray(jax.device_get(dst))
        feat_np = np.asarray(jax.device_get(feat))
        target_np = np.asarray(jax.device_get(target))

        perm_np = np.arange(N, dtype=np.int32)  # IMPORTANT: use all edges in split (no N_eff drop)
        batches_idx_np, batches_msk_np = _build_tbatch_minibatches_np(src_np, dst_np, perm_np, B)

        batches_idx = jnp.asarray(batches_idx_np)  # [Nb, B]
        batches_msk = jnp.asarray(batches_msk_np)  # [Nb, B] float32

        num_batches = int(batches_idx_np.shape[0])

        # constants for padding payload
        zero_feat = jnp.zeros((Df,), dtype=feat.dtype)
        zero_tgt = jnp.zeros((Dt,), dtype=target.dtype)

        def init(rng):
            return ((batches_idx, batches_msk, jnp.int32(0)), rng)

        @jax.jit
        def step(state, _=None):
            (idx_mat, msk_mat, bidx), rng = state

            sl = idx_mat[bidx]  # [B]
            m  = msk_mat[bidx]  # [B] in {0,1}

            s = src[sl]
            d = dst[sl]
            x = feat[sl, :]
            y = target[sl, :]

            # For padded slots (m==0): route to sentinel and zero payload
            m_bool = m > 0.5
            s = jnp.where(m_bool, s, jnp.int32(sentinel))
            d = jnp.where(m_bool, d, jnp.int32(sentinel))

            # broadcast masking for feat/target
            x = jnp.where(m_bool[:, None], x, zero_feat[None, :])
            y = jnp.where(m_bool[:, None], y, zero_tgt[None, :])

            # Here we return target as a tuple (labels, mask) so the loss can mask cleanly
            ev = (s, d, x, (y, m))

            bidx_next = bidx + jnp.int32(1)
            wrapped = bidx_next >= idx_mat.shape[0]

            def do_wrap(args):
                _idx, _msk, _rng = args
                # For temporal graphs you should not shuffle tbatches.
                return (_idx, _msk, jnp.int32(0)), _rng

            def no_wrap(args):
                _idx, _msk, _rng = args
                return (_idx, _msk, bidx_next), _rng

            (idx_next, msk_next, bidx_next), rng_next = jax.lax.cond(
                wrapped & jnp.array(loop),
                do_wrap, no_wrap, operand=(idx_mat, msk_mat, rng)
            )

            return ((idx_next, msk_next, bidx_next), rng_next), ev

        return init, step, num_nodes + 1, num_batches, Df, Dt


def get_stream_sampler_from_npz(
    npz_path: str,
    split: str = "train",               # "train" | "val" | "test"
    loop: bool = True,
    shuffle_each_epoch: bool = True,
    batch_size: Optional[int] = None,
    batching_strategy: str = "none",    # "none" | "rearranged"
    window_mult: int = 10,
):
    """
    Loads a preprocessed .npz and returns (init, step) over the chosen split.
    The .npz must have keys: src, dst, feat, target, t, num_nodes, idx_train, idx_val, idx_test
    """
    data = np.load(npz_path, allow_pickle=True)
    if split == "train":
        idx = data["idx_train"]
    elif split == "val":
        idx = data["idx_val"]
    elif split == "test":
        idx = data["idx_test"]
    else:
        raise ValueError(f"Unknown split={split}")

    src = jnp.asarray(data["src"][idx], dtype=jnp.int32)
    dst = jnp.asarray(data["dst"][idx], dtype=jnp.int32)
    feat = jnp.asarray(data["feat"][idx])
    target = jnp.asarray(data["target"][idx])
    num_nodes = int(data["num_nodes"][0]) if np.ndim(data["num_nodes"]) else int(data["num_nodes"])

    return get_stream_sampler_from_arrays(
        src, dst, feat, target,
        num_nodes=num_nodes,
        loop=loop,
        shuffle_each_epoch=shuffle_each_epoch,
        batch_size=batch_size,
        batching_strategy=batching_strategy,
        window_mult=window_mult,
    )
