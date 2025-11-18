# sampler_stream.py
import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict, Any

def _ensure_1d(x):
    x = jnp.asarray(x)
    if x.ndim == 0:
        return x[None]
    return x

def get_stream_sampler_from_arrays(
    src: jnp.ndarray,                 # [N] int32
    dst: jnp.ndarray,                 # [N] int32
    feat: jnp.ndarray,                # [N, Df] or [N] or scalar per-event -> will be 1D
    target: jnp.ndarray,              # [N, Dt] or [N] or scalar per-event -> will be 1D
    *,
    num_nodes: int,
    loop: bool = True,
    shuffle_each_epoch: bool = True,
    batch_size: Optional[int] = None,
    batching_strategy: str = "none",   # "none" | "rearranged"
) -> Tuple:
    """
    Returns (init, step) just like the toy sampler.
    Each step yields one event: (src_i, dst_i, feat_i, target_i).
    """
    # Normalize shapes
    src = jnp.asarray(src, dtype=jnp.int32)
    dst = jnp.asarray(dst, dtype=jnp.int32)
    feat = jnp.asarray(feat)
    target = jnp.asarray(target)

    if feat.ndim == 1:  # [N] -> [N,1]
        feat = feat[:, None]
    if target.ndim == 1:  # [N] -> [N,1]
        target = target[:, None]

    N = src.shape[0]
    assert dst.shape[0] == N and feat.shape[0] == N and target.shape[0] == N, "length mismatch"

    print(f"[*] Created stream sampler with {N} events, {num_nodes} nodes (total regardless of splits), feat dim {feat.shape[1]}, target dim {target.shape[1]}")

    print(f"Number of positive labels: {jnp.sum(target == 1)}")

    # We capture arrays in the closure; JIT will treat them as constants.
    def init(rng):
        perm = jnp.arange(N, dtype=jnp.int32)
        return ((perm, jnp.int32(0)), rng)

    @jax.jit
    def step(state, _=None):
        (perm, idx), rng = state

        if batch_size is None:
          i = perm[idx]

          ev_src   = src[i]
          ev_dst   = dst[i]
          ev_feat  = feat[i]        # [Df]
          ev_tgt   = target[i]      # [Dt]

          idx_next = idx + jnp.int32(1)

        else:
          if batching_strategy == "none":
            i = jax.lax.dynamic_slice(perm, (jnp.minimum(idx, N - batch_size),), (batch_size,)) # [B]

            ev_src   = src[i]            # [B]
            ev_dst   = dst[i]            # [B]
            ev_feat  = feat[i, :]        # [B,Df]
            ev_tgt   = target[i, :]      # [B,Dt]

            idx_next = idx + jnp.int32(batch_size)

          elif batching_strategy == "rearranged":
              raise NotImplementedError("Rearranged batching not implemented yet")
            
        wrapped  = idx_next >= N

        def do_wrap(args):
            # [!] Wrapping means restarting from scratch, without finishing the epoch. There should be no more events this epoch after this point.
            (_perm, _rng) = args
            if shuffle_each_epoch:
                _rng, prng = jax.random.split(_rng)
                new_perm = jax.random.permutation(prng, N)
            else:
                new_perm = _perm
            return (new_perm, jnp.int32(0)), _rng

        def no_wrap(args):
            (_perm, _rng) = args
            return (_perm, idx_next), _rng

        (perm_next, idx_next), rng_next = jax.lax.cond(
            wrapped & jnp.array(loop),
            do_wrap, no_wrap, operand=(perm, rng)
        )

        # If loop=False and we wrap, we still keep returning from index 0; we can stop externally if desired.

        next_state = ((perm_next, idx_next), rng_next)
        return next_state, (ev_src, ev_dst, ev_feat, ev_tgt)

    return init, step, num_nodes, N if batch_size is None else N // batch_size, feat.shape[1], target.shape[1]


def get_stream_sampler_from_npz(
    npz_path: str,
    split: str = "train",                   # "train" | "val" | "test"
    loop: bool = True,
    shuffle_each_epoch: bool = True,
    batch_size: Optional[int] = None,
    batching_strategy: str = "none",   # "none" | "rearranged"
):
    """
    Loads a preprocessed .npz and returns (init, step) over the chosen split.
    The .npz must have keys: src, dst, feat, target, t, num_nodes, idx_train, idx_val, idx_test
    """
    import numpy as _np
    data = _np.load(npz_path, allow_pickle=True)

    idx_map = {
        "train": data["idx_train"],
        "val":   data["idx_val"],
        "test":  data["idx_test"],
    }
    idx = idx_map[split].astype(_np.int32)

    src = jnp.asarray(data["src"][idx], dtype=jnp.int32)
    dst = jnp.asarray(data["dst"][idx], dtype=jnp.int32)
    feat = jnp.asarray(data["feat"][idx])
    target = jnp.asarray(data["target"][idx])
    num_nodes = int(data["num_nodes"])

    return get_stream_sampler_from_arrays(
        src, dst, feat, target,
        num_nodes=num_nodes,
        loop=loop,
        shuffle_each_epoch=shuffle_each_epoch,
        batch_size=batch_size,
        batching_strategy=batching_strategy
    )
