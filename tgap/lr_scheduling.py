import jax
import jax.numpy as jnp

# --- Schedules ---------------------------------------------------------------
def linear_warmup(step, base_lr, end_step, lr_min=None):
    return base_lr * (step + 1) / max(1, end_step)

def cosine_annealing(step, base_lr, end_step, lr_min=1e-6):
    count = jnp.minimum(step, end_step)
    cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * count / jnp.maximum(1, end_step)))
    return (base_lr - lr_min) * cosine_decay + lr_min

def constant_lr(step, base_lr, end_step, lr_min=None):
    return base_lr

def make_schedule(kind: str, base_lr: float, end_step: int, lr_min: float = 1e-6, warmup_steps: int = 0):
    if kind == "cosine":
        return lambda step: cosine_annealing(step, base_lr, end_step, lr_min)
    elif kind == "linear_warmup":
        return lambda step: linear_warmup(step, base_lr, end_step)
    elif kind == "constant":
        return lambda step: constant_lr(step, base_lr, end_step)
    elif kind == "warmup_cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=end_step,
            end_value=lr_min,
        )
    else:
        raise ValueError(f"Unknown schedule kind: {kind}")

# --- Param labeling for multi_transform -------------------------------------
RECURRENT_LEAF_NAMES = {"nu", "theta", "gamma_log", "B_re", "B_im"}  # recurrent, no WD and rec_base_lr
# Everything else (encoder/decoder, C_re/C_im, D, GLU/Norm, etc.) -> weight decay.

def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {k: (map_fn(v) if hasattr(v, "keys") else fn(k, v)) for k, v in nested_dict.items()}

    return map_fn

def label_param_tree(params):
    # Build a parallel pytree with labels 'rec' or 'nonrec' at the leaves.
    def label_leaf(_leaf):
        # We don't know the key name here; use a closure trick by returning a callable
        return _leaf  # placeholder — replaced below

    # Walk with access to keys by rebuilding manually:
    def walk(d):
        if not hasattr(d, "keys"):
            # leaf array (already caught above if needed)
            return "nonrec"
        out = {}
        for k, v in d.items():
            if hasattr(v, "keys"):
                out[k] = walk(v)
            else:
                out[k] = ("rec" if k in RECURRENT_LEAF_NAMES else "nonrec")
        return out
    return walk(params)

# --- Param counter (counts complex as 2 reals) -------------------------------
def count_params(params):
    def is_complex(x):
        return getattr(x, "dtype", None) in (jnp.complex64, jnp.complex128)
    def _size(x):
        try:
            n = x.size
            return int(n * (2 if is_complex(x) else 1))
        except Exception:
            return 0
    if 'params' not in params['cell']:
        param_tree = params['cell']
    else:
        param_tree = params['cell']['params']
    sizes = jax.tree_util.tree_map(_size, param_tree)
    return int(sum(jax.tree_util.tree_leaves(sizes)))

# --- Learning rate scheduling with optax -------------------------------------

import optax

def create_optimizer(args, hpt):

    reg_base_lr = float(hpt['learning_rate'])
    rec_base_lr = float(hpt['learning_rate']) * float(args.rec_learning_factor)
    wd = float(hpt['weight_decay'])
    b1, b2 = float(hpt['beta1']), float(hpt['beta2'])

    # Learning-rate schedules (no manual state)
    warmup_steps = int(args.steps_for_scheduler * args.warmup_frac)
    reg_schedule = make_schedule(args.lr_schedule, reg_base_lr, args.steps_for_scheduler, args.lr_min, warmup_steps)
    rec_schedule = make_schedule(args.lr_schedule, rec_base_lr, args.steps_for_scheduler, args.lr_min, warmup_steps)

    # group transforms
    rec_tx    = optax.adam(  learning_rate=rec_schedule, b1=b1, b2=b2)              # no WD
    nonrec_tx = optax.adamw( learning_rate=reg_schedule, b1=b1, b2=b2, weight_decay=wd)  # with WD

    # Two groups: recurrent vs nonrecurrent
    # - rec: no weight decay (adam)
    # - nonrec: decoupled wd (adamw)
    #param_labels = label_param_tree(params)  # uses leaf names to assign group

    rec_fn = map_nested_fn(
        lambda k, _: "rec"
        if k in RECURRENT_LEAF_NAMES
        else "nonrec"
    )

    tx = optax.multi_transform(
      {"rec": rec_tx, "nonrec": nonrec_tx},
      rec_fn
    )
    # Gradient accumulation if you want to keep it
    if not args.acc:
        tx = optax.MultiSteps(tx, every_k_schedule=args.num_gradient_accumulation_steps, use_grad_mean=True)

    return tx
