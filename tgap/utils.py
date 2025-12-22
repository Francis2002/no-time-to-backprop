def print_tree_keys(tree):
        # ASCII tree printer using '|' and '-' to visualize structure
        def shape_info(x):
            try:
                s = getattr(x, "shape", None)
                if s is not None and len(tuple(s)) > 0:
                    return f"  shape={tuple(s)}"
            except Exception:
                pass
            return ""

        def is_mapping(x):
            try:
                return isinstance(x, dict) or hasattr(x, "items")
            except Exception:
                return False

        def iter_children(x):
            if is_mapping(x):
                try:
                    return [(str(k), v) for k, v in x.items()]
                except Exception:
                    return []
            if isinstance(x, (list, tuple)):
                return [(f"[{i}]", v) for i, v in enumerate(x)]
            return []

        def rec(x, prefix=""):
            children = iter_children(x)
            n = len(children)
            if n == 0:
                # leaf at root
                print(prefix + "- leaf" + shape_info(x))
                return

            for i, (label, child) in enumerate(children):
                last = (i == n - 1)
                connector = "|- "
                line = prefix + connector + f"{label}"
                if is_mapping(child) or isinstance(child, (list, tuple)):
                    print(line)
                    rec(child, prefix + ("   " if last else "|  "))
                else:
                    print(line + shape_info(child))

        rec(tree, "")

# ---- Cosine helpers -------------------------------------------------

import numpy as np

def _np_array(x):
    if x is None: return None
    return np.asarray(x)

def _concat_valid(parts):
    vecs = [p.ravel() for p in parts if p is not None]
    return np.concatenate(vecs) if len(vecs) else np.array([])

def _cos(a, b):
    a = a.ravel().astype(float); b = b.ravel().astype(float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return np.nan
    return float(np.dot(a, b) / (na * nb))

def _extract_layer_seq_grads(grads, method, layer_idx, args):
    """
    Returns a dict with keys:
      'nu','theta','gamma_log','B_re','B_im','D', and 'phi' (if mixing in ['rotational', 'rotational_full']) (some may be none)
    """
    layer = grads['cell']['params']['encoder'][f'layers_{layer_idx}']['seq']
    # Some models may lack D; guard with .get
    
    return_dict = {
        'nu':        layer.get('nu',        None),
        'theta':     layer.get('theta',     None),
        'gamma_log': layer.get('gamma_log', None),
        'B_re':      layer.get('B_re',      None),
        'B_im':      layer.get('B_im',      None),
        'D':         layer.get('D',         None),
    }

    if args.mixing in ['rotational', 'rotational_full']:
        return_dict['phi'] = layer.get('phi', None)

    return return_dict

def _layer_group_vectors(grads, method, layer_idx, args):
    """
    Builds concatenated vectors for groups per layer:
      - 'lambda' = [nu, theta]
      - 'gamma'  = [gamma_log]
      - 'B'      = [B_re, B_im]
      - 'phi'    = [phi] (if mixing in ['rotational', 'rotational_full'])
      - 'all'    = all of the above
    Returns a dict of numpy vectors.
    """
    g = _extract_layer_seq_grads(grads, method, layer_idx, args)
    nu        = _np_array(g['nu'])
    theta     = _np_array(g['theta'])
    gamma_log = _np_array(g['gamma_log'])
    B_re      = _np_array(g['B_re'])
    B_im      = _np_array(g['B_im'])

    lam = _concat_valid([nu, theta])
    gam = _concat_valid([gamma_log])
    B   = _concat_valid([B_re, B_im])
    allv = _concat_valid([lam, gam, B])

    if args.mixing in ['rotational', 'rotational_full']:
        phi = _np_array(g['phi'])
        allv = _concat_valid([lam, gam, B, phi])
        return {'lambda': lam, 'gamma': gam, 'B': B, 'phi': phi, 'all': allv}

    return {'lambda': lam, 'gamma': gam, 'B': B, 'all': allv}


def _overall_vectors(grads, method, num_layers, args):
    """
    Concatenate 'all' vectors over all layers into a single vector.
    """
    parts = []
    for li in range(num_layers):
        vecs = _layer_group_vectors(grads, method, li, args)
        if vecs['all'].size:
            parts.append(vecs['all'])
    return np.concatenate(parts) if parts else np.array([])
# --------------------------------------------------------------------

def apply_hpt_to_args(args, hpt, defaults):
    """Mutate args so that everything downstream (model/loss/optimizer/jit) uses the trial values."""

    # Model depth
    if 'num_layers' in hpt:
        args.num_layers = int(hpt['num_layers'])

    # Core capacity
    if 'state_size' in hpt:
        args.num_hidden = int(hpt['state_size'])

    # d_model coupling
    if 'd_model_factor' in hpt:
        args.d_model = max(1, int(round(args.num_hidden * float(hpt['d_model_factor']))))

    # Batching / accumulation
    if 'batch_size' in hpt:
        args.batch_size = int(hpt['batch_size'])
    if 'n_accumulation_steps' in hpt:
        args.num_gradient_accumulation_steps = int(hpt['n_accumulation_steps'])

    # Regularization / loss
    if 'dropout' in hpt:
        args.dropout = float(hpt['dropout'])
    if 'pos_weight' in hpt:
        args.pos_weight = float(hpt['pos_weight'])

    # LR factor (there is NO "rec_learning_rate", only factor)
    if 'rec_learning_factor' in hpt:
        args.rec_learning_factor = float(hpt['rec_learning_factor'])
    else:
        # keep whatever CLI said
        args.rec_learning_factor = float(defaults['rec_learning_factor'])
    
    # Reset steps_for_scheduler per trial if it was auto (0)
    # We'll compute the actual numeric value later after num_steps is known, this is just for the correct branch to trigger if it should be auto
    args.steps_for_scheduler = int(defaults['steps_for_scheduler'])

