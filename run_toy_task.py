import argparse
import numpy as np
import jax

import jax.numpy as jnp
import optax
from jax.tree_util import tree_map
from jax import lax
from functools import partial
from pathlib import Path

from tgap.grnn import SymmetricGRUCell, MultiLayerLRUCellReal, SymmetricMinGRUCell, ZucchetCell
from tgap.gloss import bce, mse, MLP, with_loss, with_loss_without_mlp, with_feedforward_loss, with_feedforward_and_truncated_grads, metrics_fake_loss, compute_metrics_from_logits
from tgap.memory import state_store, store_set_dedupe, store_get, memory_store
from tgap.data.buffer_task import get_sampler_link_regression
from tgap.rec import LRU
from hpt import GridSampler
import csv
import os

from tgap.data.sampler_stream import get_stream_sampler_from_npz
from tgap.data.preprocessers.preprocess_bitcoin import preprocess_bitcoin_csv
from tgap.data.preprocessers.preprocess_epinions_ratings import preprocess_epinions_ratings_zip
from tgap.data.preprocessers.preprocess_wikirfa import preprocess_wikirfa_gz
from tgap.data.preprocessers.preprocess_kumar_temporal import preprocess_temporal_csv
import sys

from tgap.lr_scheduling import create_optimizer, count_params
from tgap.utils import print_tree_keys

import optuna

#jax.config.update('jax_disable_jit', True)

parser = argparse.ArgumentParser('Truncation Gap on Toy Data')
parser.add_argument('-m', '--method', type=str, choices=['FBPTT', 'TBPTT', 'ONLINE', 'SPATIAL', 'ALL0'], help='Method name (FBPTT or TBPTT or ONLINE or SPATIAL)')
parser.add_argument('-a', '--architecture', type=str, choices=['GRU', 'LRU', 'MIN', 'ZUC'], help='Cell architecture (GRU or LRU)')
parser.add_argument('--dedupe', action='store_true', help='Dedupe memory updates')
parser.add_argument('--dataset', type=str, default='toy', help='Dataset to use (toy, bitcoin_otc, bitcoin_alpha, wiki_rfa, epinions_ratings)')
parser.add_argument('--task', type=str, default='link_classification', help='Task to use (link_regression, link_classification)')
parser.add_argument('--batching_strategy', type=str, default='none', help='Batching strategy (none or rearranged)')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=0, help='Batch Size')
parser.add_argument('--num_epochs', type=int, default=4000, help='Number of epochs')
parser.add_argument('--num_steps', type=int, default=750, help='Number of steps per epoch')
parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
parser.add_argument('--num_hidden', type=int, default=32, help='Number of hidden units')
parser.add_argument('--d_model', type=int, default=16, help='Model dimension (only for LRU and ZUC), ignored if hpt is not optuna')
parser.add_argument('--memory', type=int, default=2, help='Number of memory units')
parser.add_argument('--activation', type=str, default='none', help='Activation function (tanh, sigmoid, gelu or full_glu or half_glu1 or half_glu2 or none)')
parser.add_argument('--prenorm', action='store_true', help='Use prenormalization')
parser.add_argument('--postnorm', action='store_true', help='Use postnormalization')
parser.add_argument('--encoder', action='store_true', help='Use encoder')
parser.add_argument('--layer_output', action='store_true', help='Use layer output')
parser.add_argument('--extra_skip', action='store_true', help='Use extra skip connection')
parser.add_argument('--decoder', type=str, default='MLP', help='Decoder type (MLP or NONE)')
parser.add_argument('--mixing', type=str, default='full', help='State coupling strategy (full, symmetric, none, rotational)')
parser.add_argument('--has_non_linearity_in_recurrence', action='store_true', help='Use non-linearity inside the recurrent cell (only for LRU and ZUC)')
parser.add_argument('--dont_store_results', action='store_true', help='Do not store results to disk')
parser.add_argument('--double_dmodel', action='store_true', help='Use d_model = 2 * num_hidden (only for LRU and ZUC)')
parser.add_argument('--equal_dmodel', action='store_true', help='Use d_model = num_hidden (only for LRU and ZUC)')
parser.add_argument('--acc', action='store_true', help='Accumulate gradients over the entire unrolled segment (for non-FBPTT methods)')
parser.add_argument('--num_gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps (default: 1)')

# ---------------------------------------------------- LR Scheduling ----------------------------------------------------

parser.add_argument('--steps_for_scheduler', type=int, default=20, help='Number of steps for the LR scheduler (default: 20)')
parser.add_argument('--lr_schedule', type=str, default='constant', choices=['cosine','linear_warmup','constant'])
parser.add_argument('--rec_lr_schedule', type=str, default='constant', choices=['cosine','linear_warmup','constant'])
parser.add_argument('--use_plateau', action='store_true', help='Enable ReduceLROnPlateau on top of the base schedule')
parser.add_argument('--plateau_factor', type=float, default=0.2)
parser.add_argument('--plateau_patience', type=int, default=20)
parser.add_argument('--lr_min', type=float, default=1e-6)
parser.add_argument('--rec_learning_rate', type=float, default=None, help='If None, use learning_rate for rec too')

# ------------------------------------------------------- Optuna ----------------------------------------------------

parser.add_argument('--hpt', type=str, choices=['grid', 'optuna'], default='grid',
                    help='Use grid search (current behavior) or Optuna.')
parser.add_argument('--n_trials', type=int, default=20,
                    help='Number of Optuna trials (ignored for grid).')


args = parser.parse_args()

method = args.method.upper()
architecture = args.architecture.upper()
NUM_EPOCHS = args.num_epochs

RESULTS_BASE = ['results_for_rotational_vs_none', f'{args.dataset}_{method}']
print(f"[*] Results will be stored in {Path(*RESULTS_BASE).absolute()}")

if args.dont_store_results:
    print("[*] NOT STORING RESULTS!!! to disk as per user request.")

base_results_path = Path(*RESULTS_BASE)
base_results_path.mkdir(parents=True, exist_ok=True)

# ---- Cosine helpers -------------------------------------------------
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

def _extract_layer_seq_grads(grads, method, layer_idx):
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

def _layer_group_vectors(grads, method, layer_idx):
    """
    Builds concatenated vectors for groups per layer:
      - 'lambda' = [nu, theta]
      - 'gamma'  = [gamma_log]
      - 'B'      = [B_re, B_im]
      - 'phi'    = [phi] (if mixing in ['rotational', 'rotational_full'])
      - 'all'    = all of the above
    Returns a dict of numpy vectors.
    """
    g = _extract_layer_seq_grads(grads, method, layer_idx)
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

def _overall_vectors(grads, method, num_layers):
    """
    Concatenate 'all' vectors over all layers into a single vector.
    """
    parts = []
    for li in range(num_layers):
        vecs = _layer_group_vectors(grads, method, li)
        if vecs['all'].size:
            parts.append(vecs['all'])
    return np.concatenate(parts) if parts else np.array([])
# --------------------------------------------------------------------


# Edge case: If no layer_output, d_model must be 2 * hidden
if not args.layer_output and architecture in ['LRU', 'ZUC']:
    args.d_model = 2 * args.num_hidden
else:
    if args.double_dmodel:
        args.d_model = 2 * args.num_hidden
    elif not args.double_dmodel and args.equal_dmodel:
        args.d_model = args.num_hidden
    else:
        args.d_model = args.num_hidden // 2
print(f"[*] d_model set to {args.d_model}")

if args.dataset in ['toy', 'epinions_ratings'] and args.task == 'link_classification':
    print("[*] Forcing task to be link_regression, because dataset does not support classification")
    args.task = 'link_regression'

hpt_space = {
    #'memory': [1, 2, 3, 4, 5],
    'memory': [args.memory],
    #'seed': [5, 11, 42, 123, 1984],
    'seed': [43],
    'learning_rate': 1e-3,
    'beta1': 0.9, 
    'beta2': 0.999,
    #'weight_decay': 0.0001,
    'weight_decay': [args.weight_decay],
    #'state_size': [32, 64, 128],
    'state_size': [args.num_hidden],
}

if args.hpt == 'grid':
    hpt_samples = GridSampler(hpt_space)
else:
    # Optuna-backed iterator that behaves like your GridSampler loop
    study = optuna.create_study(direction='minimize')
    def optuna_iter():
        for _ in range(args.n_trials):
            trial = study.ask()

            hpt = {
                # Model
                'state_size': trial.suggest_categorical('state_size', hpt_space.get('state_size', [args.num_hidden])),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                'batch_size': trial.suggest_categorical('batch_size', [args.batch_size]),
                'd_model': trial.suggesyt_categorical('d_model', [args.d_model]),

                # Optimizer
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-2, log=True),
                'learning_rate_rec': trial.suggest_float('learning_rate_rec', 1e-4, 3e-2, log=True),
                'beta1': trial.suggest_float('beta1', 0.8, 0.95),
                'beta2': trial.suggest_float('beta2', 0.95, 0.9999),
                'weight_decay': trial.suggest_float('weight_decay', 0.0, 1e-5, log=True),
                'weight_decay_on_B': trial.suggest_categorical('weight_decay_on_B', [True, False]),
                'rec_lr_on_B': trial.suggest_categorical('rec_lr_on_B', [True, False]),

                # LR Scheduling
                'early_stopping_patience': trial.suggest_int('early_stopping_patience', 10, 100),
                'warmup_end_epoch': trial.suggest_int('warmup_end_epoch', 0, 50),
                'cosine_annealing': trial.suggest_categorical('cosine_annealing', [True, False]),

                # Training Loop
                'epochs': trial.suggest_categorical('epochs', [NUM_EPOCHS]),
                'n_accumulation_steps': trial.suggest_int('n_accumulation_steps', 1, 32),
            }
            yield trial, hpt
    hpt_iter = optuna_iter()

# Choose cell type based on method.
if architecture == 'LRU':
    CELL = MultiLayerLRUCellReal
elif architecture == 'MIN':
    CELL = SymmetricMinGRUCell
elif architecture == 'ZUC':
    CELL = ZucchetCell
else:
    CELL = SymmetricGRUCell

def init_layer(layer_cls, **kwargs):
    if layer_cls == "LRU":
        layer = LRU
    return partial(layer, **kwargs)

def make_model_step(model, num_nodes, mode="training"):
    if args.dedupe:
        init_model_state = partial(memory_store, example_state=model.init_local(1), num_entries=num_nodes + 1)  # +1 to have a dummy for the dupped updates to go to
        get_state = store_get
        set_state = store_set_dedupe
    else:
        init_model_state, get_state, set_state = state_store(num_nodes, model.init_local, numpy=False)

    if method in ['ONLINE', 'TBPTT'] and mode == "training":
        if args.dedupe:
            init_lambda_traces = partial(memory_store, example_state=model.init_lambda_traces(1), num_entries=num_nodes + 1)
            init_gamma_traces = partial(memory_store, example_state=model.init_gamma_traces(1), num_entries=num_nodes + 1)
            init_B_traces = partial(memory_store, example_state=model.init_B_traces(1), num_entries=num_nodes + 1)

            get_lambda_traces = store_get
            get_gamma_traces = store_get
            get_B_traces = store_get

            set_lambda_traces = store_set_dedupe
            set_gamma_traces = store_set_dedupe
            set_B_traces = store_set_dedupe

            if args.mixing == 'rotational_full':
                init_phi_traces = partial(memory_store, example_state=model.init_phi_traces(1), num_entries=num_nodes + 1)
                get_phi_traces = store_get
                set_phi_traces = store_set_dedupe
        else:
            init_lambda_traces, get_lambda_traces, set_lambda_traces = state_store(num_nodes, model.init_lambda_traces, numpy=False)
            init_gamma_traces, get_gamma_traces, set_gamma_traces = state_store(num_nodes, model.init_gamma_traces, numpy=False)
            init_B_traces, get_B_traces, set_B_traces = state_store(num_nodes, model.init_B_traces, numpy=False)

            if args.mixing == 'rotational_full':
                init_phi_traces, get_phi_traces, set_phi_traces = state_store(num_nodes, model.init_phi_traces, numpy=False)
        
        def init_model_traces():
            lambda_traces = init_lambda_traces()
            gamma_traces = init_gamma_traces()
            B_traces = init_B_traces()
            if args.mixing == 'rotational_full':
                phi_traces = init_phi_traces()
                return (lambda_traces, gamma_traces, B_traces, phi_traces)
            else:
                return (lambda_traces, gamma_traces, B_traces)
        
        def get_traces(traces, nodes):
            batch_lambda_traces = get_lambda_traces(traces[0], nodes)
            batch_gamma_traces = get_gamma_traces(traces[1], nodes)
            batch_B_traces = get_B_traces(traces[2], nodes)
            if args.mixing == 'rotational_full':
                batch_phi_traces = get_phi_traces(traces[3], nodes)
                return (batch_lambda_traces, batch_gamma_traces, batch_B_traces, batch_phi_traces)
            else:
                return (batch_lambda_traces, batch_gamma_traces, batch_B_traces)
        
        def set_traces(traces, nodes, new_batch_traces):
            new_lambda_traces = set_lambda_traces(traces[0], nodes, new_batch_traces[0])
            new_gamma_traces = set_gamma_traces(traces[1], nodes, new_batch_traces[1])
            new_B_traces = set_B_traces(traces[2], nodes, new_batch_traces[2])
            if args.mixing == 'rotational_full':
                new_phi_traces = set_phi_traces(traces[3], nodes, new_batch_traces[3])
                return (new_lambda_traces, new_gamma_traces, new_B_traces, new_phi_traces)
            else:
                return (new_lambda_traces, new_gamma_traces, new_B_traces)

    def init_model(_=None):
        if method in ['ONLINE', 'TBPTT'] and mode == "training":
            return (init_model_state(), init_model_traces())
        else:
            return init_model_state()

    def step_model(params, states, edge, rng_model=None):
        src, dst, feature, target = edge
        nodes = jnp.array((src, dst))

        # jax.debug.print("Step model with params: {params}",
        #                 params=params)
        # jax.debug.print("States: {states}",
        #                 states=states)
        # jax.debug.print("Edge: {edge}",
        #                 edge=edge)
        # jax.debug.print("RNG Model: {rng_model}",
        #                 rng_model=rng_model)

        if method in ['ONLINE', 'TBPTT'] and mode == "training":
            states, traces = states
        
        batch_states = get_state(states, nodes)

        if args.batch_size != 0:
            batch_states = jax.tree_util.tree_map(lambda x: x.reshape((args.batch_size, 2, -1)), batch_states)

        if method in ['ONLINE', 'TBPTT'] and mode == "training":
            raw_batch_traces = get_traces(traces, nodes)
            
            if args.mixing == 'rotational_full':
                lambda_traces, gamma_traces, B_traces, phi_traces = raw_batch_traces
            else:
                lambda_traces, gamma_traces, B_traces = raw_batch_traces
                
            n_layers = len(lambda_traces)
            
            # Restructure traces by layer instead of by trace type # TODO

            if args.batch_size == 0:
                if args.mixing == 'rotational_full':
                    batch_traces = tuple(
                        (lambda_traces[i], gamma_traces[i], B_traces[i], phi_traces[i])
                        for i in range(n_layers)
                    )
                else:
                    batch_traces = tuple(
                        (lambda_traces[i], gamma_traces[i], B_traces[i])
                        for i in range(n_layers)
                    )
            else:
                if args.mixing == 'rotational_full':
                    batch_traces = tuple(
                        (lambda_traces[i].reshape((args.batch_size, 2, -1)), gamma_traces[i].reshape((args.batch_size, 2, -1)), B_traces[i].reshape((args.batch_size, 2, -1)), phi_traces[i].reshape((args.batch_size, 2, -1)))
                        for i in range(n_layers)
                    )
                else:
                    batch_traces = tuple(
                        (lambda_traces[i].reshape((args.batch_size, 2, -1)), gamma_traces[i].reshape((args.batch_size, 2, -1)), B_traces[i].reshape((args.batch_size, 2, -1)))
                        for i in range(n_layers)
                    )

            batch_states_and_traces = (batch_states, batch_traces)

        if (args.batch_size == 0) and feature.ndim == 0:
            inputs = jnp.array([feature])
        elif (args.batch_size == 0):
            inputs = jnp.array(feature)
        else:
            inputs = jnp.array(feature).reshape((args.batch_size, -1))

        if method in ['ONLINE', 'TBPTT'] and mode == "training":
            (new_batch_states, new_batch_traces), outputs = model.step(params, batch_states_and_traces, inputs, target, rng_model)

            if args.batch_size == 0:
                # Restructure traces back to original format
                new_batch_lambda_traces = tuple(new_batch_traces[i][0] for i in range(n_layers))
                new_batch_gamma_traces = tuple(new_batch_traces[i][1] for i in range(n_layers))
                new_batch_B_traces = tuple(new_batch_traces[i][2] for i in range(n_layers))
                if args.mixing == 'rotational_full':
                    new_batch_phi_traces = tuple(new_batch_traces[i][3] for i in range(n_layers))
                    new_batch_traces = (new_batch_lambda_traces, new_batch_gamma_traces, new_batch_B_traces, new_batch_phi_traces)
                else:
                    new_batch_traces = (new_batch_lambda_traces, new_batch_gamma_traces, new_batch_B_traces)
            else:
                # Restructure traces back to original format
                new_batch_lambda_traces = tuple(new_batch_traces[i][0].reshape((args.batch_size * 2, -1)) for i in range(n_layers))
                new_batch_gamma_traces = tuple(new_batch_traces[i][1].reshape((args.batch_size * 2, -1)) for i in range(n_layers))
                new_batch_B_traces = tuple(new_batch_traces[i][2].reshape((args.batch_size * 2,) + new_batch_traces[i][2].shape[2::]) for i in range(n_layers))
                if args.mixing == 'rotational_full':
                    new_batch_phi_traces = tuple(new_batch_traces[i][3].reshape((args.batch_size * 2, -1)) for i in range(n_layers))
                    new_batch_traces = (new_batch_lambda_traces, new_batch_gamma_traces, new_batch_B_traces, new_batch_phi_traces)
                else:
                    new_batch_traces = (new_batch_lambda_traces, new_batch_gamma_traces, new_batch_B_traces)
        else:
            
            new_batch_states, outputs = model.step(params, batch_states, inputs, target, rng_model)

        # Reshape back to what the state store expects
        if args.batch_size != 0:
            new_batch_states = jax.tree_util.tree_map(lambda x: x.reshape((args.batch_size * 2, -1)), new_batch_states)

            nodes = nodes.reshape((args.batch_size * 2, ))

        states = set_state(states, nodes, new_batch_states)

        if method in ['ONLINE', 'TBPTT'] and mode == "training":
            traces = set_traces(traces, nodes, new_batch_traces)

        if method in ['ONLINE', 'TBPTT'] and mode == "training":
            return (states, traces), outputs
        else:
            return states, outputs

    if method in ['ONLINE', 'TBPTT'] and mode == "training":
        return init_model, step_model, get_traces
    else:
        return init_model, step_model


def make_fbptt_unrolled(step_fun, step_data, optimizer, num_steps, rng_model=None, cell=None, get_traces=None, mode="training"):

    def episodic_step(params, states, _=None):
        data_state, model_state = states
        data_state, new_edge = step_data(data_state)
        model_state, loss = step_fun(params, model_state, new_edge, rng_model)
        return (data_state, model_state), loss

    def unrolled_step(params, state, mode="training"):
        state, losses = lax.scan(partial(episodic_step, params), state, None, num_steps)

        if mode == "no_training" and args.task in ['link_classification']:
            bces, metrics = losses
            return (jnp.mean(bces), metrics), state
        else:
            return jnp.mean(losses), state

    unrolled_step_with_grads = jax.value_and_grad(unrolled_step, has_aux=True)
    
    @jax.jit
    def unrolled_episode(state):
        params, optimizer_state, data_state, model_state, _ = state
        
        (loss, (data_state, model_state)), grads = unrolled_step_with_grads(params, (data_state, model_state))

        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        
        return (params, optimizer_state, data_state, model_state, grads), loss
    
    @jax.jit
    def no_training_unrolled_episode(state):
        params, optimizer_state, data_state, model_state, _ = state
        
        # No training in toy means we want fbptt grads for cos-sim
        if args.dataset in ['toy']:
            (loss, (data_state, model_state)), grads = unrolled_step_with_grads(params, (data_state, model_state))

            return (params, optimizer_state, data_state, model_state, grads), loss
        else:
            loss, (data_state, model_state) = unrolled_step(params, (data_state, model_state), mode="no_training")

            return (params, optimizer_state, data_state, model_state, None), loss
    
    if mode == "training":
        return unrolled_episode
    else:   
        return no_training_unrolled_episode


def make_bptt_unrolled(step_fun, step_data, optimizer, num_steps, rng_model=None, cell=None, get_traces=None):
    
    def episodic_step(states, _=None):

        accumulator, data_state, model_state, optimizer_state, params = states
        data_state, new_edge = step_data(data_state)
        model_state, (loss, grads) = step_fun(params, model_state, new_edge, rng_model)

        if method in ['ONLINE', 'TBPTT']:
            if cell is None:
                NotImplementedError("cell is not implemented")
            (_, traces) = model_state

            # Get the traces for the current batch
            nodes = jnp.array((new_edge[0], new_edge[1]))
            raw_batch_traces = get_traces(traces, nodes)

            if args.mixing == 'rotational_full':
                lambda_traces, gamma_traces, B_traces, phi_traces = raw_batch_traces
            else:
                lambda_traces, gamma_traces, B_traces = raw_batch_traces

            n_layers = len(lambda_traces)
            
            # Restructure traces by layer instead of by trace type
            if args.batch_size == 0:
                if args.mixing == 'rotational_full':
                    batch_traces = tuple(
                        (lambda_traces[i], gamma_traces[i], B_traces[i], phi_traces[i])
                        for i in range(n_layers)
                    )
                else:
                    batch_traces = tuple(
                        (lambda_traces[i], gamma_traces[i], B_traces[i])
                        for i in range(n_layers)
                    )
            else:
                if args.mixing == 'rotational_full':
                    batch_traces = tuple(
                        (lambda_traces[i].reshape((args.batch_size, 2, -1)), gamma_traces[i].reshape((args.batch_size, 2, -1)), B_traces[i].reshape((args.batch_size, 2, -1)), phi_traces[i].reshape((args.batch_size, 2, -1)))
                        for i in range(n_layers)
                    )
                else:
                    batch_traces = tuple(
                        (lambda_traces[i].reshape((args.batch_size, 2, -1)), gamma_traces[i].reshape((args.batch_size, 2, -1)), B_traces[i].reshape((args.batch_size, 2, -1)))
                        for i in range(n_layers)
                    )

                grads['cell']['params'] = jax.tree_util.tree_map(
                    lambda s: jnp.repeat(s[None, :], args.batch_size, axis=0) / args.batch_size,
                    grads['cell']['params'],
                )

            grads['cell']['params'] = cell.update_gradients(params['cell'], grads, batch_traces, grads['cell']['perturbations'])

            if args.batch_size != 0:
                grads['cell']['params'] = jax.tree_util.tree_map(lambda s: jnp.sum(s, axis=0), grads['cell']['params'])
        
        accumulator = tree_map(jnp.add, accumulator, grads)

        if not args.acc:
            updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
            params = optax.apply_updates(params, updates)

        return (accumulator, data_state, model_state, optimizer_state, params), loss

    @jax.jit
    def unrolled_episode(state):
        params, optimizer_state, data_state, model_state, _ = state
        accumulator = tree_map(jnp.zeros_like, params)

        (accumulator, data_state, model_state, optimizer_state, params), loss = lax.scan(episodic_step, (accumulator, data_state, model_state, optimizer_state, params), None, num_steps)

        grads = tree_map(lambda a: a/num_steps, accumulator)
        if args.acc:
            updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
            params = optax.apply_updates(params, updates)
        
        return (params, optimizer_state, data_state, model_state, grads), jnp.mean(loss)
    
    return unrolled_episode


for iter_num, item in enumerate(hpt_samples):

    if args.hpt == 'grid':
        hpt = item
        trial = None
    else:
        trial, hpt = item

    print('Running iteration:', iter_num)
    memory = hpt['memory']
    state_size = hpt['state_size']
    
    if args.dataset == 'toy':
        # Load data
        init_data, step_data, _, _, feature_size, output_size = get_sampler_link_regression(args.num_nodes, 
                                                        delay=memory,
                                                        feedthrough=False, batch_size=args.batch_size if args.batch_size != 0 else None, batching_strategy=args.batching_strategy)
        print(f"[*] Created toy data with {args.num_nodes} nodes")
        num_nodes = args.num_nodes
        num_steps = args.num_steps
    
    elif args.dataset == 'bitcoin_otc':
        # Example (OTC):
        preprocess_bitcoin_csv("/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/csvs/soc-sign-bitcoinotc.csv", "/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/bitcoin_otc_stream.npz", use_sign_label=False, val_ratio=0.15, test_ratio=0.15)

        init_data, step_data, num_nodes, num_steps, feature_size, output_size = get_stream_sampler_from_npz("/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/bitcoin_otc_stream.npz", split="train", shuffle_each_epoch=False, batch_size=args.batch_size if args.batch_size != 0 else None, batching_strategy=args.batching_strategy)

        init_val_data, val_step_data, _, num_val_steps, _, _ = get_stream_sampler_from_npz("/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/bitcoin_otc_stream.npz", split="val", shuffle_each_epoch=False, batch_size=args.batch_size if args.batch_size != 0 else None, batching_strategy=args.batching_strategy)

        init_test_data, test_step_data, _, num_test_steps, _, _ = get_stream_sampler_from_npz("/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/bitcoin_otc_stream.npz", split="test", shuffle_each_epoch=False, batch_size=args.batch_size if args.batch_size != 0 else None, batching_strategy=args.batching_strategy)

        print(f"[*] Loaded data from /Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/bitcoin_otc_stream.npz with {num_nodes} nodes")

    elif args.dataset == 'bitcoin_alpha':
        # Example (Alpha):
        preprocess_bitcoin_csv("/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/csvs/soc-sign-bitcoinalpha.csv", "/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/bitcoin_alpha_stream.npz", use_sign_label=False, val_ratio=0.15, test_ratio=0.15)

        init_data, step_data, num_nodes, num_steps, feature_size, output_size = get_stream_sampler_from_npz("/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/bitcoin_alpha_stream.npz", split="train", shuffle_each_epoch=False, batch_size=args.batch_size if args.batch_size != 0 else None, batching_strategy=args.batching_strategy)

        init_val_data, val_step_data, _, num_val_steps, _, _ = get_stream_sampler_from_npz("/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/bitcoin_alpha_stream.npz", split="val", shuffle_each_epoch=False, batch_size=args.batch_size if args.batch_size != 0 else None, batching_strategy=args.batching_strategy)

        init_test_data, test_step_data, _, num_test_steps, _, _ = get_stream_sampler_from_npz("/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/bitcoin_alpha_stream.npz", split="test", shuffle_each_epoch=False, batch_size=args.batch_size if args.batch_size != 0 else None, batching_strategy=args.batching_strategy)

        print(f"[*] Loaded data from /Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/bitcoin_alpha_stream.npz with {num_nodes} nodes")

    elif args.dataset == 'wiki_rfa':
        preprocess_wikirfa_gz("/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/gzs/wiki-RfA.txt.gz", "/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/wiki_rfa.npz", val_ratio=0.15, test_ratio=0.15, drop_neutral=True)

        init_data, step_data, num_nodes, num_steps, feature_size, output_size = get_stream_sampler_from_npz("/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/wiki_rfa.npz", split="train", shuffle_each_epoch=False, batch_size=args.batch_size if args.batch_size != 0 else None, batching_strategy=args.batching_strategy)

        init_val_data, val_step_data, _, num_val_steps, _, _ = get_stream_sampler_from_npz("/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/wiki_rfa.npz", split="val", shuffle_each_epoch=False, batch_size=args.batch_size if args.batch_size != 0 else None, batching_strategy=args.batching_strategy)

        init_test_data, test_step_data, _, num_test_steps, _, _ = get_stream_sampler_from_npz("/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/wiki_rfa.npz", split="test", shuffle_each_epoch=False, batch_size=args.batch_size if args.batch_size != 0 else None, batching_strategy=args.batching_strategy)

        print(f"[*] Loaded data from /Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/wiki_rfa.npz with {num_nodes} nodes")

    elif args.dataset == 'epinions_ratings':
        preprocess_epinions_ratings_zip("/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/zips/epinions_with_rating_timestamp_txt.zip", "/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/epinions_ratings.npz", val_ratio=0.15, test_ratio=0.15, as_regression=True, min_events_item=1, min_events_user=1)

        init_data, step_data, num_nodes, num_steps, feature_size, output_size = get_stream_sampler_from_npz("/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/epinions_ratings.npz", split="train", shuffle_each_epoch=False, batch_size=args.batch_size if args.batch_size != 0 else None, batching_strategy=args.batching_strategy)

        init_val_data, val_step_data, _, num_val_steps, _, _ = get_stream_sampler_from_npz("/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/epinions_ratings.npz", split="val", shuffle_each_epoch=False, batch_size=args.batch_size if args.batch_size != 0 else None, batching_strategy=args.batching_strategy)

        init_test_data, test_step_data, _, num_test_steps, _, _ = get_stream_sampler_from_npz("/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/epinions_ratings.npz", split="test", shuffle_each_epoch=False, batch_size=args.batch_size if args.batch_size != 0 else None, batching_strategy=args.batching_strategy)

        print(f"[*] Loaded data from /Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/epinions_ratings.npz with {num_nodes} nodes")

    elif args.dataset in ['reddit', 'mooc', 'wikipedia']:
        preprocess_temporal_csv(
            f"/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/csvs/{args.dataset}.csv", 
            f"/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/{args.dataset}_stream.npz", 
            args.dataset,
            val_ratio=0.15, 
            test_ratio=0.15,
            drop_hod_dow=True
        )
        
        init_data, step_data, num_nodes, num_steps, feature_size, output_size = get_stream_sampler_from_npz(f"/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/{args.dataset}_stream.npz", split="train", shuffle_each_epoch=False, batch_size=args.batch_size if args.batch_size != 0 else None, batching_strategy=args.batching_strategy)

        init_val_data, val_step_data, _, num_val_steps, _, _ = get_stream_sampler_from_npz(f"/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/{args.dataset}_stream.npz", split="val", shuffle_each_epoch=False, batch_size=args.batch_size if args.batch_size != 0 else None, batching_strategy=args.batching_strategy)

        init_test_data, test_step_data, _, num_test_steps, _, _ = get_stream_sampler_from_npz(f"/Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/{args.dataset}_stream.npz", split="test", shuffle_each_epoch=False, batch_size=args.batch_size if args.batch_size != 0 else None, batching_strategy=args.batching_strategy)

        print(f"[*] Loaded data from /Users/franciscosilva/Documents/Projects/no-time-to-backprop/tgap/data/npzs/{args.dataset}_stream.npz with {num_nodes} nodes")

    # define model and optimizer
    if architecture == 'ZUC':
        aditional_arguments = {}
        aditional_arguments["r_min"] = 0
        aditional_arguments["r_max"] = 1.0

        if method == 'ONLINE':
            training_mode = 'online_full'
        elif method == 'FBPTT':
            training_mode = 'bptt'
        elif method == 'TBPTT':
            training_mode = 'online_1truncated'
        elif method == 'SPATIAL':
            training_mode = 'online_spatial'
        else:
            raise ValueError(f"Unknown method {method}")

        rec = init_layer(
            layer_cls="LRU",
            d_hidden=state_size,
            d_model=args.d_model,
            seq_length=1,
            training_mode=training_mode,
            has_layer_output=args.layer_output,
            mixing=args.mixing,
            has_non_linearity_in_recurrence=args.has_non_linearity_in_recurrence,
            **aditional_arguments
        )
        input_rec = init_layer(
            layer_cls="LRU",
            d_hidden=state_size,
            d_model=args.d_model,
            seq_length=1,
            training_mode=training_mode,
            has_layer_output=args.layer_output,
            mixing=args.mixing,
            d_in=feature_size if not args.encoder else None,
            has_non_linearity_in_recurrence=args.has_non_linearity_in_recurrence,
            **aditional_arguments
        )
        output_rec = init_layer(
            layer_cls="LRU",
            d_hidden=state_size,
            d_model=args.d_model,
            seq_length=1,
            training_mode=training_mode,
            has_layer_output=True,
            mixing=args.mixing,
            d_out=output_size if args.decoder == "NONE" else None, # if we have no decoder, then the last layer must project to output size
            has_non_linearity_in_recurrence=args.has_non_linearity_in_recurrence,
            **aditional_arguments
        )
        single_unique_rec = init_layer(  # special case, where a single layer is both input and output
            layer_cls="LRU",
            d_hidden=state_size,
            d_model=args.d_model,
            seq_length=1,
            training_mode=training_mode,
            has_layer_output=True,
            mixing=args.mixing,
            d_in=feature_size if not args.encoder else None,
            d_out=output_size if args.decoder == "NONE" else None, # if we have no decoder, then the last layer must project to output size
            has_non_linearity_in_recurrence=args.has_non_linearity_in_recurrence,
            **aditional_arguments
        )
        gcell = CELL(
            rec=rec,
            input_rec=input_rec,
            output_rec=output_rec if args.decoder == "NONE" else None, # if we have a decoder, then pass None here to skip output_rec creation
            single_unique_rec=single_unique_rec if not args.encoder and args.decoder == "NONE" and args.num_layers == 1 else None, # if we have both no encoder and no decoder and only 1 layer, we use this special layer, else pass None here to skip single_unique_rec creation
            rec_type='LRU',
            d_input=feature_size,
            d_output=output_size,
            d_model=args.d_model,
            n_layers=args.num_layers,
            seq_length=1,
            padded=False,
            activation=args.activation,
            readout=0,
            dropout=args.dropout,
            mode='none',
            prenorm=args.prenorm,
            postnorm=args.postnorm,
            multidim=1,
            training=True,
            training_mode=training_mode,
            d_hidden=state_size,
            in_dim=feature_size,
            bsz=args.batch_size if args.batch_size != 0 else None,
            has_encoder=args.encoder,
            decoder_type=args.decoder,
            has_extra_skip=args.extra_skip,
            mixing=args.mixing,
            has_non_linearity_in_recurrence=args.has_non_linearity_in_recurrence,
        )

        # No training version: bptt mode, training False, bsz None ----------------------------------------------------

        no_training_rec = init_layer(
            layer_cls="LRU",
            d_hidden=state_size,
            d_model=args.d_model,
            seq_length=1,
            training_mode='bptt',
            has_layer_output=args.layer_output,
            mixing=args.mixing,
            has_non_linearity_in_recurrence=args.has_non_linearity_in_recurrence,
            **aditional_arguments
        )
        no_training_input_rec = init_layer(
            layer_cls="LRU",
            d_hidden=state_size,
            d_model=args.d_model,
            seq_length=1,
            training_mode='bptt',
            has_layer_output=args.layer_output,
            mixing=args.mixing,
            d_in=feature_size if not args.encoder else None,
            has_non_linearity_in_recurrence=args.has_non_linearity_in_recurrence,
            **aditional_arguments
        )
        no_training_output_rec = init_layer(
            layer_cls="LRU",
            d_hidden=state_size,
            d_model=args.d_model,
            seq_length=1,
            training_mode='bptt',
            has_layer_output=True,
            mixing=args.mixing,
            d_out=output_size if args.decoder == "NONE" else None, # if we have no decoder, then the last layer must project to output size
            has_non_linearity_in_recurrence=args.has_non_linearity_in_recurrence,
            **aditional_arguments
        )
        no_training_single_unique_rec = init_layer(  # special case, where a single layer is both input and output
            layer_cls="LRU",
            d_hidden=state_size,
            d_model=args.d_model,
            seq_length=1,
            training_mode='bptt',
            has_layer_output=True,
            mixing=args.mixing,
            d_in=feature_size if not args.encoder else None,
            d_out=output_size if args.decoder == "NONE" else None, # if we have no decoder, then the last layer must project to output size
            has_non_linearity_in_recurrence=args.has_non_linearity_in_recurrence,
            **aditional_arguments
        )
        no_training_gcell = CELL(
            rec=no_training_rec,
            input_rec=no_training_input_rec,
            output_rec=no_training_output_rec if args.decoder == "NONE" else None, # if we have a decoder, then pass None here to skip output_rec creation
            single_unique_rec=no_training_single_unique_rec if not args.encoder and args.decoder == "NONE" and args.num_layers == 1 else None, # if we have both no encoder and no decoder and only 1 layer, we use this special layer, else pass None here to skip single_unique_rec creation
            rec_type='LRU',
            d_input=feature_size,
            d_output=output_size,
            d_model=args.d_model,
            n_layers=args.num_layers,
            seq_length=1,
            padded=False,
            activation=args.activation,
            readout=0,
            dropout=0.0,
            mode='none',
            prenorm=args.prenorm,
            postnorm=args.postnorm,
            multidim=1,
            training=False,
            training_mode='bptt',
            d_hidden=state_size,
            in_dim=feature_size,
            bsz=args.batch_size if args.batch_size != 0 else None,
            has_encoder=args.encoder,
            decoder_type=args.decoder,
            has_extra_skip=args.extra_skip,
            mixing=args.mixing,
            has_non_linearity_in_recurrence=args.has_non_linearity_in_recurrence,
        )
        
    else:
        gcell = CELL(state_size, feature_size)
        no_training_gcell = CELL(state_size, feature_size)
    regressor = MLP([state_size, 1], 2*state_size, scalar_output=True)

    trained_with_mse = False
    if args.task == 'link_classification':
        loss_function = bce() 
        #trained_with_mse = True
        #loss_function = mse(scale_pos_weight=float((1.0 - 0.9) / 0.9)) # assuming ~90% positive class
        fake_loss_function = metrics_fake_loss(loss_function)
        fake_loss = with_loss_without_mlp(fake_loss_function)
    else:
        loss_function = mse()
    
    if architecture in ['LRU', 'ZUC']:
        loss = with_loss_without_mlp(loss_function)
    else:
        loss = with_loss(regressor, loss_function)

    optimizer = optax.chain(
        #optax.clip(1.0),
        optax.adamw(hpt['learning_rate'], b1=hpt['beta1'], b2=hpt['beta2'], weight_decay=hpt['weight_decay'])
    )

    optimizer = create_optimizer(args, hpt)

    # make episode step
    if method in ['FBPTT', 'SPATIAL']:
        model = with_feedforward_loss(gcell, loss)
        make_unrolled = make_fbptt_unrolled
    else:
        model = with_feedforward_and_truncated_grads(gcell, loss)
        make_unrolled = make_bptt_unrolled

    if method not in ['FBPTT', 'SPATIAL']:
        init_model, step_model, get_traces = make_model_step(model, num_nodes)
    else:
        init_model, step_model = make_model_step(model, num_nodes)

    if args.dataset not in ['toy'] and args.task == 'link_classification':
        no_training_model = with_feedforward_loss(no_training_gcell, fake_loss)
    else:
        no_training_model = with_feedforward_loss(no_training_gcell, loss)

    _, no_training_step_model = make_model_step(no_training_model, num_nodes, mode="no_training")
    
    rng_model = jax.random.PRNGKey(hpt['seed'])

    make_no_training_unrolled = partial(make_fbptt_unrolled, mode='no_training')

    unrolled_episode = make_unrolled(step_model, step_data, optimizer, num_steps, rng_model, cell=gcell, get_traces=get_traces if method in ['ONLINE', 'TBPTT'] else None)

    if args.dataset in ['toy']:
        unrolled_episode_for_fbptt_grads = make_no_training_unrolled(no_training_step_model, step_data, optimizer, num_steps, rng_model, cell=gcell, get_traces=get_traces if method in ['ONLINE', 'TBPTT'] else None)

    if args.dataset not in ['toy']:
        unrolled_val_episode = make_no_training_unrolled(no_training_step_model, val_step_data, optimizer, num_val_steps, rng_model, cell=gcell, get_traces=get_traces if method in ['ONLINE', 'TBPTT'] else None)

        unrolled_test_episode = make_no_training_unrolled(no_training_step_model, test_step_data, optimizer, num_test_steps, rng_model, cell=gcell, get_traces=get_traces if method in ['ONLINE', 'TBPTT'] else None)

    # initialize model/data/optimizer state/params
    seed = hpt['seed']
    rng_key = jax.random.PRNGKey(seed)
    rng_data, rng_params = jax.random.split(rng_key, 2)
    data_state = init_data(rng_data)
    params = model.init_params(rng_params)
    print(f"[*] Trainable Parameters: {count_params(params)}")
    # Verify device
    first_param = jax.tree_util.tree_leaves(params)[0]
    print(f"[*] Model running on device: {first_param.device}")
    model_state = init_model()
    optimizer_state = optimizer.init(params)

    print("All params tree leaf keys:")
    print_tree_keys(params)
    print("----")
    # training loop
    losses = []
    val_losses = []
    val_metrics_history = []
    test_losses = []
    test_metrics_history = []
    gammas = []
    lambdas = []
    all_grads = []
    cosine_history = []
    best_val_roc_auc = 0.0
    best_test_roc_auc = 0.0
    best_val_epoch = -1
    best_test_epoch = -1
    for epoch in range(NUM_EPOCHS):

        print_condition = epoch % 500 == 0 or (method in ['ONLINE', 'SPATIAL'] and num_steps*(args.batch_size + 1) > 1000 and epoch % 50 == 0)  or (args.dataset not in ['toy'] and epoch % 1 == 0) or (args.dataset in ['toy'] and args.num_epochs < 50 and epoch % 1 == 0)
        if print_condition:
            print(f"[*] Starting Training Epoch {epoch + 1}...")

        if method not in ['FBPTT', 'SPATIAL']:

            params['cell']["perturbations"] = jax.tree_util.tree_map(
                lambda s: jnp.zeros_like(s), params['cell']["perturbations"]
            )

        state = (params, optimizer_state, data_state, model_state, None) # None for grads, will be computed in unrolled_episode

        if args.dataset in ['toy']:
            if method in ['ONLINE', 'TBPTT']:
                model_state_for_fbptt, _ = model_state  # discard traces
            else:
                model_state_for_fbptt = model_state

            state_for_fbptt_grads = (params, optimizer_state, data_state, model_state_for_fbptt, None) # None for grads, will be computed in unrolled_episode_for_fbptt_grads
            (_, _, _, _, grads_for_cossim), _ = unrolled_episode_for_fbptt_grads(state_for_fbptt_grads)
        
        (params, optimizer_state, data_state, model_state_for_val, grads_for_debug), l = unrolled_episode(state)
        
        # only at last epoch, print the grads and model state for debugging
        if epoch == NUM_EPOCHS - 1 and args.dont_store_results and (not (args.dataset in ['toy'] and args.num_epochs < 50) or (args.num_epochs == 1)):
            print(f"[*] Epoch {epoch + 1} completed")
            print(f"[*] Loss: {l}")
            print(f"[*] grads_for_debug: {grads_for_debug}")
            print(f"[*] grads_for_cossim: {grads_for_cossim}")

        if print_condition:
            print(f"[*] Training Loss: {l}")

        losses.append(float(l))

        # Val and test evaluation - the split is temporal, so we dont re init things, we keep going but with a model that doesnt compute grads
        if args.dataset not in ['toy']:

            if len(model_state_for_val) == 2:
                model_state_for_val, _ = model_state_for_val  # discard traces

            val_data_state = init_val_data(data_state[-1])
            no_training_state = (params, optimizer_state, val_data_state, model_state_for_val, None)
            (params, optimizer_state, val_data_state, model_state_for_test, _), val_loss = unrolled_val_episode(no_training_state)

            if args.task == 'link_classification':
                val_bce, (val_logits, val_labels) = val_loss
                val_metrics = compute_metrics_from_logits(val_logits, val_labels, print_stuff=print_condition, trained_with_mse=trained_with_mse)

                if print_condition:
                    print(f"[*] Val Loss: {val_bce}")
                    for metric_name, metric_value in val_metrics.items():
                        print(f"[*] Val {metric_name}: {metric_value}")
                    print()

                val_losses.append(float(val_bce))
                val_metrics_history.append(val_metrics)

                if val_metrics['roc_auc'] > best_val_roc_auc:
                    best_val_roc_auc = val_metrics['roc_auc']
                    best_val_epoch = epoch
            else:
                if print_condition:
                    print(f"[*] Val Loss: {val_loss}")
                val_losses.append(float(val_loss))

            test_data_state = init_test_data(val_data_state[-1])
            no_training_state = (params, optimizer_state, test_data_state, model_state_for_test, None)
            (params, optimizer_state, test_data_state, _, _), test_loss = unrolled_test_episode(no_training_state)
            
            if args.task == 'link_classification':

                test_bce, (test_logits, test_labels) = test_loss
                test_metrics = compute_metrics_from_logits(test_logits, test_labels, print_stuff=print_condition, trained_with_mse=trained_with_mse)

                if print_condition:
                    print(f"[*] Test Loss: {test_bce}")
                    for metric_name, metric_value in test_metrics.items():
                        print(f"[*] Test {metric_name}: {metric_value}")
                    print()

                test_losses.append(float(test_bce))
                test_metrics_history.append(test_metrics)

                if test_metrics['roc_auc'] > best_test_roc_auc:
                    best_test_roc_auc = test_metrics['roc_auc']
                    best_test_epoch = epoch
            else:
                if print_condition:
                    print(f"[*] Test Loss: {test_loss}")
                test_losses.append(float(test_loss))

        if not args.dont_store_results:
            # 3) Cosine similarity bookkeeping (toy only)
            if args.dataset in ['toy'] and method not in ['FBPTT']:
                per_epoch = {'overall': None, 'layers': {}}

                # overall (all layers, all groups)
                v_md_all = _overall_vectors(grads_for_debug,   method,       args.num_layers)
                v_fb_all = _overall_vectors(grads_for_cossim, 'FBPTT',       args.num_layers)
                per_epoch['overall'] = _cos(v_md_all, v_fb_all)

                # per-layer, by group (lambda, gamma, B) and 'all'
                for li in range(args.num_layers):
                    md_vecs = _layer_group_vectors(grads_for_debug,   method,  li)
                    fb_vecs = _layer_group_vectors(grads_for_cossim, 'FBPTT',  li)
                    per_epoch['layers'][f'layer_{li}'] = {
                        'lambda': _cos(md_vecs['lambda'], fb_vecs['lambda']) if md_vecs['lambda'].size and fb_vecs['lambda'].size else float('nan'),
                        'gamma' : _cos(md_vecs['gamma'],  fb_vecs['gamma'])  if md_vecs['gamma'].size  and fb_vecs['gamma'].size  else float('nan'),
                        'B'     : _cos(md_vecs['B'],      fb_vecs['B'])      if md_vecs['B'].size      and fb_vecs['B'].size      else float('nan'),
                        'all'   : _cos(md_vecs['all'],    fb_vecs['all'])    if md_vecs['all'].size    and fb_vecs['all'].size    else float('nan'),
                    }
                    if args.mixing in ['rotational', 'rotational_full']:
                        per_epoch['layers'][f'layer_{li}']['phi'] = _cos(md_vecs['phi'], fb_vecs['phi']) if md_vecs['phi'].size and fb_vecs['phi'].size else float('nan')

                cosine_history.append(per_epoch)

                if(epoch % 500 == 0 or (method in ['ONLINE', 'SPATIAL'] and num_steps*(args.batch_size + 1) > 1000 and epoch % 50 == 0)):
                    print(f"[*] Cosine Similarity (overall): {per_epoch['overall']}")
                    for li in range(args.num_layers):
                        layer_cos = per_epoch['layers'][f'layer_{li}']
                        if args.mixing in ['rotational', 'rotational_full']:
                            print(f"    Layer {li}: lambda: {layer_cos['lambda']}, gamma: {layer_cos['gamma']}, B: {layer_cos['B']}, phi: {layer_cos['phi']}, all: {layer_cos['all']}")
                        else:
                            print(f"    Layer {li}: lambda: {layer_cos['lambda']}, gamma: {layer_cos['gamma']}, B: {layer_cos['B']}, all: {layer_cos['all']}")
                    print()

        data_state = init_data(data_state[-1])
        # No need for init_model() because model_state is never updated here, and thus is always all 0s

        if print_condition and args.dataset not in ['toy']:
            print(f"[*] Best Val ROC AUC: {best_val_roc_auc} at epoch {best_val_epoch + 1}")
            print(f"[*] Best Test ROC AUC: {best_test_roc_auc} at epoch {best_test_epoch + 1}")
            print("-----------------------------------------------------")

    losses = np.array(losses)

    if args.dataset not in ['toy']:
        val_losses = np.array(val_losses)
        test_losses = np.array(test_losses)

    if trial is not None:
        study.tell(trial, np.min(val_losses))

    general_config_sub_folder = f'memory_{memory}_hidden_{state_size}_layers_{args.num_layers}_act_{args.activation}'
    flag_configuration_id = f'prenorm_{args.prenorm}_postnorm_{args.postnorm}_encoder_{args.encoder}_layerout_{args.layer_output}_extraskip_{args.extra_skip}_decoder_{args.decoder}_mixing_{args.mixing}_nonlinrec_{args.has_non_linearity_in_recurrence}_batchsize_{args.batch_size}_seed_{seed}'

    if args.dont_store_results:
        print("[*] Results storage disabled, skipping saving results to disk.")
        continue
    
    # NEW: save cosine history (toy only)
    if args.dataset in ['toy'] and len(cosine_history):
        cos_path = base_results_path / general_config_sub_folder / 'cosine_similarity'
        cos_path.mkdir(parents=True, exist_ok=True)
        cos_file = cos_path / f'{flag_configuration_id}.npy'
        with open(cos_file, 'wb') as f:
            np.save(f, cosine_history, allow_pickle=True)

    # Prepare data for CSV - each row will have configuration parameters and the best loss
    
    csv_path = base_results_path / 'global_results.csv'
    file_exists = os.path.isfile(csv_path)
    
    # Data to save
    data = {
        'memory': memory,
        'state_size': state_size,
        'num_layers': args.num_layers,
        'architecture': architecture,
        'method': method,
        'activation': args.activation,
        'prenorm': args.prenorm,
        'postnorm': args.postnorm,
        'encoder': args.encoder,
        'layer_output': args.layer_output,
        'extra_skip': args.extra_skip,
        'decoder': args.decoder,
        'mixing': args.mixing,
        'non_linearity_in_recurrence': args.has_non_linearity_in_recurrence,
        'seed': seed,
        'learning_rate': hpt['learning_rate'],
        'beta1': hpt['beta1'],
        'beta2': hpt['beta2'],
        'weight_decay': hpt['weight_decay'],
        'num_epochs': NUM_EPOCHS,
        'num_steps': num_steps,
        'best_loss': float(np.min(losses)),
        'average_final_100_loss': float(np.mean(losses[-100:])),
        'final_loss': float(losses[-1])
    }

    if args.dataset not in ['toy']:
        # add validation and test losses to the data dict
        data['best_val_loss'] = float(np.min(val_losses))
        data['average_final_100_val_loss'] = float(np.mean(val_losses[-100:]))
        data['final_val_loss'] = float(val_losses[-1])
        data['best_test_loss'] = float(np.min(test_losses))
        data['average_final_100_test_loss'] = float(np.mean(test_losses[-100:]))
        data['final_test_loss'] = float(test_losses[-1])

    # Write to CSV
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = list(data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data)
    
    # Still save the full loss trajectory as numpy for detailed analysis
    results_path = base_results_path / general_config_sub_folder / f'loss_trajectory' / f'{flag_configuration_id}.npy'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'wb') as file:
        np.save(file, losses)

    if args.dataset not in ['toy']:
        val_results_path = base_results_path / general_config_sub_folder / f'val_loss_trajectory' / f'{flag_configuration_id}.npy'
        val_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(val_results_path, 'wb') as file:
            np.save(file, val_losses)

        test_results_path = base_results_path / general_config_sub_folder / f'test_loss_trajectory' / f'{flag_configuration_id}.npy'
        test_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_results_path, 'wb') as file:
            np.save(file, test_losses)

