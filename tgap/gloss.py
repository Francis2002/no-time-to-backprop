import jax
import jax.numpy as jnp
from jax import nn
from dataclasses import dataclass
from typing import Callable
import numpy as np

from tgap.grnn import BaseGRNNCell, GRNNCell

# ---------- Metric helpers (works on JAX arrays or NumPy) ----------
def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

def f1_from_counts(tp, fp, fn):
    denom = (2*tp + fp + fn) + 1e-12
    return (2*tp) / denom

def f1_binary_macro_micro(preds, labels):
    # preds, labels ∈ {0,1}, shape (N,)
    preds = preds.astype(jnp.int32)
    labels = labels.astype(jnp.int32)
    # Class 1 counts
    tp1 = jnp.sum((preds == 1) & (labels == 1))
    fp1 = jnp.sum((preds == 1) & (labels == 0))
    fn1 = jnp.sum((preds == 0) & (labels == 1))
    f1_pos = f1_from_counts(tp1, fp1, fn1)
    # Class 0 counts (treat "0" as the positive in a one-vs-rest view)
    tp0 = jnp.sum((preds == 0) & (labels == 0))
    fp0 = jnp.sum((preds == 0) & (labels == 1))
    fn0 = jnp.sum((preds == 1) & (labels == 0))
    f1_neg = f1_from_counts(tp0, fp0, fn0)
    # Macro = average over classes
    f1_macro = 0.5 * (f1_pos + f1_neg)
    # Micro = global counts
    tp = tp1 + tp0
    fp = fp1 + fp0
    fn = fn1 + fn0
    f1_micro = f1_from_counts(tp, fp, fn)
    # Binary (positive class F1)
    f1_binary = f1_pos
    return f1_binary, f1_macro, f1_micro

def roc_auc(scores, labels):
    # Binary ROC-AUC from scores (logits or probs) and {0,1} labels
    # Use logits directly; monotone transform is fine for ROC
    scores = jnp.asarray(scores).reshape(-1)
    labels = jnp.asarray(labels).reshape(-1).astype(jnp.int32)
    N = scores.shape[0]
    # Sort by descending score
    order = jnp.argsort(scores)[::-1]
    s = scores[order]
    y = labels[order]

    # Cumulate positives/negatives
    P = jnp.sum(y == 1)
    Nn = jnp.sum(y == 0)
    # Edge cases
    P = jnp.maximum(P, 1)
    Nn = jnp.maximum(Nn, 1)

    tps = jnp.cumsum((y == 1).astype(jnp.float32))
    fps = jnp.cumsum((y == 0).astype(jnp.float32))
    tpr = tps / P
    fpr = fps / Nn

    # Trapezoidal area under the curve
    # (prepend 0, append 1 to complete the curve)
    tpr = jnp.concatenate([jnp.array([0.0]), tpr, jnp.array([1.0])])
    fpr = jnp.concatenate([jnp.array([0.0]), fpr, jnp.array([1.0])])
    auc = jnp.trapz(tpr, fpr)
    return auc



@dataclass
class FFModel:
    init_params: Callable
    apply: Callable
    has_aux: bool = False

    def init_params(rng: jax.random.KeyArray):
        """Initializes parameters of model.

        Args:
            rng (KeyArray): PRNG key for initializer.

        Returns:
            pytree: Parameters of model.
        """
        raise NotImplementedError
    
    def apply(params, inputs):
        """Applies model.
        """
        raise NotImplementedError


def Dropout(rate=0.1):
    def apply(input, rng):
        if rng is None:
            return input
        u = jax.random.uniform(rng, input.shape)
        return jnp.where(u < rate, 0., input) / (1 - rate)

    return apply if rate > 0 else lambda input, rng: input


def MLP(
        hidden_sizes: list[int], 
        input_size: int, 
        activation: Callable=nn.relu, 
        last_activation: bool=False, 
        scalar_output: bool=False, 
        initializer=nn.initializers.lecun_normal,
        dropout=0
        ) -> FFModel:
    num_layers = len(hidden_sizes)
    apply_dropout = Dropout(dropout)

    def apply(params, inputs, rng=None):
        inputs = apply_dropout(inputs, rng)
        for i, param in enumerate(params):
            inputs = jnp.dot(inputs, param['w']) + param['b']
            if last_activation or i < num_layers - 1:
                inputs = activation(inputs)
            if i < num_layers - 1:
                inputs = apply_dropout(inputs, rng)
        return inputs
    
    def init(rng): 
        initializer_instance = initializer()
        rngs = jax.random.split(rng, num_layers)
        params = [
            {
                'w': initializer_instance(rng, (prev_size, size)),
                'b': jnp.zeros(size)
            }
            for rng, prev_size, size in zip(rngs, [input_size] + hidden_sizes, hidden_sizes)
        ]
        if scalar_output:
            assert hidden_sizes[-1] == 1
            params[-1]['w'] = params[-1]['w'][..., 0]
            params[-1]['b'] = params[-1]['b'][..., 0]
        return tuple(params)
    
    return FFModel(init, apply)


def bce(scale_pos_weight = 1, reduction=jnp.mean):
    def loss(logits, targets):
        abs_logits = jnp.abs(logits)
        losses = jnp.log1p(jnp.exp(-abs_logits))

        # target dependant part
        margin = logits * (1 - 2 * targets)
        losses += jnp.maximum(0, margin)
        if scale_pos_weight != 1:
            losses *= jnp.where(targets, scale_pos_weight, 1)
        return reduction(losses)

    return loss

def mse(scale_pos_weight = 1, reduction=jnp.mean):
    def loss(logits, targets):
        losses = (logits - targets) ** 2
        if scale_pos_weight != 1:
            losses *= jnp.where(targets >= 0, scale_pos_weight, 1)
        return reduction(losses)
    return loss

def _flatten_binary(logits, labels):
    logits = jnp.asarray(logits)
    labels = jnp.asarray(labels)
    # Flatten *all* leading dims into one vector
    logits = logits.reshape(-1)
    labels = labels.reshape(-1).astype(jnp.int32)
    return logits, labels

def metrics_fake_loss(loss_fn):

    def metrics_fake_loss_function(logits, targets):
        # 1) ordinary batch loss using the existing loss function
        loss_batch = loss_fn(logits, targets)  # this should already reduce over the batch

        # 2) return the raw batch logits and targets so we can concatenate later
        # (No need to "copy" in JAX; returning the arrays is fine.)
        return loss_batch, (jnp.asarray(logits), jnp.asarray(targets))
    
    return metrics_fake_loss_function

def compute_metrics_from_logits(all_logits, all_targets, print_stuff=False, trained_with_mse=False):
    all_logits, all_targets = _flatten_binary(all_logits, all_targets)

    all_labels = all_targets.astype(jnp.int32)

    from sklearn.metrics import roc_curve, roc_auc_score

    def recall_at_fpr(y_true, y_pred, target_fpr=0.01):
        fprs, tprs, _ = roc_curve(y_true, y_pred)
        return np.interp(target_fpr, fprs, tprs)

    if print_stuff:
        labels = (all_targets > 0).astype(jnp.int32)  # see label note below
        pos = all_logits[labels == 1]
        neg = all_logits[labels == 0]

        print("prevalence =", float(jnp.mean(labels)))
        if not trained_with_mse:
            print("mean sig(logit) pos =", float(jnp.mean(nn.sigmoid(pos))))
            print("mean sig(logit) neg =", float(jnp.mean(nn.sigmoid(neg))))
            
        print("mean logit pos =", float(jnp.mean(pos)))
        print("mean logit neg =", float(jnp.mean(neg)))

    if trained_with_mse:
        # If trained with MSE, logits are trying to reproduce the targets in [-10..+10].
        # We can still do ROC-AUC and recall@FPR, but we need to threshold to get binary preds.
        all_preds = (all_logits < 0.0).astype(jnp.int32) # Logits are floats from -10 to 10, collapse to binary

        print("Frequency of positive predictions:", float(jnp.mean(all_preds)))

        auc = roc_auc_score(all_labels, all_logits)
        recall_at_0_01 = recall_at_fpr(all_labels, all_logits, target_fpr=0.01)
        recall_at_0_05 = recall_at_fpr(all_labels, all_logits, target_fpr=0.05)
        accuracy = jnp.mean(all_preds == all_labels)

    else:
        auc = roc_auc_score(all_labels, all_logits)
        # recall_at_0_01 = recall_at_fpr(all_labels, all_logits, target_fpr=0.01)
        # recall_at_0_05 = recall_at_fpr(all_labels, all_logits, target_fpr=0.05)

    return {
        'roc_auc': auc,
        # 'recall_at_0.01fpr': recall_at_0_01,
        # 'recall_at_0.05fpr': recall_at_0_05,
        # 'accuracy': accuracy if trained_with_mse else float('nan'),
    }

def with_loss(mlp: FFModel, loss: Callable, mlp_output: bool=False) -> FFModel:
    def apply_with_loss(params, inputs, targets, rng=None):
        output = mlp.apply(params, inputs, rng=rng)
        l = loss(output, targets)
        return (l, output) if mlp_output else l
    
    return FFModel(mlp.init_params, apply_with_loss, mlp_output)

def with_loss_without_mlp(loss: Callable) -> FFModel:
    def apply_with_loss(params, inputs, targets, rng=None):
        return loss(inputs, targets)
    
    return FFModel(lambda rng: None, apply_with_loss)

def with_feedforward_loss(cell: BaseGRNNCell, loss: FFModel) -> GRNNCell:
    def init_params(rng):
        cell_rng, loss_rng = jax.random.split(rng, 2)
        return {
            'cell': cell.init_params(cell_rng),
            'loss': loss.init_params(loss_rng)
        }
    
    def step_with_loss(params, states, inputs, targets, rng=None):
        new_states, outputs = cell.step(params['cell'], states, inputs, rng=rng)
        return new_states, loss.apply(params['loss'], outputs, targets, rng=rng)
    
    return GRNNCell(init_params, cell.init_local, step_with_loss, init_B_traces=cell.init_B_traces, init_lambda_traces=cell.init_lambda_traces, init_gamma_traces=cell.init_gamma_traces, init_phi_traces=cell.init_phi_traces)

def reverse_outputs(f: Callable):
    return lambda *args, **kwargs: f(*args, **kwargs)[::-1]


def with_truncated_grads(cell: BaseGRNNCell, has_aux=False) -> GRNNCell:
    if has_aux:
        def update_and_loss(*args, **kwargs):
            states, outputs = cell.step(*args, **kwargs)
            return outputs[0], (states, *outputs[1:])
        
        update_and_loss = jax.value_and_grad(update_and_loss, has_aux=True)

        def step_with_loss(params, states, inputs, targets, rng=None):
            (loss, (new_states, *extra_outputs)), grads = update_and_loss(params, states, inputs, targets, rng=rng)

            # Print the jaxpr of the value and grad to inspect the computation graph
            jaxpr_with_grad = jax.make_jaxpr(update_and_loss)(params, states, inputs, targets, rng=rng)

            jax.debug.print("jaxpr_with_grad: {}", jaxpr_with_grad)

            return new_states, (loss, grads, *extra_outputs)
    else:
        update_and_loss = jax.value_and_grad(reverse_outputs(cell.step), has_aux=True)

        def step_with_loss(params, states, inputs, targets, rng=None):
            (loss, new_states), grads = update_and_loss(params, states, inputs, targets, rng=rng)

            # Print the jaxpr of the value and grad to inspect the computation graph
            jaxpr_with_grad = jax.make_jaxpr(update_and_loss)(params, states, inputs, targets, rng=rng)

            # jax.debug.print("jaxpr_with_grad: {}", jaxpr_with_grad)

            return new_states, (loss, grads)
        
    return GRNNCell(cell.init_params, cell.init_local, step_with_loss, init_B_traces=cell.init_B_traces, init_lambda_traces=cell.init_lambda_traces, init_gamma_traces=cell.init_gamma_traces, init_phi_traces=cell.init_phi_traces)


def with_feedforward_and_truncated_grads(cell: GRNNCell, loss: FFModel) -> GRNNCell:
    return with_truncated_grads(with_feedforward_loss(cell, loss), has_aux=loss.has_aux)


