import numpy as np
import jax
import jax.numpy as jnp
from functools import wraps
from jax import nn
from jax.tree_util import tree_map
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, Union

from tgap.seq_model import ClassificationModel, StackedEncoder, BatchClassificationModel


def local_state_initializer(entity_state_sizes: Union[int, list]):
    """Defines decorator for a local state initializer function.

    Args:
        entity_state_sizes (int | list): size of hidden state (if one type of entity) 
        or list of hidden state sizes for each type of entity.

    Returns:
        A decorator to apply to an initializer function.
    """
    if np.isscalar(entity_state_sizes): 
        def _local_state_initializer(initializer):
            @wraps(initializer)
            def wrapped_initializer(batch_size: Union[int, list], entity: None=None):
                shape = [batch_size, entity_state_sizes] if np.isscalar(batch_size) else [*batch_size, entity_state_sizes]
                    
                return initializer(shape)

            return wrapped_initializer
    else:
        def _local_state_initializer(initializer):
            @wraps(initializer)
            def wrapped_initializer(batch_size: Union[int, list], entity: int):
                state_size = entity_state_sizes[entity]
                shape = [batch_size, state_size] if np.isscalar(batch_size) else [*batch_size, state_size]
                    
                return initializer(shape)

            return wrapped_initializer
    
    return _local_state_initializer


def zero_local_state_initializer(entity_state_sizes: Union[int, list], dtype=jnp.float32):
    """Defines a local state initializer that outputs a single N-D zero array.

    Args:
        entity_state_sizes (int | list): size of hidden state (if one type of entity) 
        or list of hidden state sizes for each type of entity.

    Returns:
        An initializer function that outputs a N-D zero array of the appropriate size.
    """

    @local_state_initializer(entity_state_sizes)
    def initializer(shape):
        return jnp.zeros(shape, dtype=dtype)
    
    return initializer


class BaseGRNNCell:
    has_global: ClassVar = False

    @staticmethod
    def init_global():
        return None


@dataclass(frozen=True)
class GRNNCell(BaseGRNNCell):
    init_params: Callable
    init_local: Callable
    step: Callable
    num_entities: int = 2
    symmetric: Optional[bool] = False
    init_B_traces: Optional[Callable] = None
    init_lambda_traces: Optional[Callable] = None
    init_gamma_traces: Optional[Callable] = None
    init_phi_traces: Optional[Callable] = None
    update_gradients: Optional[Callable] = None


def SymmetricGRUCell(state_size, input_size, initializer=nn.initializers.lecun_normal):
    def init_params(rng):
        initializer_instance = initializer()
        
        return {
            'wh_zr': initializer_instance(rng, (2, state_size, 2*state_size)),
            'wh_h': initializer_instance(rng, (2, state_size, state_size)),
            'wi': initializer_instance(rng, (input_size, 3*state_size)),
            'b': jnp.zeros(3*state_size),
        }

    init_state = zero_local_state_initializer(state_size)

    # States shape is (B, 2, H)
    # Inputs shape is (B, I) or (B, 2, I) in case inputs to src and dst are different
    state_axes = 2
    entity_axis = -2
    def step(params, states, inputs, rng):
    
        winputs = jnp.tensordot(inputs, params['wi'], axes=1) + params['b']
        winputs_zr, winputs_h = winputs[..., :2*state_size], winputs[..., 2*state_size:]

        wstates_zr = (jnp.tensordot(states, params['wh_zr'], axes=state_axes), jnp.tensordot(states, params['wh_zr'][::-1], axes=state_axes))
        wstates_zr = jnp.stack(wstates_zr, axis=entity_axis) #-2
        zr = jax.nn.sigmoid(wstates_zr + winputs_zr) # [B, 2, 2H]
        
        z, r = jnp.split(zr, 2, axis=-1) # [B, 2, H]

        rstates = r*states
        wstates_h = (jnp.tensordot(rstates, params['wh_h'], axes=state_axes), jnp.tensordot(rstates, params['wh_h'][::-1], axes=state_axes))
        wstates_h = jnp.stack(wstates_h, axis=entity_axis) #-2
        h = jnp.tanh(wstates_h + winputs_h)

        next_state = (1 - z)*states + z*h

        return next_state, next_state.reshape(next_state.shape[:-2] + (-1,))

    return GRNNCell(init_params, init_state, step, symmetric=True)

def ZucchetCell(
    rec=None,
    input_rec=None,
    output_rec=None,
    single_unique_rec=None,
    rec_type='LRU',
    d_input=1,
    d_output=1,
    d_model=32,
    n_layers=6,
    seq_length=1,
    padded=False,
    activation='full_glu',
    readout=0,
    dropout=0.1,
    mode='none',
    prenorm=True,
    postnorm=False,
    multidim=1,
    training=True,
    training_mode='bptt',
    d_hidden=64,
    in_dim=1,
    bsz=None,
    has_encoder=True,
    decoder_type="MLP",
    has_extra_skip=True,
    mixing="full",
    has_non_linearity_in_recurrence=False,
):
    """
    ZucchetCell: A configurable recurrent cell with multiple layer support.
    
    Args:
        rec: Recurrent component configuration
        rec_type: Type of recurrence ('LRU', etc.)
        d_input: Input dimension
        d_output: Output dimension
        d_model: Model dimension (intermediate size -> not hidden state size)
        n_layers: Number of layers
        seq_length: Sequence length
        padded: Whether to use padding
        activation: Activation function type
        readout: Readout configuration
        dropout: Dropout rate
        mode: Operating mode
        prenorm: Whether to use pre-normalization
        multidim: Multi-dimension parameter
        training: Whether in training mode
        training_mode: Training mode type
        initializer: Weight initializer function
        
    Returns:
        A GRNNCell instance
    """

    if bsz == None:

        model = ClassificationModel(
            rec=rec,
            input_rec=input_rec,
            output_rec=output_rec,
            single_unique_rec=single_unique_rec,
            rec_type=rec_type,
            d_input=d_input,
            d_output=d_output,
            d_model=d_model,
            n_layers=n_layers,
            seq_length=seq_length,
            padded=padded,
            activation=activation,
            readout=readout,
            dropout=dropout,
            mode=mode,
            prenorm=prenorm,
            postnorm=postnorm,
            multidim=multidim,
            training=training,
            training_mode=training_mode,
            has_encoder=has_encoder,
            decoder_type=decoder_type,
            has_extra_skip=has_extra_skip,
            has_non_linearity_in_recurrence=has_non_linearity_in_recurrence,
        )
    
    else:
        model = BatchClassificationModel(
            rec=rec,
            input_rec=input_rec,
            output_rec=output_rec,
            single_unique_rec=single_unique_rec,
            rec_type=rec_type,
            d_input=d_input,
            d_output=d_output,
            d_model=d_model,
            n_layers=n_layers,
            seq_length=seq_length,
            padded=padded,
            activation=activation,
            readout=readout,
            dropout=dropout,
            mode=mode,
            prenorm=prenorm,
            postnorm=postnorm,
            multidim=multidim,
            training=training,
            training_mode=training_mode,
            has_encoder=has_encoder,
            decoder_type=decoder_type,
            has_extra_skip=has_extra_skip,
            has_non_linearity_in_recurrence=has_non_linearity_in_recurrence,
        )


    def init_params(rng):

        dummy_input = jnp.ones((in_dim, )) if bsz is None else jnp.ones((bsz, in_dim))
        dummy_hidden_states = tuple(jnp.zeros((2, d_hidden)) for _ in range(n_layers)) if bsz is None else tuple(jnp.zeros((bsz, 2, d_hidden)) for _ in range(n_layers))

        init_rng, dropout_rng = jax.random.split(rng, num=2)
        variables = model.init({'params': init_rng, 'dropout': dropout_rng}, dummy_input, dummy_hidden_states)

        if training_mode in ['bptt', 'online_spatial']:
            return variables['params']
        
        # Perturbations
        param_and_perts = {
            "params": variables['params'],
            "perturbations": variables.get('perturbations')
        }

        return param_and_perts
        
    # Each layer's state is expected to be of shape (2, state_size) per example.
    def init_state(batch_size):
        return tuple(zero_local_state_initializer(d_hidden, dtype=jnp.complex64)(batch_size) for _ in range(n_layers))
    
    def init_B_traces(batch_size):
        # Order of traces is lambda, gamma, B

        if has_encoder:
            return tuple(jnp.zeros((batch_size, 2*d_hidden, d_model), dtype=jnp.complex64) for _ in range(n_layers))
        else:
            # First layer uses d_input, subsequent layers use d_model
            traces = []
            for layer_idx in range(n_layers):
                if layer_idx == 0:
                    traces.append(jnp.zeros((batch_size, 2*d_hidden, d_input), dtype=jnp.complex64))
                else:
                    traces.append(jnp.zeros((batch_size, 2*d_hidden, d_model), dtype=jnp.complex64))
            return tuple(traces)
    
    def init_lambda_traces(batch_size):
        return tuple(zero_local_state_initializer(4*d_hidden, dtype=jnp.complex64)(batch_size) for _ in range(n_layers))

    def init_gamma_traces(batch_size):
        return tuple(zero_local_state_initializer(2*d_hidden, dtype=jnp.complex64)(batch_size) for _ in range(n_layers))
    
    def init_phi_traces(batch_size):
        return tuple(zero_local_state_initializer(d_hidden, dtype=jnp.complex64)(batch_size) for _ in range(n_layers))

    def step(params, states, inputs, rng):

        if training_mode in ['bptt', 'online_spatial']:
            # If there is nested params ({"params": ... }, unwrap it)
            if 'params' in params:
                params = params['params']
                
            out, new_states = model.apply({"params": params}, inputs, states, rngs={'dropout': rng})
            return tuple(new_states), out
        # Unpack the states and traces
        states, traces = states
        out, new_states, new_traces = model.apply(params, inputs, states, traces, rngs={'dropout': rng})
        return (tuple(new_states), tuple(new_traces)), out
    
    def update_gradients(params, grads, traces, perturbation_grads):
        updated_grads = model.apply(params, grads['cell']['params'], traces, perturbation_grads, method=model.update_gradients)
        return updated_grads
        
    return GRNNCell(init_params, init_state, step, symmetric=True, init_lambda_traces=init_lambda_traces, init_gamma_traces=init_gamma_traces, init_B_traces=init_B_traces, init_phi_traces=init_phi_traces, update_gradients=update_gradients)