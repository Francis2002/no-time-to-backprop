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

def SymmetricLRUCell(state_size, input_size, initializer=nn.initializers.lecun_normal):
    """
    LRU cell with complex-valued parameters.
    
    Forward pass:
      h_t = diag_lambda * h_{t-1} + diag_gamma * (W * (inputs transformed))
      
    where:
      diag_lambda = exp(-exp(nu) + 1j * exp(theta))
      diag_gamma  = exp(gamma)
      W = W_re + 1j * W_im
      
    The cell uses an extra parameter 'wi' to transform the incoming lower-layer inputs,
    so that the interface matches that of your GRU cell.
    """
    def init_params(rng):
        initializer_instance = initializer()
        return {
            # For simplicity we use zeros for the 1D parameters.
            'nu': jnp.zeros((2, state_size)),
            'theta': jnp.zeros((2, state_size)),
            'gamma': jnp.sqrt(1 - jnp.exp(jnp.zeros((2, state_size)))),
            # Input transformation matrix.
            'wi_re': initializer_instance(rng, (input_size, state_size)),
            'wi_im': initializer_instance(rng, (input_size, state_size)),
            'b': jnp.zeros(state_size),
        }
    
    # We continue to use the same initializer for the hidden state.
    init_state = zero_local_state_initializer(state_size)

    state_axes = 2
    entity_axis = -2
    
    def step(params, states, inputs, rng):
        
        # Transform the lower-layer input to state-space.
        # The tensordot contracts over the input dimension, yielding shape (B, 2, state_size).
        wi_complex = params['wi_re'] + 1j * params['wi_im']
        winputs = jnp.tensordot(inputs, wi_complex, axes=1) + params['b']

        print("winputs.shape", winputs.shape)
        print("params['nu'].shape", params['nu'].shape)
        print("states.shape", states.shape)
        
        # Compute the complex diagonal factors:
        # diag_lambda = exp(-exp(nu) + 1j * exp(theta))
        lambda_val = jnp.exp(-jnp.exp(params['nu']) + 1j * jnp.exp(params['theta']))
        # diag_gamma = exp(gamma)
        gamma_val = jnp.exp(params['gamma'])
        
        # Ensure that the recurrent state is complex.
        states_complex = states.astype(jnp.complex64)

        lambda_h = (jnp.sum(lambda_val * states_complex, axis=entity_axis), jnp.sum(lambda_val[::-1] * states_complex, axis=entity_axis))

        print("lambda_h.shape before", [w.shape for w in lambda_h])

        lambda_h = jnp.stack(lambda_h, axis=entity_axis) #-2

        print("lambda_h.shape after", lambda_h.shape)
        
        # Update the state.
        new_state = lambda_h + gamma_val * winputs # Here there is broadcasting happening, that's why the sizes don't seem to match.

        print("new_state.shape", new_state.shape)
        
        # Return both the new state and a flattened version.
        return new_state, jnp.real(new_state.reshape(new_state.shape[:-2] + (-1,)))
    
    return GRNNCell(init_params, init_state, step, symmetric=True)



def SymmetricLRUCellReal(state_size, input_size, initializer=nn.initializers.lecun_normal):
    """
    Simpler (real-only) LRU cell.
    
    Forward pass:
      h_t = diag_lambda * h_{t-1} + diag_gamma * (W * (inputs transformed))
      
    where:
      diag_lambda = exp(-exp(nu))
      diag_gamma  = exp(gamma)
      W is a real weight matrix.
      
    An input transformation 'wi' is used so that this cell can replace the GRU cell.
    """
    def init_params(rng):
        initializer_instance = initializer()
        return {
            # parameters for the diagonal recurrent update (real-valued)
            'nu': jnp.zeros((2, state_size)),
            'gamma': jnp.sqrt(1 - jnp.exp(jnp.zeros((2, state_size)))),
            # input transformation (from lower layer to state space)
            'wi': initializer_instance(rng, (input_size, state_size)),
            'b': jnp.zeros(state_size),
        }
    
    # Initial state as zeros with shape (2, state_size) and real dtype.
    init_state = zero_local_state_initializer(state_size)

    state_axes = 2
    entity_axis = -2
    
    def step(params, states, inputs, rng):
        # Process inputs:

        winputs = jnp.tensordot(inputs, params['wi'], axes=1) + params['b']

        print("winputs.shape", winputs.shape)
        print("params['nu'].shape", params['nu'].shape)
        print("states.shape", states.shape)

        diag_lambda = jnp.exp(-jnp.exp(params['nu']))
        gamma_val  = jnp.exp(params['gamma'])        
        lambda_h = (jnp.sum(diag_lambda * states, axis=entity_axis), jnp.sum(diag_lambda[::-1] * states, axis=entity_axis))

        print("wstates_h.shape before", [w.shape for w in lambda_h])

        lambda_h = jnp.stack(lambda_h, axis=entity_axis) #-2

        print("wstates_h.shape after", lambda_h.shape)

        # Update state:
        new_state = lambda_h + gamma_val * winputs

        print("new_state.shape", new_state.shape)

        # Return both the new state (for storage) and a flattened version (for output)
        return new_state, new_state.reshape(new_state.shape[:-2] + (-1,))
    
    return GRNNCell(init_params, init_state, step, symmetric=True)

def MultiLayerLRUCellReal(state_size, input_size, initializer=nn.initializers.lecun_normal, num_layers=4):
    """
    A multi-layer real-valued LRU cell.
    
    For the first layer (l=0):
      h_t^(0) = exp(-exp(nu^(0))) ⊙ h_{t-1}^(0) + exp(gamma^(0)) ⊙ (W_in · input)
      
    For layers l >= 1:
      h_t^(l) = exp(-exp(nu^(l))) ⊙ h_{t-1}^(l) + exp(gamma^(l)) ⊙ (W^(l) · h_t^(l-1))
      
    The cell returns the new multi-layer state (as a tuple of states, one per layer)
    and a flattened version of the final layer's state.
    """
    def init_params(rng):
        # We use a 1-D initializer for nu and gamma.
        zeros_init = jax.nn.initializers.zeros
        initializer_instance = initializer()
        params = {}
        rngs = jax.random.split(rng, num_layers)
        # Layer 0: has an input transformation (wi)
        params['layer0'] = {
            # parameters for the diagonal recurrent update (real-valued)
            'nu': jnp.zeros((2, state_size)),
            'gamma': jnp.sqrt(1 - jnp.exp(jnp.zeros((2, state_size)))),
            # input transformation (from lower layer to state space)
            'wi': initializer_instance(rng, (input_size, state_size)),
            'b': jnp.zeros(state_size),
        }
        # Layers 1...num_layers-1: each has a weight matrix mapping from previous layer state.
        for l in range(1, num_layers):
            params[f'layer{l}'] = {
                # parameters for the diagonal recurrent update (real-valued)
                'nu': jnp.zeros((2, state_size)),
                'gamma': jnp.sqrt(1 - jnp.exp(jnp.zeros((2, state_size)))),
                # input transformation (from lower layer to state space)
                'W': initializer_instance(rng, (state_size, state_size)),
                'b': jnp.zeros(state_size),
            }
        return params

    # Initial state for each layer: we use the same zero initializer as before.
    # Each state is assumed to be of shape (2, state_size) (for the two entities).
    def init_state(batch_size):
        return tuple(zero_local_state_initializer(state_size)(batch_size) for _ in range(num_layers))

    state_axes = 2
    entity_axis = -2
    
    def single_layer_step(params, states, inputs, input_layer=False):

        # Process inputs.
        if input_layer:
            winputs = jnp.tensordot(inputs, params['wi'], axes=1) + params['b']
        else:
            winputs = jnp.tensordot(inputs, params['W'], axes=1) + params['b']

        print("winputs.shape", winputs.shape)
        print("params['nu'].shape", params['nu'].shape)
        print("states.shape", states.shape)

        diag_lambda = jnp.exp(-jnp.exp(params['nu']))
        gamma_val  = jnp.exp(params['gamma'])        
        lambda_h = (jnp.sum(diag_lambda * states * 0.5, axis=entity_axis), jnp.sum(diag_lambda[::-1] * states * 0.5, axis=entity_axis))
        print("wstates_h.shape before", [w.shape for w in lambda_h])

        lambda_h = jnp.stack(lambda_h, axis=entity_axis) #-2

        print("wstates_h.shape after", lambda_h.shape)

        # Update state:
        new_state = lambda_h + gamma_val * winputs

        print("new_state.shape", new_state.shape)

        # Return both the new state (for storage) and a flattened version (for output)
        return new_state, new_state.reshape(new_state.shape[:-2] + (-1,))


    def step(params, states, inputs, rng):
        """
        'states' is a tuple of states for each layer.
        For layer 0, 'inputs' is the external input.
        For l>=1, the input is the output (hidden state) of the previous layer.
        """
        new_states = []
        # Process layer 0.
        s0, _ = single_layer_step(params['layer0'], states[0], inputs, input_layer=True)
        new_states.append(s0)
        current_input = s0  # for the next layer
        # Process subsequent layers.
        for l in range(1, num_layers):
            sl, _ = single_layer_step(params[f'layer{l}'], states[l], current_input, input_layer=False)
            new_states.append(sl)
            current_input = sl
        # Use the last layer's flattened state as the output.
        output = new_states[-1].reshape(new_states[-1].shape[:-2] + (-1,))
        return tuple(new_states), output

    return GRNNCell(init_params, init_state, step, symmetric=True)

def MultiLayerLRUCell(state_size, input_size, initializer=nn.initializers.lecun_normal, num_layers=4):
    """
    Multi-layer complex-valued LRU cell.
    
    For layer 0 (input layer):
      h_t^(0) = exp(-exp(nu^(0)) + 1j * exp(theta^(0))) ⊙ h_{t-1}^(0)
                + exp(gamma^(0)) ⊙ (wi · input)
    
    For layers l >= 1:
      h_t^(l) = exp(-exp(nu^(l)) + 1j * exp(theta^(l))) ⊙ h_{t-1}^(l)
                + exp(gamma^(l)) ⊙ (W^(l) · h_t^(l-1))
      where W^(l) = W_re^(l) + 1j * W_im^(l)
    
    The cell returns a tuple of new states (one per layer) and a flattened real-valued output
    from the final layer.
    """
    def init_params(rng):
        initializer_instance = initializer()
        params = {}
        # Split the RNG for each layer.
        rngs = jax.random.split(rng, num_layers)
        # Layer 0: uses an input transformation "wi".
        params['layer0'] = {
            'nu': jnp.zeros(state_size),
            'theta': jnp.zeros(state_size),
            'gamma': jnp.zeros(state_size),
            'wi': initializer_instance(rngs[0], (input_size, state_size))
        }
        # Layers 1,...,num_layers-1: each uses a complex weight matrix.
        for l in range(1, num_layers):
            params[f'layer{l}'] = {
                'nu': jnp.zeros(state_size),
                'theta': jnp.zeros(state_size),
                'gamma': jnp.zeros(state_size),
                'W_re': initializer_instance(rngs[l], (state_size, state_size)),
                'W_im': initializer_instance(rngs[l], (state_size, state_size))
            }
        return params

    # Define an initializer for the cell's state.
    # Each layer's state is expected to be of shape (2, state_size) for a single example.
    def init_state(batch_size):
        return tuple(zero_local_state_initializer(state_size)(batch_size) for _ in range(num_layers))
    
    def single_layer_step(params_layer, state, inp, input_layer=False):
        # Ensure 'state' has a batch dimension.
        squeeze = False
        if state.ndim == 2:  # (2, state_size)
            state = jnp.expand_dims(state, axis=0)  # becomes (1, 2, state_size)
            squeeze = True

        # Process the input for this layer.
        if input_layer:
            # For the input layer, if inp is 1D, add a batch dimension.
            if inp.ndim == 1:
                inp = jnp.expand_dims(inp, axis=0)  # (1, input_size)
            if inp.ndim == 2:
                # Expand to (B, 2, input_size)
                inp = jnp.expand_dims(inp, axis=1).repeat(2, axis=1)
            # Transform the input via 'wi'.
            transformed = jnp.tensordot(inp, params_layer['wi'], axes=([-1], [0]))  # (B, 2, state_size)
            feedforward = transformed  # No extra weight multiplication in layer 0.
        else:
            # For higher layers, inp is the output from the previous layer.
            if inp.ndim == 2:  # missing batch dim? (2, state_size)
                inp = jnp.expand_dims(inp, axis=0)  # (1, 2, state_size)
            # Assemble the complex weight matrix.
            W_complex = params_layer['W_re'] + 1j * params_layer['W_im']
            feedforward = jnp.einsum('ij,bnj->bni', W_complex, inp)  # (B, 2, state_size)

        # Compute complex diagonal factors.
        diag_lambda = jnp.exp(-jnp.exp(params_layer['nu']) + 1j * jnp.exp(params_layer['theta']))  # (state_size,)
        diag_gamma = jnp.exp(params_layer['gamma'])  # (state_size,)

        # Make sure the state is complex.
        state_complex = state.astype(jnp.complex64)
        new_state = diag_lambda * state_complex + diag_gamma * feedforward

        if squeeze:
            new_state = new_state[0]
        flat_out = new_state.reshape(new_state.shape[:-2] + (-1,))
        # Cast the flattened output to real.
        return new_state, jnp.real(flat_out)
    
    def step(params, states, inputs, rng):
        new_states = []
        # Process layer 0.
        s0, _ = single_layer_step(params['layer0'], states[0], inputs, input_layer=True)
        new_states.append(s0)
        current_inp = s0
        # Process subsequent layers.
        for l in range(1, num_layers):
            s_l, _ = single_layer_step(params[f'layer{l}'], states[l], current_inp, input_layer=False)
            new_states.append(s_l)
            current_inp = s_l
        # Flatten the output of the final layer and cast to real.
        out = new_states[-1].reshape(new_states[-1].shape[:-2] + (-1,))
        return tuple(new_states), jnp.real(out)
    
    return GRNNCell(init_params, init_state, step, symmetric=True)


def SymmetricMinGRUCell(state_size, input_size, initializer=nn.initializers.lecun_normal):
    def init_params(rng):
        initializer_instance = initializer()
        # We now need only one weight matrix for the input:
        # - The first half produces the update gate z (using a sigmoid).
        # - The second half produces the candidate h_tilde (with a linear transformation).
        return {
            'wh_h': initializer_instance(rng, (2, state_size, state_size)),
            'nu': jnp.zeros((2, state_size)),
            'theta': jnp.zeros((2, state_size)),
            'wi': initializer_instance(rng, (input_size, 2*state_size)),
            'b': jnp.zeros(2*state_size),
        }

    # Use the same initializer for the state as in the GRU.
    init_state = zero_local_state_initializer(state_size)

    # The GRNN framework expects states of shape (B, 2, H) and allows for
    # inputs of shape (B, I) or (B, 2, I) in case the src and dst features differ.
    state_axes = 2
    entity_axis = -2

    def step(params, states, inputs, rng):
        
        winputs = jnp.tensordot(inputs, params['wi'], axes=1) + params['b']
        winputs_z, winputs_h = jnp.split(winputs, 2, -1)
        z = jax.nn.sigmoid(winputs_z) # [B, 2, 2H]

        print("winputs.shape", winputs.shape)
        print("winputs_z.shape", winputs_z.shape)
        print("winputs_h.shape", winputs_h.shape)
        print("z.shape", z.shape)

        print("params['wh_h'].shape", params['wh_h'].shape)
        
        #wstates_h = (jnp.tensordot(states, params['wh_h'], axes=state_axes), jnp.tensordot(states, params['wh_h'][::-1], axes=state_axes))

        print("params['nu'].shape", params['nu'].shape)
        print("states.shape", states.shape)

        diag_wh_h = jnp.exp(-jnp.exp(params['nu']))
        wstates_h = (jnp.sum(diag_wh_h * states, axis=entity_axis), jnp.sum(diag_wh_h[::-1] * states, axis=entity_axis))

        print("wstates_h.shape before", [w.shape for w in wstates_h])

        wstates_h = jnp.stack(wstates_h, axis=entity_axis) #-2

        print("wstates_h.shape after", wstates_h.shape)
        h_tilde = wstates_h + winputs_h # chould have a non-linearity here

        next_state = (1 - z) * states + z * h_tilde

        print("next_state.shape", next_state.shape)

        return next_state, next_state.reshape(next_state.shape[:-2] + (-1,))

    # Return a GRNNCell constructed with our parameters and step function.
    return GRNNCell(init_params, init_state, step, symmetric=True)

def MultiLayerMinGRUCell(state_size, input_size, initializer=nn.initializers.lecun_normal, num_layers=4):
    """
    Multi-layer minGRU cell.
    
    For layer 0 (input layer):
      h_t^(0) = (1 - z) ⊙ h_{t-1}^(0) + z ⊙ (wi · input + b)
      where z = sigmoid((wi · input + b)_first_half)
            candidate = (wi · input + b)_second_half

    For layers l >= 1:
      h_t^(l) = (1 - z) ⊙ h_{t-1}^(l) + z ⊙ (W^(l) · h_t^(l-1) + b)
      where z = sigmoid((W^(l) · h_t^(l-1) + b)_first_half)
            candidate = (W^(l) · h_t^(l-1) + b)_second_half

    The cell returns a tuple of new states (one per layer) and a flattened real-valued output
    from the final layer.
    """
    def init_params(rng):
        initializer_instance = initializer()
        params = {}
        rngs = jax.random.split(rng, num_layers)
        # Layer 0: transform the input.
        params['layer0'] = {
            'wh_h': initializer_instance(rngs[0], (2, state_size, state_size)),
            'nu': jnp.zeros((2, state_size)),
            'theta': jnp.zeros((2, state_size)),
            'wi': initializer_instance(rngs[0], (input_size, 2*state_size)),
            'b': jnp.zeros(2*state_size),
        }
        # Layers 1,...,num_layers-1: transform the previous layer's state.
        for l in range(1, num_layers):
            params[f'layer{l}'] = {
                'wh_h': initializer_instance(rngs[l], (2, state_size, state_size)),
                'nu': jnp.zeros((2, state_size)),
                'theta': jnp.zeros((2, state_size)),
                'W': initializer_instance(rngs[l], (state_size, 2*state_size)),
                'b': jnp.zeros(2*state_size),
            }
        return params

    # Each layer's state is expected to be of shape (2, state_size) per example.
    def init_state(batch_size):
        return tuple(zero_local_state_initializer(state_size)(batch_size) for _ in range(num_layers))
    
    state_axes = 2
    entity_axis = -2

    def single_layer_step(params_layer, states, inp, input_layer=False):

        if input_layer:
            winputs = jnp.tensordot(inp, params_layer['wi'], axes=1) + params_layer['b']
        else:
            winputs = jnp.tensordot(inp, params_layer['W'], axes=1) + params_layer['b']

        winputs_z, winputs_h = jnp.split(winputs, 2, -1)
        z = jax.nn.sigmoid(winputs_z) # [B, 2, 2H]

        print("winputs.shape", winputs.shape)
        print("winputs_z.shape", winputs_z.shape)
        print("winputs_h.shape", winputs_h.shape)
        print("z.shape", z.shape)

        print("params['wh_h'].shape", params_layer['wh_h'].shape)
        
        #wstates_h = (jnp.tensordot(states, params_layer['wh_h'], axes=state_axes), jnp.tensordot(states, params_layer['wh_h'][::-1], axes=state_axes))

        print("params['nu'].shape", params_layer['nu'].shape)
        print("states.shape", states.shape)

        diag_wh_h = jnp.exp(-jnp.exp(params_layer['nu']))
        wstates_h = (jnp.sum(diag_wh_h * states, axis=entity_axis), jnp.sum(diag_wh_h[::-1] * states, axis=entity_axis))

        print("wstates_h.shape before", [w.shape for w in wstates_h])

        wstates_h = jnp.stack(wstates_h, axis=entity_axis) #-2

        print("wstates_h.shape after", wstates_h.shape)
        h_tilde = wstates_h + winputs_h # chould have a non-linearity here

        next_state = (1 - z) * states + z * h_tilde

        print("next_state.shape", next_state.shape)

        return next_state, next_state.reshape(next_state.shape[:-2] + (-1,))

    def step(params, states, inputs, rng):
        new_states = []
        # Process layer 0 with the raw input.
        s0, _ = single_layer_step(params['layer0'], states[0], inputs, input_layer=True)
        new_states.append(s0)
        current_inp = s0
        # Process subsequent layers.
        for l in range(1, num_layers):
            s_l, _ = single_layer_step(params[f'layer{l}'], states[l], current_inp, input_layer=False)
            new_states.append(s_l)
            current_inp = s_l
        out = current_inp.reshape(current_inp.shape[:-2] + (-1,))
        return tuple(new_states), out

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