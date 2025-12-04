from tgap.rec import LRU
import jax.numpy as jnp
import math

inputs = jnp.array([1.0, 2.0]) # shape
y = jnp.ones([1, 3])
# custom init of the params and state

mixing='none'
d_hidden = 3
d_model = 2

if mixing != 'rotational':
    params_states = {
        "params": {
            "B_re": jnp.ones((2*d_hidden, d_model)),
            "B_im": jnp.ones((2*d_hidden, d_model)),
            "C_re": jnp.ones((d_model, 2*d_hidden)),
            "C_im": jnp.ones((d_model, 2*d_hidden)),
            "D": jnp.zeros(d_model),
            "gamma_log": jnp.ones((2*d_hidden,)),
            "nu": jnp.log(jnp.log(2.0 * jnp.ones((4*d_hidden,)))),
            "theta": -1e8 * jnp.ones((4*d_hidden,)),
        },
        "perturbations": {
            "hidden_states": jnp.zeros(2*d_hidden, dtype=jnp.complex64),
        },
    }
else:
    params_states = {
        "params": {
            "B_re": jnp.ones((2*d_hidden, d_model)),
            "B_im": jnp.ones((2*d_hidden, d_model)),
            "C_re": jnp.ones((d_model, 2*d_hidden)),
            "C_im": jnp.ones((d_model, 2*d_hidden)),
            "D": jnp.zeros(d_model),
            "gamma_log": jnp.ones((2*d_hidden,)),
            "nu": jnp.log(jnp.log(2.0 * jnp.ones((4*d_hidden,)))),
            "theta": -1e8 * jnp.ones((4*d_hidden,)),
            "phi": math.pi * 0.25 * jnp.ones((d_hidden,)),
        },
        "perturbations": {
            "hidden_states": jnp.zeros(2*d_hidden, dtype=jnp.complex64),
        },
    }

old_hidden_states = jnp.ones((2, d_hidden), dtype=jnp.complex64) # d_hidden = 3, we have 2 hiddens => 6 here
# for traces, we need a tuple with 3 elements (lambda gama and B) with the same shape as the parameters
old_traces = (
    jnp.zeros((2, 4*d_hidden), dtype=jnp.complex64),
    jnp.zeros((2, 2*d_hidden), dtype=jnp.complex64),
    jnp.zeros((2, 2*d_hidden, d_model), dtype=jnp.complex64),
)

lru = LRU(d_hidden=d_hidden, d_model=d_model, seq_length=1, training_mode='online_full', mixing=mixing)
lru.rec_type = 'LRU'

lru_output, new_hiddens, new_traces = lru.apply(
    params_states,
    inputs,
    old_hidden_states,
    old_traces,
)

print("LRU output:", lru_output)

def print_pytree(obj, indent=0):
    spacing = '  ' * indent
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{spacing}{key}:")
            print_pytree(value, indent + 1)
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            print(f"{spacing}[{index}]:")
            print_pytree(item, indent + 1)
    else:
        print(f"{spacing}{obj}")

print("\nParameters and States Pytree:")
print_pytree(params_states)

print("\nNew Hidden States Pytree:")
print_pytree(new_hiddens)

print("\nNew Traces Pytree:")
print_pytree(new_traces)

print("Lambda:", lru.apply(
    params_states,
    method='get_diag_lambda',
))