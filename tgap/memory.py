import jax.numpy as jnp
import numpy as np
from jax import jit, tree_map
from jax.tree_util import tree_flatten, tree_unflatten

from jax import vmap
from functools import partial


def memory_store(example_state, num_entries):
    # Remove the batch dimension from example_state before creating the store
    unbatched_state = tree_map(lambda x: x[0], example_state)
    init = vmap(jnp.zeros_like, in_axes=None, out_axes=0, axis_size=num_entries)
    return tree_map(init, unbatched_state)
    
def store_get(store, indices):
    print('compiling getter...')
    getter = partial(jnp.take, indices=indices, axis=0, mode='clip')
    return tree_map(getter, store)

def store_set(store, indices, values):
    print('compiling setter...')
    setter = lambda s, v: s.at[indices].set(v)
    return tree_map(setter, store, values)

def dedupe_indices(indices):
    indices, dindices = jnp.unique(indices[::-1], return_index=True, size=indices.size, fill_value=-1)
    dindices = indices.size - dindices - 1
    return indices, dindices

def store_set_dedupe(store, indices, values):
    print('compiling setter...')
    indices, dindices = dedupe_indices(indices)
    setter = lambda s, v: s.at[indices].set(v[dindices])
    return tree_map(setter, store, values)

#%%
# example_state = {'a': jnp.arange(10), 'b': (3*jnp.ones(5), 2*jnp.zeros(3))}

# store = memory_store(example_state, 10)
# getter = jit(store_get)
# setter = jit(store_set)
# #%%
# along_batch = jnp.arange(1, 4)[:, None]
# new_state = tree.map(lambda a: a*along_batch, example_state)

# new_store = setter(store, jnp.asarray([3,2,3]), new_state)
# new_store


def state_store(num_nodes, init_state, numpy=True):
    sample_state = init_state(1)
    leaves, treedef = tree_flatten(sample_state)
    del sample_state
    shapes = [(-1,) + np.shape(l)[1:] for l in leaves]

    sizes = np.cumsum([0] + [np.size(l) for l in leaves])

    def unvectorize(v):
        v = [v[..., start:stop].reshape(shape) for start, stop, shape in zip(sizes, sizes[1:], shapes)]
        return tree_unflatten(treedef, v)

    def vectorize(u):
        return jnp.concatenate([ui.reshape((ui.shape[0], -1)) for ui in tree_flatten(u)[0]], -1)
    
    jit_vectorize = jit(vectorize)

    def init(*args, **kwargs):
        state = vectorize(init_state(num_nodes, *args, **kwargs))
        return np.array(state) if numpy else state
    
    @jit
    def get(store, indexes):
        return unvectorize(store[indexes])

    if numpy:
        def set(store, indexes, values):
            store[indexes] = jit_vectorize(values)
            return store
    else:
        @jit
        def set(store, indexes, values):
            return store.at[indexes].set(vectorize(values))

    return init, get, set
