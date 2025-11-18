import jax
import jax.numpy as jnp


def get_sampler_link_regression(num_nodes, delay=0, alpha=0.5, feedthrough=0, batch_size=None, batching_strategy="none"):
    
    def init(rng):
        buffer = jnp.zeros((num_nodes, delay))
        return (buffer, rng)
    
    def process_edge(buffer, edge):
        src, dst, jmp = edge

        if delay==0:
            return buffer, - jmp
        
        y_src, y_dst = buffer[src, -1], buffer[dst, -1]
        buffer = (buffer
            .at[src, 1:].set(buffer[src, :-1])
            .at[dst, 1:].set(buffer[dst, :-1])
            .at[src, 0].set(alpha*y_dst + (1-alpha)*jmp)
            .at[dst, 0].set(alpha*y_src + (1-alpha)*jmp)
        )

        return buffer, y_src + y_dst - feedthrough*jmp

    def choose_edge(rng):
        src, dst = jax.random.choice(rng[0], num_nodes, (2, batch_size) if batch_size is not None else (2,), replace=False)
        #return 1, 0
        return src, dst

    @jax.jit
    def step(state, _=None):
        buffers, rng = state
        rng, jumps_rng, *edge_rng = jax.random.split(rng, 4)

        src, dst = choose_edge(edge_rng) 
        
        jumps = jax.random.normal(jumps_rng, (batch_size,) if batch_size is not None else ())
        buffers, target = process_edge(buffers, (src, dst, jumps))

        return (buffers, rng), (src, dst, jumps, target)

    return init, step, num_nodes, None, 1, 1  # feature size = 1 (the jump), output size = 1 (the regression target)
