import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from flowjax.distributions import StandardNormal, Transformed
from flowjax.bijections import Coupling, RationalQuadraticSpline, Chain, Permute, Affine

def count_parameters(model):
    """
    Counts the number of learnable parameters in an Equinox model.
    """
    params = eqx.filter(model, eqx.is_inexact_array)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    return num_params

def build_flow_model(key):
    bound = 5.0
    K = 8
    num_layers = 16
    base_dist = StandardNormal((3,))

    # Use dummy loc and scale for the Affine renormalizer (doesn't affect parameter count)
    loc = jnp.zeros(3)
    scale = jnp.ones(3)
    renormalizer = Affine(loc=loc, scale=scale)

    layers = []
    flow_key = key
    
    for i in range(num_layers):
        flow_key, subkey = jr.split(flow_key, 2)
        
        layers.append(Coupling(
            key=subkey,
            transformer=RationalQuadraticSpline(knots=K, interval=bound),
            untransformed_dim=2,  
            dim=3,
            nn_width=32,
            nn_depth=2
        ))
        
        permutation = jnp.roll(jnp.arange(3), shift=1)
        layers.append(Permute(permutation))

    layers.append(renormalizer)

    bijection = Chain(layers)
    flow = Transformed(base_dist, bijection)
    
    return flow

if __name__ == '__main__':
    # Initialize a random key
    key = jr.PRNGKey(127)
    
    # Build the exact flowjax model from the original script
    flow_model = build_flow_model(key)
    print("FlowJAX model initialized.")
    
    # Calculate parameters
    total_params = count_parameters(flow_model)
    
    print(f"Total number of learnable parameters: {total_params:,}")
