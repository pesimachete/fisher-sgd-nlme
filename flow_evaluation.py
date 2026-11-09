import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
jax.config.update("jax_enable_x64", True)
import equinox as eqx
import matplotlib.pyplot as plt
import pickle 
import os

from flowjax.distributions import StandardNormal, Transformed
from flowjax.bijections import Coupling, RationalQuadraticSpline, Chain, Permute, Affine

def main():
    # Create output directory for plots
    output_dir = 'flow_plots'
    os.makedirs(output_dir, exist_ok=True)

    ex_res_path = 'example_res.npy'
    model_path = os.path.join(output_dir, 'trained_flow_model.eqx')
    limit = 100000
    
    print(f"Loading {ex_res_path}...")
    try:
        with open(ex_res_path, 'rb') as f:
            ex_res = pickle.load(f)
    except Exception as e:
        print(f"Error loading {ex_res_path}: {e}")
        return

    evol_z = ex_res.evol_z[:limit]
    factor = ex_res.factor[:limit]
    n_individuals = evol_z.shape[1]

    # Calculate Burn-in
    factor1 = np.where(factor == 1)[0]
    burn_in_idx = factor1[-1] if len(factor1) > 0 else int(limit * 0.2)

    # Calculate MAP / Posterior Mean for each individual
    print(f"Calculating Posterior Mean (MAP) using iterations {burn_in_idx} to {evol_z.shape[0]}...")
    valid_z_samples = evol_z[burn_in_idx:, :, :]
    map_z = np.mean(valid_z_samples, axis=0)  # Shape: (n_individuals, 3)

    latent_names = ['Asymptotic (z_0)', 'Inflexion (z_1)', 'Tau (z_2)']

    x_train = map_z 
    
    # Reconstruct the model structure
    bound = 5.0
    K = 8
    num_layers = 16  # Number of coupling layers
    base_dist = StandardNormal((3,))

    loc = jnp.mean(x_train, axis=0)
    scale = jnp.std(x_train, axis=0)
    renormalizer = Affine(loc=loc, scale=scale)

    layers = []
    flow_key = jr.PRNGKey(127)
    for i in range(num_layers):
        flow_key, subkey1 = jr.split(flow_key, 2)
        layers.append(Coupling(
            key=subkey1,
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

    # Load the trained model
    print(f"Loading trained model from {model_path}...")
    try:
        flow = eqx.tree_deserialise_leaves(model_path, flow)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Sample from the model
    sample_key = jax.random.key(999)
    num_generated_samples = 100000
    x_generated = flow.sample(sample_key, (num_generated_samples,))
    
    # Filter generated samples that fall way outside the training data distribution to improve visualization
    valid_mask = jnp.ones(num_generated_samples, dtype=bool)
    for d in range(3):
        # Calculate sensible bounds based on training data, with a buffer
        buffer = 3.0 * jnp.std(x_train[:, d]) # e.g., 3 standard deviations
        lower_bound = jnp.min(x_train[:, d]) - buffer
        upper_bound = jnp.max(x_train[:, d]) + buffer
        valid_mask = valid_mask & (x_generated[:, d] >= lower_bound) & (x_generated[:, d] <= upper_bound)
        
    x_generated_filtered = x_generated[valid_mask]
    
    print(f"Kept {len(x_generated_filtered)} out of {num_generated_samples} generated samples for plotting.")

    # 2D Histograms (Pixeled Heatmaps)
    pairs = [(0, 1), (0, 2), (1, 2)]
    dim_names = [
        "Dimension 0 (Asymptotic ~ 200)", 
        "Dimension 1 (Inflexion ~ 500)", 
        "Dimension 2 (Tau ~ 150)"
    ]
    
    for (d1, d2) in pairs:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calculate shared bounds based strictly on the training data range (plus a tiny margin)
        # to prevent outliers in the generated data from ruining the scale.
        margin_x = (x_train[:, d1].max() - x_train[:, d1].min()) * 0.05
        margin_y = (x_train[:, d2].max() - x_train[:, d2].min()) * 0.05
        
        x_min = x_train[:, d1].min() - margin_x
        x_max = x_train[:, d1].max() + margin_x
        y_min = x_train[:, d2].min() - margin_y
        y_max = x_train[:, d2].max() + margin_y
        
        # Training Data Heatmap
        ax = axes[0]
        h1 = ax.hist2d(x_train[:, d1], x_train[:, d2], bins=50, cmap='Purples', 
                       range=[[x_min, x_max], [y_min, y_max]])
        fig.colorbar(h1[3], ax=ax)
        ax.set_title(f"Training Data KDE: {dim_names[d1]} vs {dim_names[d2]}")
        ax.set_xlabel(dim_names[d1])
        ax.set_ylabel(dim_names[d2])
        
        # Generated Data Heatmap
        ax = axes[1]
        h2 = ax.hist2d(x_generated_filtered[:, d1], x_generated_filtered[:, d2], bins=50, cmap='Purples',
                       range=[[x_min, x_max], [y_min, y_max]])
        fig.colorbar(h2[3], ax=ax)
        ax.set_title(f"Generated Flow KDE: {dim_names[d1]} vs {dim_names[d2]}")
        ax.set_xlabel(dim_names[d1])
        ax.set_ylabel(dim_names[d2])
        
        plt.tight_layout()
        plt_path = os.path.join(output_dir, f"heatmap_pixel_dim_{d1}_vs_{d2}.svg")
        plt.savefig(plt_path, format="svg")
        plt.close()
        print(f"Saved Heatmap plot for dimensions {d1} vs {d2} to {plt_path}")

if __name__ == '__main__':
    main()