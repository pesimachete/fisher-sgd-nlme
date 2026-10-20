import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def plot_z_population_summary(ex_res_path="example_res.npy", true_z_path="true_z.npy", limit=100000):
    os.makedirs('plots', exist_ok=True)
    
    print("Loading data...")
    # Load MCMC results
    with open(ex_res_path, 'rb') as f:
        ex_res = pickle.load(f)
        
    # Load true z values
    try:
        true_z = np.load(true_z_path)
    except FileNotFoundError:
        print(f"Error: {true_z_path} not found. Cannot plot True vs MAP without it.")
        return

    # Extract z history and factor
    if not hasattr(ex_res, 'evol_z'):
        print("Error: 'evol_z' not found in example_res.npy. Please ensure you saved the z history.")
        return
        
    evol_z = ex_res.evol_z[:limit]  # Shape: (iterations, n_individuals, 3)
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
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    
    # ==========================================
    # ROW 1: Population Distributions (Histograms)
    # ==========================================
    for j in range(3):
        ax = axes[0, j]
        
        # True distribution
        ax.hist(true_z[:, j], bins=20, alpha=0.5, label='True Z Population', 
                color='gray', edgecolor='black', density=True)
        # Estimated (MAP) distribution
        ax.hist(map_z[:, j], bins=20, alpha=0.6, label='Estimated MAP Population', 
                color=colors[j], edgecolor='black', density=True)
                
        ax.set_title(f"Population Distribution:\n{latent_names[j]}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Density", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # ==========================================
    # ROW 2: True vs. Estimated (Scatter)
    # ==========================================
    for j in range(3):
        ax = axes[1, j]
        
        # Scatter individual points
        ax.scatter(true_z[:, j], map_z[:, j], color=colors[j], alpha=0.7, edgecolors='k')
        
        # 1-to-1 reference line (y=x)
        min_val = min(np.min(true_z[:, j]), np.min(map_z[:, j]))
        max_val = max(np.max(true_z[:, j]), np.max(map_z[:, j]))
        padding = (max_val - min_val) * 0.05
        
        ax.plot([min_val - padding, max_val + padding], 
                [min_val - padding, max_val + padding], 
                'r--', linewidth=2, label='Perfect Estimation (y=x)')
                
        ax.set_title(f"True vs Estimated:\n{latent_names[j]}", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"True {latent_names[j]}", fontsize=12)
        ax.set_ylabel(f"Estimated MAP {latent_names[j]}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Latent Variables ($z$) Population Analysis ({n_individuals} Individuals)", fontsize=18, y=1.02)
    fig.tight_layout()
    
    output_filename = 'plots/population_z_analysis.pdf'
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Population Z plots successfully generated and saved to '{output_filename}'!")

if __name__ == "__main__":
    plot_z_population_summary()