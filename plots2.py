import matplotlib.pyplot as plt
import numpy as np
import pickle
import model
import jax.numpy as jnp
import os

def vect_params(p):
    il1 = jnp.tril_indices(3)
    return jnp.concatenate((
        jnp.array([p.asymptotic, p.inflexion, p.tau]), 
        p.cov_latent[il1], 
        jnp.array([p.var_residual])
    ))

def generate_all_plots(ex_res_path="example_res.npy", true_z_path="true_z.npy", limit=100000, num_individuals=5):
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)

    # ==========================================
    # 1. SETUP & DATA LOADING
    # ==========================================
    theta0 = model.parametrization.params_to_reals1d(
        asymptotic=200, inflexion=500, tau=150,
        cov_latent=jnp.diag(jnp.array([40, 100, 100])),
        var_residual=100,
    )
    p0 = model.parametrization.reals1d_to_params(theta0)
    p0_vec = vect_params(p0)

    subtitles_params = [
        r'$\beta_1$', r'$\beta_2$', r'$\alpha$', 
        r'$\Gamma_{11}$', r'$\Gamma_{12}$', r'$\Gamma_{22}$', 
        r'$\Gamma_{13}$', r'$\Gamma_{23}$', r'$\Gamma_{33}$', 
        r'$\sigma^2$'
    ]

    print("Loading data...")
    with open(ex_res_path, 'rb') as f:
        ex_res = pickle.load(f)

    try:
        true_z = np.load(true_z_path)
    except FileNotFoundError:
        print(f"Warning: {true_z_path} not found. True z values will not be plotted.")
        true_z = None

    evol_theta_lim = ex_res.evol_theta[:limit, :]
    factor = ex_res.factor[:limit]
    evol_grad = ex_res.evol_grad[:limit, :] if hasattr(ex_res, 'evol_grad') else np.zeros_like(evol_theta_lim)
    evol_z = ex_res.evol_z[:limit] if hasattr(ex_res, 'evol_z') else None

    # Calculate parameter matrix
    evol_p = np.zeros(shape=(evol_theta_lim.shape[0], p0_vec.size))
    for i in range(evol_p.shape[0]):
        evol_p[i,:] = vect_params(model.parametrization.reals1d_to_params(evol_theta_lim[i,:]))

    # Heating and Burn-in markers
    factor1 = jnp.where(factor == 1)[0]
    if len(factor1) > 0:
        k1, k2 = factor1[0], factor1[-1]
        burn_in_idx = factor1[-1]
    else:
        k1 = k2 = burn_in_idx = int(limit * 0.2)

    # ==========================================
    # 2. FIGURE 1: Parameter Evolution
    # ==========================================
    print("Generating Parameter Evolution plot...")
    fig_params, axes_params = plt.subplots(nrows=5, ncols=2, sharex=True, figsize=(12, 16))
    axes_params = axes_params.ravel()
    
    for i in range(10):
        axes_params[i].axhline(p0_vec[i], color='y', linewidth=2, label='True Value' if i==0 else "")
        axes_params[i].plot(evol_p[:, i], color='tab:blue', linewidth=1.5, alpha=0.9)
        axes_params[i].axvline(k1, color='red', linestyle=':', label='Start Heating' if i==0 else "")
        axes_params[i].axvline(k2, color='green', linestyle=':', label='End Heating' if i==0 else "")
        axes_params[i].set_title(f"Parameter: {subtitles_params[i]}", fontsize=14)
        axes_params[i].grid(True, alpha=0.3)
    
    axes_params[8].set_xlabel("Iterations", fontsize=12)
    axes_params[9].set_xlabel("Iterations", fontsize=12)
    fig_params.legend(loc='upper right', bbox_to_anchor=(0.95, 0.98), ncol=3)
    fig_params.tight_layout(rect=[0, 0, 1, 0.95])
    fig_params.savefig('plots/evol_parameters_detailed.pdf', dpi=300)
    plt.close(fig_params)

    # ==========================================
    # 3. FIGURE 2: Gradient Evolution
    # ==========================================
    print("Generating Gradients plot...")
    fig_grads, axes_grads = plt.subplots(nrows=5, ncols=2, sharex=True, figsize=(12, 16))
    axes_grads = axes_grads.ravel()
    
    for i in range(10):
        axes_grads[i].plot(evol_grad[:, i], color='tab:orange', linewidth=1.2, alpha=0.8)
        axes_grads[i].axhline(0, color='black', linestyle='--', linewidth=1.5)
        axes_grads[i].axvline(k1, color='red', linestyle=':', alpha=0.7)
        axes_grads[i].axvline(k2, color='green', linestyle=':', alpha=0.7)
        axes_grads[i].set_title(f"Gradient of {subtitles_params[i]}", fontsize=14)
        axes_grads[i].grid(True, alpha=0.3)
        
    axes_grads[8].set_xlabel("Iterations", fontsize=12)
    axes_grads[9].set_xlabel("Iterations", fontsize=12)
    fig_grads.tight_layout()
    fig_grads.savefig('plots/evol_gradients_detailed.pdf', dpi=300)
    plt.close(fig_grads)

    # ==========================================
    # 4. FIGURE 3: Learning Rate (Factor)
    # ==========================================
    print("Generating Learning Rate plot...")
    fig_lr, ax_lr = plt.subplots(figsize=(14, 5))
    ax_lr.plot(factor, color='purple', linewidth=2)
    ax_lr.axvline(k1, color='red', linestyle=':', alpha=0.7, label='Start Heating')
    ax_lr.axvline(k2, color='green', linestyle=':', alpha=0.7, label='End Heating')
    ax_lr.set_yscale('log')
    ax_lr.set_title("Learning Rate Evolution (Factor)", fontsize=16)
    ax_lr.set_xlabel("Iterations", fontsize=12)
    ax_lr.set_ylabel("Learning Rate", fontsize=12)
    ax_lr.grid(True, which="both", alpha=0.3)
    ax_lr.legend()
    
    fig_lr.tight_layout()
    fig_lr.savefig('plots/evol_learning_rate_detailed.pdf', dpi=300)
    plt.close(fig_lr)

    # ==========================================
    # 5. FIGURE 4: MCMC Parameter Chains
    # ==========================================
    print("Generating MCMC Parameter Chains plot...")
    fig_mcmc, axes_mcmc = plt.subplots(nrows=10, ncols=2, figsize=(15, 25))
    
    for i in range(10):
        # Left: Trace
        ax_trace = axes_mcmc[i, 0]
        ax_trace.plot(evol_p[:, i], color='tab:blue', alpha=0.7, linewidth=1)
        ax_trace.axhline(p0_vec[i], color='tab:orange', linewidth=2, linestyle='--', label='True Value')
        ax_trace.axvline(burn_in_idx, color='red', linestyle=':', label='End Burn-in')
        ax_trace.set_title(f"{subtitles_params[i]} Trace", fontsize=14, fontweight='bold')
        ax_trace.set_ylabel("Value", fontsize=11)
        ax_trace.grid(True, alpha=0.3)
        if i == 0: ax_trace.legend(loc="upper right")
            
        # Right: Density
        ax_dens = axes_mcmc[i, 1]
        valid_samples = evol_p[burn_in_idx:, i]
        ax_dens.hist(valid_samples, bins=50, density=True, color='tab:blue', alpha=0.6, edgecolor='black', linewidth=0.5)
        ax_dens.axvline(p0_vec[i], color='tab:orange', linewidth=2, linestyle='--', label='True Value')
        ax_dens.axvline(np.mean(valid_samples), color='green', linewidth=2, label='Sample Mean')
        ax_dens.set_title(f"{subtitles_params[i]} Density (Post Burn-in)", fontsize=14, fontweight='bold')
        ax_dens.set_ylabel("Density", fontsize=11)
        ax_dens.grid(True, alpha=0.3)
        if i == 0: ax_dens.legend(loc="upper right")

    axes_mcmc[-1, 0].set_xlabel("Iterations", fontsize=12)
    axes_mcmc[-1, 1].set_xlabel("Parameter Value", fontsize=12)
    fig_mcmc.tight_layout(pad=2.0)
    fig_mcmc.savefig('plots/evol_mcmc_chains.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig_mcmc)

    # ==========================================
    # 6. FIGURE 5: Individual Z Chains
    # ==========================================
    if evol_z is not None:
        print("Generating MCMC Individual Z Chains plot...")
        n_total = evol_z.shape[1]
        n_plot = min(num_individuals, n_total)
        latent_names = ['Asymptotic (z_0)', 'Inflexion (z_1)', 'Tau (z_2)']
        
        fig_z, axes_z = plt.subplots(nrows=n_plot, ncols=3, figsize=(15, 3 * n_plot), sharex='col')
        if n_plot == 1: axes_z = np.expand_dims(axes_z, axis=0)

        for i in range(n_plot):
            for j in range(3):
                ax = axes_z[i, j]
                chain = evol_z[:, i, j]
                
                # Trace
                ax.plot(chain, color='tab:blue', alpha=0.7, linewidth=0.8)
                ax.axvline(burn_in_idx, color='red', linestyle=':', alpha=0.8, label='End Burn-in')
                
                # True Z
                actual_val = None
                if true_z is not None:
                    actual_val = true_z[i, j]
                    ax.axhline(actual_val, color='tab:green', linewidth=2, linestyle='--', label='True z')

                if i == 0:
                    ax.set_title(f"{latent_names[j]}", fontsize=13, fontweight='bold')
                    if j == 2: ax.legend(loc='upper right')
                if j == 0: ax.set_ylabel(f"Individual {i}", fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Density right axis
                ax_dens = ax.twiny()
                valid_samples = chain[burn_in_idx:]
                ax_dens.hist(valid_samples, bins=50, density=True, color='tab:orange', alpha=0.4, orientation='horizontal')
                if actual_val is not None:
                    ax_dens.axhline(actual_val, color='tab:green', linewidth=2, linestyle='--')
                ax_dens.set_xticks([]) 

        for j in range(3):
            axes_z[-1, j].set_xlabel("Iterations", fontsize=12)

        fig_z.suptitle(f"MCMC Chains for Latent Variables $z$ (First {n_plot} Individuals)", fontsize=16, y=1.02)
        fig_z.tight_layout()
        fig_z.savefig('plots/evol_individuals_z.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig_z)
    else:
        print("Skipping Z chains plot: 'evol_z' not found in example_res.npy")

    print("All plots successfully generated and saved to the 'plots/' directory!")

if __name__ == "__main__":
    generate_all_plots()