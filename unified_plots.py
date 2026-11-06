import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import jax.numpy as jnp
from scipy.special import expit
from scipy.stats import gaussian_kde
import model

# Safely attempt to import Plotly for interactive plots
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Notice: 'plotly' is not installed. Skipping interactive plot generation. Install via 'pip install plotly'.")

def vect_params(p):
    il1 = jnp.tril_indices(3)
    return jnp.concatenate((
        jnp.array([p.asymptotic, p.inflexion, p.tau]), 
        p.cov_latent[il1], 
        jnp.array([p.var_residual])
    ))

def compute_ess(x):
    """
    Computes the Effective Sample Size (ESS) using the heuristic 
    of summing autocorrelations until the first negative value.
    """
    n = len(x)
    var = np.var(x)
    if var == 0:
        return 0.0

    x_centered = x - np.mean(x)
    fft_x = np.fft.fft(x_centered, n=2*n)
    acf_full = np.fft.ifft(fft_x * np.conj(fft_x))[:n].real
    acf_full = acf_full / acf_full[0]
    
    neg_idx = np.where(acf_full < 0)[0]
    k = neg_idx[0] if len(neg_idx) > 0 else n
    ess = n / (1 + 2 * np.sum(acf_full[1:k]))
    
    return ess

def generate_all_plots(ex_res_path="example_res.npy", true_z_path="true_z.npy", limit=200000, num_individuals=5):
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)

    # ==========================================
    # 1. SETUP & DATA LOADING
    # ==========================================
    theta0 = model.parametrization.params_to_reals1d(
        asymptotic=200, inflexion=500, tau=150,
        cov_latent=jnp.diag(jnp.array([40, 100, 100])),
        var_residual=1,
    )
    p0 = model.parametrization.reals1d_to_params(theta0)
    p0_vec = vect_params(p0)

    subtitles_params = [
        r'$\beta_1$', r'$\beta_2$', r'$\alpha$', 
        r'$\Gamma_{11}$', r'$\Gamma_{12}$', r'$\Gamma_{22}$', 
        r'$\Gamma_{13}$', r'$\Gamma_{23}$', r'$\Gamma_{33}$', 
        r'$\sigma^2$'
    ]

    print("Loading data for generate_all_plots...")
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
        plot_data = evol_p[:, i].copy()
        true_val = p0_vec[i]
        
        # Log scale for Variances
        if i in [3, 5, 8, 9]:
            axes_params[i].set_yscale('log')
            
        # Calculate True Correlation for Off-Diagonals
        elif i == 4: # Gamma_12
            plot_data = plot_data / np.sqrt(evol_p[:, 3] * evol_p[:, 5])
            true_val = true_val / np.sqrt(p0_vec[3] * p0_vec[5])
            axes_params[i].set_ylabel("Correlation", fontsize=10)
        elif i == 6: # Gamma_13
            plot_data = plot_data / np.sqrt(evol_p[:, 3] * evol_p[:, 8])
            true_val = true_val / np.sqrt(p0_vec[3] * p0_vec[8])
            axes_params[i].set_ylabel("Correlation", fontsize=10)
        elif i == 7: # Gamma_23
            plot_data = plot_data / np.sqrt(evol_p[:, 5] * evol_p[:, 8])
            true_val = true_val / np.sqrt(p0_vec[5] * p0_vec[8])
            axes_params[i].set_ylabel("Correlation", fontsize=10)

        axes_params[i].axhline(true_val, color='y', linewidth=2, label='True Value' if i==0 else "")
        axes_params[i].plot(plot_data, color='tab:blue', linewidth=1.5, alpha=0.9)
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
        plot_data = evol_p[:, i].copy()
        true_val = p0_vec[i]
        
        is_log_scale = i in [3, 5, 8, 9]

        if i == 4:
            plot_data = plot_data / np.sqrt(evol_p[:, 3] * evol_p[:, 5])
            true_val = true_val / np.sqrt(p0_vec[3] * p0_vec[5])
        elif i == 6:
            plot_data = plot_data / np.sqrt(evol_p[:, 3] * evol_p[:, 8])
            true_val = true_val / np.sqrt(p0_vec[3] * p0_vec[8])
        elif i == 7:
            plot_data = plot_data / np.sqrt(evol_p[:, 5] * evol_p[:, 8])
            true_val = true_val / np.sqrt(p0_vec[5] * p0_vec[8])
            
        valid_samples = plot_data[burn_in_idx:]
        
        # Left: Trace
        ax_trace = axes_mcmc[i, 0]
        ax_trace.plot(plot_data, color='tab:blue', alpha=0.7, linewidth=1)
        ax_trace.axhline(true_val, color='tab:orange', linewidth=2, linestyle='--', label='True Value')
        ax_trace.axvline(burn_in_idx, color='red', linestyle=':', label='End Burn-in')
        ax_trace.set_title(f"{subtitles_params[i]} Trace", fontsize=14, fontweight='bold')
        ax_trace.set_ylabel("Value", fontsize=11)
        ax_trace.grid(True, alpha=0.3)
        if is_log_scale:
            ax_trace.set_yscale('log')
        if i == 0: ax_trace.legend(loc="upper right")
            
        # Right: Density
        ax_dens = axes_mcmc[i, 1]
        if is_log_scale:
            min_val, max_val = np.min(valid_samples), np.max(valid_samples)
            bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
            ax_dens.set_xscale('log')
        else:
            bins = 50
            
        ax_dens.hist(valid_samples, bins=bins, density=True, color='tab:blue', alpha=0.6, edgecolor='black', linewidth=0.5)
        ax_dens.axvline(true_val, color='tab:orange', linewidth=2, linestyle='--', label='True Value')
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
        
        fig_z, axes_z = plt.subplots(nrows=n_plot, ncols=6, figsize=(24, 3 * n_plot))
        if n_plot == 1: axes_z = np.expand_dims(axes_z, axis=0)

        for i in range(n_plot):
            for j in range(3):
                ax_trace = axes_z[i, j*2]
                ax_dens = axes_z[i, j*2 + 1]
                chain = evol_z[:, i, j]
                valid_samples = chain[burn_in_idx:]
                
                ax_trace.plot(chain, color='tab:blue', alpha=0.7, linewidth=0.8)
                ax_trace.axvline(burn_in_idx, color='red', linestyle=':', alpha=0.8, label='End Burn-in')
                
                actual_val = None
                if true_z is not None:
                    actual_val = true_z[i, j]
                    ax_trace.axhline(actual_val, color='black', linewidth=2, linestyle='--', label='True z')

                if i == 0:
                    ax_trace.set_title(f"{latent_names[j]} Trace", fontsize=13, fontweight='bold')
                    if j == 2: ax_trace.legend(loc='upper right')
                if j == 0: ax_trace.set_ylabel(f"Individual {i}", fontsize=11, fontweight='bold')
                ax_trace.grid(True, alpha=0.3)
                
                ax_dens.hist(valid_samples, bins=50, density=True, color='gold', alpha=0.7, edgecolor='goldenrod')
                if actual_val is not None:
                    ax_dens.axvline(actual_val, color='black', linewidth=2, linestyle='--', label='True z')
                ax_dens.axvline(np.mean(valid_samples), color='green', linewidth=1.5, linestyle='-.', label='Mean')

                if i == 0:
                    ax_dens.set_title(f"{latent_names[j]} Density", fontsize=13, fontweight='bold')
                    if j == 2: ax_dens.legend(loc='upper right')
                ax_dens.grid(True, alpha=0.3)

        for j in range(6):
            axes_z[-1, j].set_xlabel("Iterations" if j%2==0 else "Value", fontsize=12)

        fig_z.suptitle(f"MCMC Chains for Latent Variables $z$ (First {n_plot} Individuals)", fontsize=16, y=1.02)
        fig_z.tight_layout()
        fig_z.savefig('plots/evol_individuals_z.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig_z)
        
        # ==========================================
        # 7. INTERACTIVE PLOT (Z Chains)
        # ==========================================
        if HAS_PLOTLY:
            print("Generating Interactive Z Chains Plot...")
            fig_inter = make_subplots(rows=n_plot, cols=3, 
                                      subplot_titles=[f"Ind {i} - {latent_names[j]}" for i in range(n_plot) for j in range(3)])
            
            step = max(1, len(evol_z) // 5000)
            for i in range(n_plot):
                for j in range(3):
                    chain = evol_z[::step, i, j]
                    x_vals = np.arange(0, len(evol_z), step)
                    fig_inter.add_trace(
                        go.Scatter(x=x_vals, y=chain, mode='lines', name=f"Ind {i} {latent_names[j]}",
                                   opacity=0.7, line=dict(width=1)),
                        row=i+1, col=j+1
                    )
                    if true_z is not None:
                        fig_inter.add_trace(
                            go.Scatter(x=[0, len(evol_z)], y=[true_z[i, j], true_z[i, j]],
                                       mode='lines', name=f"True Z (Ind {i})", line=dict(color='black', dash='dash')),
                            row=i+1, col=j+1
                        )
            fig_inter.update_layout(height=300*n_plot, title_text="Interactive Z Chains (Subsampled)")
            fig_inter.write_html("plots/evol_individuals_z_interactive.html")
            
    else:
        print("Skipping Z chains plot: 'evol_z' not found in example_res.npy")


def plot_z_population_summary(ex_res_path="example_res.npy", true_z_path="true_z.npy", limit=100000):
    os.makedirs('plots', exist_ok=True)
    
    print("Loading data for plot_z_population_summary...")
    with open(ex_res_path, 'rb') as f:
        ex_res = pickle.load(f)
        
    try:
        true_z = np.load(true_z_path)
    except FileNotFoundError:
        print(f"Error: {true_z_path} not found. Cannot plot True vs MAP without it.")
        return

    if not hasattr(ex_res, 'evol_z'):
        print("Error: 'evol_z' not found in example_res.npy.")
        return
        
    evol_z = ex_res.evol_z[:limit]
    factor = ex_res.factor[:limit]
    n_individuals = evol_z.shape[1]

    factor1 = np.where(factor == 1)[0]
    burn_in_idx = factor1[-1] if len(factor1) > 0 else int(limit * 0.2)

    print(f"Calculating Posterior Mean (MAP) from index {burn_in_idx}...")
    valid_z_samples = evol_z[burn_in_idx:, :, :]
    map_z = np.mean(valid_z_samples, axis=0)

    latent_names = ['Asymptotic (z_0)', 'Inflexion (z_1)', 'Tau (z_2)']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    
    for j in range(3):
        ax = axes[0, j]
        min_val = min(np.min(true_z[:, j]), np.min(map_z[:, j]))
        max_val = max(np.max(true_z[:, j]), np.max(map_z[:, j]))
        padding = (max_val - min_val) * 0.2
        x_grid = np.linspace(min_val - padding, max_val + padding, 1000)
        
        kde_true = gaussian_kde(true_z[:, j], bw_method=0.15)
        ax.plot(x_grid, kde_true(x_grid), color='gray', label='True Z Population', linewidth=1)
        ax.fill_between(x_grid, kde_true(x_grid), alpha=0.25, color='gray')
        
        kde_map = gaussian_kde(map_z[:, j], bw_method=0.15)
        ax.plot(x_grid, kde_map(x_grid), color=colors[j], label='Estimated MAP Population', linewidth=1)
        ax.fill_between(x_grid, kde_map(x_grid), alpha=0.25, color=colors[j])
                
        ax.set_title(f"Population Distribution:{latent_names[j]}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Density", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for j in range(3):
        ax = axes[1, j]
        ax.scatter(true_z[:, j], map_z[:, j], color=colors[j], alpha=0.7, edgecolors='k')
        
        min_val = min(np.min(true_z[:, j]), np.min(map_z[:, j]))
        max_val = max(np.max(true_z[:, j]), np.max(map_z[:, j]))
        padding = (max_val - min_val) * 0.05
        
        ax.plot([min_val - padding, max_val + padding], 
                [min_val - padding, max_val + padding], 
                'r--', linewidth=2, label='Perfect Estimation (y=x)')
                
        ax.set_title(f"True vs Estimated:{latent_names[j]}", fontsize=14, fontweight='bold')
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

def analyze_all_individuals_metrics(ex_res_path="example_res.npy", tail=40000):
    os.makedirs('plots', exist_ok=True)
    
    print(f"Loading data from {ex_res_path} for MCMC metrics...")
    with open(ex_res_path, 'rb') as f:
        ex_res = pickle.load(f)
        
    if not hasattr(ex_res, 'evol_z'):
        print("Error: 'evol_z' not found in example_res.npy.")
        return

    evol_z_tail = ex_res.evol_z[-tail:, :, :]
    actual_tail = evol_z_tail.shape[0]
    n_total = evol_z_tail.shape[1]
    latent_names = ['Asymptotic (z_0)', 'Inflexion (z_1)', 'Tau (z_2)']
    
    all_ess = np.zeros((n_total, 3))
    all_ar = np.zeros((n_total, 3))
    
    print(f"Calculating metrics for all {n_total} individuals (Last {actual_tail} iterations)...")
    for i in range(n_total):
        for j in range(3):
            chain = evol_z_tail[:, i, j]
            jumps = np.sum(chain[1:] != chain[:-1])
            all_ar[i, j] = jumps / (len(chain) - 1)
            all_ess[i, j] = compute_ess(chain)

    p_bottom, p_top = 5, 95
    worst_ar_indices, best_ar_indices = [], []
    worst_ess_indices, best_ess_indices = [], []
    
    for j in range(3):
        thresh_ar_bot = np.percentile(all_ar[:, j], p_bottom)
        thresh_ar_top = np.percentile(all_ar[:, j], p_top)
        thresh_ess_bot = np.percentile(all_ess[:, j], p_bottom)
        thresh_ess_top = np.percentile(all_ess[:, j], p_top)
        
        worst_ar_indices.append(set(np.where(all_ar[:, j] <= thresh_ar_bot)[0]))
        best_ar_indices.append(set(np.where(all_ar[:, j] >= thresh_ar_top)[0]))
        worst_ess_indices.append(set(np.where(all_ess[:, j] <= thresh_ess_bot)[0]))
        best_ess_indices.append(set(np.where(all_ess[:, j] >= thresh_ess_top)[0]))

    inter_worst_ar = worst_ar_indices[0].intersection(worst_ar_indices[1], worst_ar_indices[2])
    inter_best_ar = best_ar_indices[0].intersection(best_ar_indices[1], best_ar_indices[2])
    inter_worst_ess = worst_ess_indices[0].intersection(worst_ess_indices[1], worst_ess_indices[2])
    inter_best_ess = best_ess_indices[0].intersection(best_ess_indices[1], best_ess_indices[2])
    
    bottom_plot_inds = set().union(*worst_ar_indices, *worst_ess_indices)
    top_plot_inds = set().union(*best_ar_indices, *best_ess_indices)
    
    print("\n" + "="*60)
    print(f"POPULATION SUMMARY METRICS (N = {n_total})")
    print("="*60)
    
    for j in range(3):
        print(f"\n--- {latent_names[j]} ---")
        print(f"  Acceptance Rate : Mean = {np.mean(all_ar[:, j]):.1%} | Min = {np.min(all_ar[:, j]):.1%} | Max = {np.max(all_ar[:, j]):.1%}")
        print(f"  ESS             : Mean = {np.mean(all_ess[:, j]):.1f} | Min = {np.min(all_ess[:, j]):.1f} | Max = {np.max(all_ess[:, j]):.1f}")
        
        poor_mixing = np.where((all_ar[:, j] < 0.05) | (all_ess[:, j] < 100))[0]
        if len(poor_mixing) > 0:
            print(f"  [!] Warning: Individuals with AR < 5% or ESS < 100: {poor_mixing.tolist()}")

    print("\n" + "="*60)
    print("EXTREMA ANALYSIS (Top & Bottom 5%)")
    print("="*60)
    print(f"Lowest AR across ALL variables: {sorted(list(inter_worst_ar)) if inter_worst_ar else 'None'}")
    print(f"Highest AR across ALL variables: {sorted(list(inter_best_ar)) if inter_best_ar else 'None'}")
    print(f"Lowest ESS across ALL variables: {sorted(list(inter_worst_ess)) if inter_worst_ess else 'None'}")
    print(f"Highest ESS across ALL variables: {sorted(list(inter_best_ess)) if inter_best_ess else 'None'}")
    
    def print_extreme_vals(inds, label):
        if not inds: return
        print(f"\nValues for {label} individuals:")
        for idx in sorted(list(inds))[:10]:
            print(f"  Individual {idx}:")
            for j in range(3):
                print(f"    {latent_names[j][:10]}... | AR: {all_ar[idx, j]:.2%} | ESS: {all_ess[idx, j]:.1f}")

    print_extreme_vals(bottom_plot_inds, "BOTTOM (Worst)")
    print_extreme_vals(top_plot_inds, "TOP (Best)")

    # 1. Population Distributions (Histograms)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for j in range(3):
        ax_ar = axes[0, j]
        ax_ar.hist(all_ar[:, j] * 100, bins=20, color=colors[j], alpha=0.7, edgecolor='black')
        ax_ar.axvline(np.mean(all_ar[:, j]) * 100, color='red', linestyle='--', label=f'Mean: {np.mean(all_ar[:, j]):.1%}')
        ax_ar.set_title(f"Acceptance Rate: {latent_names[j]}", fontsize=13, fontweight='bold')
        ax_ar.set_xlabel("Acceptance Rate (%)")
        ax_ar.set_ylabel("Number of Individuals")
        ax_ar.grid(True, alpha=0.3)
        ax_ar.legend()

        ax_ess = axes[1, j]
        ax_ess.hist(all_ess[:, j], bins=20, color=colors[j], alpha=0.7, edgecolor='black')
        ax_ess.axvline(np.mean(all_ess[:, j]), color='red', linestyle='--', label=f'Mean: {np.mean(all_ess[:, j]):.0f}')
        ax_ess.set_title(f"Effective Sample Size: {latent_names[j]}", fontsize=13, fontweight='bold')
        ax_ess.set_xlabel("ESS")
        ax_ess.set_ylabel("Number of Individuals")
        ax_ess.grid(True, alpha=0.3)
        ax_ess.legend()

    fig.suptitle(f"MCMC Chain Health Across All {n_total} Individuals (Last {actual_tail} Iterations)", fontsize=16, y=1.02)
    fig.tight_layout()
    output_filename = 'plots/population_acf_ar_ess_summary.pdf'
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Population metric plots successfully generated and saved to '{output_filename}'!")

    # 2. Extreme Traces
    def plot_extreme_subset(inds, title_prefix, filename, color):
        if not inds: return
        plot_list = sorted(list(inds))[:5] 
        n_plot = len(plot_list)
        
        fig_ext, axes_ext = plt.subplots(nrows=n_plot, ncols=3, figsize=(15, 3 * n_plot), sharex='col')
        if n_plot == 1: axes_ext = np.expand_dims(axes_ext, axis=0)

        for row_idx, ind_idx in enumerate(plot_list):
            for j in range(3):
                ax = axes_ext[row_idx, j]
                chain = evol_z_tail[:, ind_idx, j]
                ax.plot(chain, color=color, alpha=0.8, linewidth=1)
                
                if row_idx == 0:
                    ax.set_title(f"{latent_names[j]}", fontsize=13, fontweight='bold')
                if j == 0: 
                    ax.set_ylabel(f"Ind {ind_idx}", fontsize=11, fontweight='bold')
                
                ax.text(0.95, 0.05, f"AR: {all_ar[ind_idx, j]:.1%}\nESS: {all_ess[ind_idx, j]:.0f}", 
                        transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.grid(True, alpha=0.3)

        for j in range(3):
            axes_ext[-1, j].set_xlabel("Iterations (Tail)", fontsize=12)

        fig_ext.suptitle(f"MCMC Traces: {title_prefix}", fontsize=16, y=1.02)
        fig_ext.tight_layout()
        fig_ext.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig_ext)
        print(f"Traces saved to '{filename}'.")

    plot_extreme_subset(bottom_plot_inds, "Worst 5% Individuals (Poor Mixing)", 'plots/evol_individuals_z_bottom_extremes.pdf', 'tab:red')
    plot_extreme_subset(top_plot_inds, "Top 5% Individuals (Best Mixing)", 'plots/evol_individuals_z_top_extremes.pdf', 'tab:green')

    # 3. Scatter Plots & Correlations (AR vs ESS & Cross-AR)
    print("Generating AR vs ESS and Cross-Latent AR scatter plots...")
    fig_corr, axes_corr = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
    
    for j in range(3):
        ax = axes_corr[0, j]
        x_val = all_ar[:, j] * 100 
        y_val = all_ess[:, j]
        ax.scatter(x_val, y_val, alpha=0.6, color=colors[j], edgecolors='w', linewidth=0.5)
        
        if np.std(x_val) > 0 and np.std(y_val) > 0:
            corr_val = np.corrcoef(x_val, y_val)[0, 1]
            m, b = np.polyfit(x_val, y_val, 1)
            ax.plot(x_val, m * x_val + b, color='black', linestyle='--', alpha=0.8)
        else:
            corr_val = 0.0
            
        ax.set_title(f"{latent_names[j]}\nAR vs ESS (Corr: {corr_val:.2f})", fontsize=13, fontweight='bold')
        ax.set_xlabel("Acceptance Rate (%)")
        ax.set_ylabel("ESS")
        ax.grid(True, alpha=0.3)

    pairs = [(0, 1), (0, 2), (1, 2)]
    short_names = ['Asymptotic', 'Inflexion', 'Tau']
    
    for idx, (j1, j2) in enumerate(pairs):
        ax = axes_corr[1, idx]
        x_val = all_ar[:, j1] * 100
        y_val = all_ar[:, j2] * 100
        ax.scatter(x_val, y_val, alpha=0.6, color='tab:purple', edgecolors='w', linewidth=0.5)
        
        if np.std(x_val) > 0 and np.std(y_val) > 0:
            corr_val = np.corrcoef(x_val, y_val)[0, 1]
            m, b = np.polyfit(x_val, y_val, 1)
            ax.plot(x_val, m * x_val + b, color='black', linestyle='--', alpha=0.8)
        else:
            corr_val = 0.0
            
        ax.set_title(f"AR: {short_names[j1]} vs {short_names[j2]}\n(Corr: {corr_val:.2f})", fontsize=13, fontweight='bold')
        ax.set_xlabel(f"{short_names[j1]} AR (%)")
        ax.set_ylabel(f"{short_names[j2]} AR (%)")
        ax.grid(True, alpha=0.3)

    fig_corr.suptitle(f"Correlations: AR vs ESS and Cross-Latent AR (N = {n_total})", fontsize=16, y=1.02)
    fig_corr.tight_layout()
    corr_filename = 'plots/population_correlations.pdf'
    fig_corr.savefig(corr_filename, dpi=300, bbox_inches='tight')
    plt.close(fig_corr)
    print(f"Correlation plots successfully generated and saved to '{corr_filename}'!")

if __name__ == "__main__":
    generate_all_plots()
    plot_z_population_summary()
    analyze_all_individuals_metrics()
