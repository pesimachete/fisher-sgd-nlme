import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

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

def analyze_all_individuals_metrics(ex_res_path="example_res.npy", tail=40000):
    os.makedirs('plots', exist_ok=True)
    
    print(f"Loading data from {ex_res_path}...")
    with open(ex_res_path, 'rb') as f:
        ex_res = pickle.load(f)
        
    if not hasattr(ex_res, 'evol_z'):
        print("Error: 'evol_z' not found in example_res.npy.")
        return

    # Extract the last `tail` iterations
    evol_z_tail = ex_res.evol_z[-tail:, :, :]
    actual_tail = evol_z_tail.shape[0]
    n_total = evol_z_tail.shape[1]
    
    latent_names = ['Asymptotic (z_0)', 'Inflexion (z_1)', 'Tau (z_2)']
    
    # Arrays to store the metrics for all individuals
    all_ess = np.zeros((n_total, 3))
    all_ar = np.zeros((n_total, 3))
    
    print(f"\nCalculating metrics for all {n_total} individuals (Last {actual_tail} iterations)...")
    
    for i in range(n_total):
        for j in range(3):
            chain = evol_z_tail[:, i, j]
            
            # Acceptance Rate
            jumps = np.sum(chain[1:] != chain[:-1])
            all_ar[i, j] = jumps / (len(chain) - 1)
            
            # ESS
            all_ess[i, j] = compute_ess(chain)

    # ==========================================
    # EXTREMA ANALYSIS (Top & Bottom 5%)
    # ==========================================
    p_bottom, p_top = 5, 95
    
    worst_ar_indices, best_ar_indices = [], []
    worst_ess_indices, best_ess_indices = [], []
    
    for j in range(3):
        # Thresholds
        thresh_ar_bot = np.percentile(all_ar[:, j], p_bottom)
        thresh_ar_top = np.percentile(all_ar[:, j], p_top)
        thresh_ess_bot = np.percentile(all_ess[:, j], p_bottom)
        thresh_ess_top = np.percentile(all_ess[:, j], p_top)
        
        # Collect indices per latent variable
        worst_ar_indices.append(set(np.where(all_ar[:, j] <= thresh_ar_bot)[0]))
        best_ar_indices.append(set(np.where(all_ar[:, j] >= thresh_ar_top)[0]))
        
        worst_ess_indices.append(set(np.where(all_ess[:, j] <= thresh_ess_bot)[0]))
        best_ess_indices.append(set(np.where(all_ess[:, j] >= thresh_ess_top)[0]))

    # Find intersections: Individuals at the extremes across ALL latent variables
    inter_worst_ar = worst_ar_indices[0].intersection(worst_ar_indices[1], worst_ar_indices[2])
    inter_best_ar = best_ar_indices[0].intersection(best_ar_indices[1], best_ar_indices[2])
    
    inter_worst_ess = worst_ess_indices[0].intersection(worst_ess_indices[1], worst_ess_indices[2])
    inter_best_ess = best_ess_indices[0].intersection(best_ess_indices[1], best_ess_indices[2])
    
    # Combine unique extreme individuals for plotting
    bottom_plot_inds = set().union(*worst_ar_indices, *worst_ess_indices)
    top_plot_inds = set().union(*best_ar_indices, *best_ess_indices)
    
    # ==========================================
    # PRINT SUMMARY INFO TO CONSOLE
    # ==========================================
    print("\n" + "="*60)
    print(f"POPULATION SUMMARY METRICS (N = {n_total})")
    print("="*60)
    
    for j in range(3):
        print(f"\n--- {latent_names[j]} ---")
        print(f"  Acceptance Rate : Mean = {np.mean(all_ar[:, j]):.1%} | Min = {np.min(all_ar[:, j]):.1%} | Max = {np.max(all_ar[:, j]):.1%}")
        print(f"  ESS             : Mean = {np.mean(all_ess[:, j]):.1f} | Min = {np.min(all_ess[:, j]):.1f} | Max = {np.max(all_ess[:, j]):.1f}")
        
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

    # ==========================================
    # PLOT POPULATION DISTRIBUTIONS
    # ==========================================
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for j in range(3):
        # ROW 1: Acceptance Rate Histograms
        ax_ar = axes[0, j]
        ax_ar.hist(all_ar[:, j] * 100, bins=20, color=colors[j], alpha=0.7, edgecolor='black')
        ax_ar.axvline(np.mean(all_ar[:, j]) * 100, color='red', linestyle='--', label=f'Mean: {np.mean(all_ar[:, j]):.1%}')
        ax_ar.set_title(f"Acceptance Rate: {latent_names[j]}", fontsize=13, fontweight='bold')
        ax_ar.set_xlabel("Acceptance Rate (%)")
        ax_ar.set_ylabel("Number of Individuals")
        ax_ar.grid(True, alpha=0.3)
        ax_ar.legend()

        # ROW 2: ESS Histograms
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
    print(f"\nPopulation metric plots successfully generated and saved to '{output_filename}'!")

    # ==========================================
    # PLOT EXTREME CHAINS TRACES (Top vs Bottom)
    # ==========================================
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

if __name__ == "__main__":
    analyze_all_individuals_metrics()