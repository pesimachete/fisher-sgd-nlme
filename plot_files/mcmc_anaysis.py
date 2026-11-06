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

def analyze_all_individuals_metrics(ex_res_path="example_res.npy", tail=20000):
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
    # PRINT SUMMARY INFO TO CONSOLE
    # ==========================================
    print("\n" + "="*60)
    print(f"POPULATION SUMMARY METRICS (N = {n_total})")
    print("="*60)
    
    for j in range(3):
        print(f"\n--- {latent_names[j]} ---")
        print(f"  Acceptance Rate : Mean = {np.mean(all_ar[:, j]):.1%} | Min = {np.min(all_ar[:, j]):.1%} | Max = {np.max(all_ar[:, j]):.1%}")
        print(f"  ESS             : Mean = {np.mean(all_ess[:, j]):.1f} | Min = {np.min(all_ess[:, j]):.1f} | Max = {np.max(all_ess[:, j]):.1f}")
        
        # Identify problematic individuals (e.g., AR < 5% or ESS < 100)
        poor_mixing = np.where((all_ar[:, j] < 0.05) | (all_ess[:, j] < 100))[0]
        if len(poor_mixing) > 0:
            print(f"  [!] Warning: Individuals with AR < 5% or ESS < 100: {poor_mixing.tolist()}")

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

if __name__ == "__main__":
    analyze_all_individuals_metrics()