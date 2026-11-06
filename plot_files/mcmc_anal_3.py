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

    # ==========================================
    # SCATTER PLOTS & CORRELATIONS (AR vs ESS & Cross-AR)
    # ==========================================
    print("\nGenerating AR vs ESS and Cross-Latent AR scatter plots...")
    fig_corr, axes_corr = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # ROW 1: AR vs ESS per latent variable
    for j in range(3):
        ax = axes_corr[0, j]
        x_val = all_ar[:, j] * 100 
        y_val = all_ess[:, j]
        
        ax.scatter(x_val, y_val, alpha=0.6, color=colors[j], edgecolors='w', linewidth=0.5)
        
        # Calculate correlation and trendline
        corr_val = np.corrcoef(x_val, y_val)[0, 1]
        m, b = np.polyfit(x_val, y_val, 1)
        ax.plot(x_val, m * x_val + b, color='black', linestyle='--', alpha=0.8)
        
        ax.set_title(f"{latent_names[j]}\nAR vs ESS (Corr: {corr_val:.2f})", fontsize=13, fontweight='bold')
        ax.set_xlabel("Acceptance Rate (%)")
        ax.set_ylabel("ESS")
        ax.grid(True, alpha=0.3)

    # ROW 2: Cross-Latent Acceptance Rates
    pairs = [(0, 1), (0, 2), (1, 2)]
    short_names = ['Asymptotic', 'Inflexion', 'Tau']
    
    for idx, (j1, j2) in enumerate(pairs):
        ax = axes_corr[1, idx]
        x_val = all_ar[:, j1] * 100
        y_val = all_ar[:, j2] * 100
        
        ax.scatter(x_val, y_val, alpha=0.6, color='tab:purple', edgecolors='w', linewidth=0.5)
        
        # Calculate correlation and trendline
        corr_val = np.corrcoef(x_val, y_val)[0, 1]
        m, b = np.polyfit(x_val, y_val, 1)
        ax.plot(x_val, m * x_val + b, color='black', linestyle='--', alpha=0.8)
        
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
    analyze_all_individuals_metrics()