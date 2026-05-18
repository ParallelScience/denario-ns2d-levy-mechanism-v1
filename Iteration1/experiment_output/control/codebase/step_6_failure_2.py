# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

plt.rcParams['text.usetex'] = False

def compute_pdf(data, num_bins=50):
    data = data[data > 0]
    if len(data) == 0:
        return None, None
    min_val = np.min(data)
    max_val = np.max(data)
    if min_val == max_val:
        return np.array([min_val]), np.array([1.0])
    bins = np.logspace(np.log10(min_val), np.log10(max_val), num_bins)
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    valid = hist > 0
    return bin_centers[valid], hist[valid]

def plot_conditional_autocorr(tau_lags, R_v_vortex, R_v_strain, timestamp):
    plt.figure(figsize=(8, 6))
    plt.plot(tau_lags, R_v_vortex, label='Vortex Regions', color='blue', linewidth=2)
    plt.plot(tau_lags, R_v_strain, label='Strain Regions', color='red', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Lag time tau')
    plt.ylabel('Velocity Autocorrelation R_v(tau)')
    plt.title('Conditional Lagrangian Velocity Autocorrelation')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    filepath = os.path.join('data', 'conditional_autocorr_1_' + timestamp + '.png')
    plt.savefig(filepath, dpi=300)
    plt.close()
    print('Plot saved to ' + filepath)

def plot_trapping_times(trapping_durations, mu_trap, x_min_trap, timestamp):
    plt.figure(figsize=(8, 6))
    x, y = compute_pdf(trapping_durations, num_bins=50)
    if x is not None:
        plt.loglog(x, y, 'bo', markersize=5, label='Data')
        if not np.isnan(mu_trap) and not np.isnan(x_min_trap):
            x_fit = x[x >= x_min_trap]
            if len(x_fit) > 0:
                idx_match = np.argmin(np.abs(x - x_min_trap))
                C = y[idx_match] / (x[idx_match]**(-mu_trap))
                y_fit = C * x_fit**(-mu_trap)
                plt.loglog(x_fit, y_fit, 'k--', linewidth=2, label='Fit: ~ tau^-' + str(round(mu_trap, 2)))
                plt.axvline(x_min_trap, color='gray', linestyle=':', label='x_min = ' + str(round(x_min_trap, 1)))
    plt.xlabel('Trapping Duration tau_trap')
    plt.ylabel('Probability Density P(tau_trap)')
    plt.title('Distribution of Trapping Times')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    filepath = os.path.join('data', 'trapping_times_pdf_2_' + timestamp + '.png')
    plt.savefig(filepath, dpi=300)
    plt.close()
    print('Plot saved to ' + filepath)

def plot_jump_sizes(jump_sizes, mu_jump, x_min_jump, timestamp):
    plt.figure(figsize=(8, 6))
    x, y = compute_pdf(jump_sizes, num_bins=50)
    if x is not None:
        plt.loglog(x, y, 'ro', markersize=5, label='Data')
        if not np.isnan(mu_jump) and not np.isnan(x_min_jump):
            x_fit = x[x >= x_min_jump]
            if len(x_fit) > 0:
                idx_match = np.argmin(np.abs(x - x_min_jump))
                C = y[idx_match] / (x[idx_match]**(-mu_jump))
                y_fit = C * x_fit**(-mu_jump)
                plt.loglog(x_fit, y_fit, 'k--', linewidth=2, label='Fit: ~ |dx|^-' + str(round(mu_jump, 2)))
                plt.axvline(x_min_jump, color='gray', linestyle=':', label='x_min = ' + str(round(x_min_jump, 2)))
    plt.xlabel('Jump Size |dx|')
    plt.ylabel('Probability Density P(|dx|)')
    plt.title('Distribution of Jump Sizes')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    filepath = os.path.join('data', 'jump_sizes_pdf_3_' + timestamp + '.png')
    plt.savefig(filepath, dpi=300)
    plt.close()
    print('Plot saved to ' + filepath)

def plot_cross_corr(correlation_lags, cross_corr, timestamp):
    plt.figure(figsize=(8, 6))
    plt.plot(correlation_lags, cross_corr, 'g-o', linewidth=2, markersize=5)
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Lag relative to vortex exit (time)')
    plt.ylabel('Probability of Large Jump')
    plt.title('Cross-correlation: Large Jumps vs Vortex Exits')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    filepath = os.path.join('data', 'cross_corr_jumps_exits_4_' + timestamp + '.png')
    plt.savefig(filepath, dpi=300)
    plt.close()
    print('Plot saved to ' + filepath)

if __name__ == '__main__':
    timestamp = str(int(time.time()))
    stats = np.load('data/conditional_stats.npz')
    tau_lags = stats['tau_lags']
    R_v_vortex = stats['R_v_vortex']
    R_v_strain = stats['R_v_strain']
    correlation_lags = stats['correlation_lags']
    cross_corr = stats['cross_corr']
    mu_trap = stats['mu_trap'].item()
    x_min_trap = stats['x_min_trap'].item()
    mu_jump = stats['mu_jump'].item()
    x_min_jump = stats['x_min_jump'].item()
    with open('data/trapping_jump_dists.pkl', 'rb') as f:
        dists = pickle.load(f)
    trapping_durations = dists['trapping_durations']
    jump_sizes = dists['jump_sizes']
    plot_conditional_autocorr(tau_lags, R_v_vortex, R_v_strain, timestamp)
    plot_trapping_times(trapping_durations, mu_trap, x_min_trap, timestamp)
    plot_jump_sizes(jump_sizes, mu_jump, x_min_jump, timestamp)
    plot_cross_corr(correlation_lags, cross_corr, timestamp)