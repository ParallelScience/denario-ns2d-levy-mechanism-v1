# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from scipy.optimize import curve_fit

plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 12

def linear_fit(x, a, b):
    return a * x + b

if __name__ == '__main__':
    data_dir = 'data/'
    stats_path = os.path.join(data_dir, 'step1_lagrangian_stats.npz')
    pdfs_path = os.path.join(data_dir, 'step1_pdfs.pkl')
    stats = np.load(stats_path)
    t_traj = stats['t_traj']
    msd = stats['msd']
    H = stats['H']
    vacf = stats['vacf']
    T_L = stats['T_L']
    msd_sub = stats['msd_sub']
    H_sub = stats['H_sub']
    tau_indices = stats['tau_indices']
    with open(pdfs_path, 'rb') as f:
        pdfs_data = pickle.load(f)
    disp_pdfs = pdfs_data['disp_pdfs']
    v_inc_pdfs = pdfs_data['v_inc_pdfs']
    alpha_values = pdfs_data['alpha_values']
    timestamp = int(time.time())
    fig1, axs1 = plt.subplots(1, 3, figsize=(18, 5))
    ax = axs1[0]
    ax.loglog(t_traj, msd, label='Original MSD', color='blue', linewidth=2)
    ax.loglog(t_traj, msd_sub, label='Mean-subtracted MSD', color='red', linewidth=2)
    fit_mask = (t_traj > 10) & (msd > 0)
    if np.any(fit_mask):
        popt, _ = curve_fit(linear_fit, np.log(t_traj[fit_mask]), np.log(msd[fit_mask]))
        fit_line = np.exp(popt[1]) * t_traj[fit_mask]**popt[0]
        ax.loglog(t_traj[fit_mask], fit_line, 'k--', label='Fit Original (H=' + str(round(float(H), 3)) + ')')
        popt_sub, _ = curve_fit(linear_fit, np.log(t_traj[fit_mask]), np.log(msd_sub[fit_mask]))
        fit_line_sub = np.exp(popt_sub[1]) * t_traj[fit_mask]**popt_sub[0]
        ax.loglog(t_traj[fit_mask], fit_line_sub, 'k:', label='Fit Subtracted (H=' + str(round(float(H_sub), 3)) + ')')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Mean Squared Displacement (MSD)')
    ax.set_title('MSD vs Time')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax = axs1[1]
    ax.plot(t_traj, vacf, color='green', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag time tau')
    ax.set_ylabel('Velocity Autocorrelation Rv(tau)')
    ax.set_title('Lagrangian Velocity Autocorrelation\nTL = ' + str(round(float(T_L), 2)))
    ax.set_xlim(0, min(200, t_traj[-1]))
    ax.grid(True, ls='--', alpha=0.5)
    ax = axs1[2]
    taus = []
    alphas = []
    alpha_errs = []
    for lag in tau_indices:
        lag = int(lag)
        taus.append(alpha_values[lag]['tau'])
        alphas.append(alpha_values[lag]['alpha'])
        alpha_errs.append(alpha_values[lag]['alpha_err'])
    ax.errorbar(taus, alphas, yerr=alpha_errs, fmt='o-', color='purple', capsize=5, linewidth=2)
    ax.axhline(2.0, color='black', linestyle='--', label='Gaussian (alpha=2)')
    ax.set_xlabel('Lag time tau')
    ax.set_ylabel('Stable index alpha')
    ax.set_title('Estimated Stable Index vs Lag Time')
    ax.set_ylim(0, 2.2)
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)
    fig1.tight_layout()
    plot1_filename = os.path.join(data_dir, 'msd_results_1_' + str(timestamp) + '.png')
    fig1.savefig(plot1_filename, dpi=300)
    print('Plot saved to ' + plot1_filename)
    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 6))
    ax = axs2[0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(tau_indices)))
    max_disp = 0
    for i, lag in enumerate(tau_indices):
        lag = int(lag)
        tau = disp_pdfs[lag]['tau']
        bin_centers = disp_pdfs[lag]['bin_centers']
        hist = disp_pdfs[lag]['hist']
        ax.semilogy(bin_centers, hist, color=colors[i], label='tau=' + str(round(float(tau), 1)), linewidth=2)
        variance = np.trapz(hist * bin_centers**2, bin_centers)
        if variance > 0:
            sigma = np.sqrt(variance)
            gaussian = 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (bin_centers / sigma)**2)
            ax.semilogy(bin_centers, gaussian, color=colors[i], linestyle='--', alpha=0.7)
        mask = hist > 1e-5
        if np.any(mask):
            max_disp = max(max_disp, np.max(np.abs(bin_centers[mask])))
    ax.set_xlabel('Displacement Delta x, Delta y')
    ax.set_ylabel('Probability Density')
    ax.set_title('Displacement PDFs with Gaussian References (dashed)')
    ax.set_ylim(bottom=1e-5)
    if max_disp > 0:
        ax.set_xlim(-max_disp, max_disp)
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)
    ax = axs2[1]
    max_v_inc = 0
    for i, lag in enumerate(tau_indices):
        lag = int(lag)
        tau = v_inc_pdfs[lag]['tau']
        bin_centers = v_inc_pdfs[lag]['bin_centers']
        hist = v_inc_pdfs[lag]['hist']
        ax.semilogy(bin_centers, hist, color=colors[i], label='tau=' + str(round(float(tau), 1)), linewidth=2)
        mask = hist > 1e-5
        if np.any(mask):
            max_v_inc = max(max_v_inc, np.max(np.abs(bin_centers[mask])))
    ax.set_xlabel('Velocity Increment delta v')
    ax.set_ylabel('Probability Density')
    ax.set_title('Velocity Increment PDFs')
    ax.set_ylim(bottom=1e-5)
    if max_v_inc > 0:
        ax.set_xlim(-max_v_inc, max_v_inc)
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)
    fig2.tight_layout()
    plot2_filename = os.path.join(data_dir, 'pdf_results_2_' + str(timestamp) + '.png')
    fig2.savefig(plot2_filename, dpi=300)
    print('Plot saved to ' + plot2_filename)