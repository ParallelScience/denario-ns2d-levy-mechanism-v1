# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
plt.rcParams['text.usetex'] = False
def plot_all():
    timestamp = str(int(time.time()))
    disp_dists = np.load('data/displacement_distributions.npz')
    lags = sorted(list(set([k.split('_')[1] for k in disp_dists.keys() if k.startswith('x_')])), key=int)
    fig, axes = plt.subplots(1, len(lags), figsize=(5 * len(lags), 5))
    if len(lags) == 1:
        axes = [axes]
    for i in range(len(lags)):
        lag = lags[i]
        ax = axes[i]
        data = disp_dists['x_' + lag]
        data = data[~np.isnan(data)]
        counts, bins, _ = ax.hist(data, bins=100, density=True, alpha=0.6, label='Data')
        mu, std = np.mean(data), np.std(data)
        x = np.linspace(bins[0], bins[-1], 200)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k--', linewidth=2, label='Gaussian fit')
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-5)
        ax.set_title('Lag = ' + lag + ' steps')
        ax.set_xlabel('Displacement dx [length]')
        ax.set_ylabel('PDF')
        ax.legend()
    plt.tight_layout()
    filepath = 'data/displacement_pdfs_1_' + timestamp + '.png'
    plt.savefig(filepath, dpi=300)
    print('Plot saved to ' + filepath)
    plt.close()
    print('Recomputing stable index alpha(tau)...')
    unwrapped_x = np.load('data/unwrapped_traj_x.npy')
    unwrapped_y = np.load('data/unwrapped_traj_y.npy')
    t_traj = np.load('/home/node/work/projects/ns2d_levy_v1/data/t_traj.npy')
    dt = t_traj[1] - t_traj[0]
    n_steps = unwrapped_x.shape[0]
    max_lag = n_steps // 2
    tau_indices = np.unique(np.logspace(0, np.log10(max_lag-1), 20).astype(int))
    alphas = []
    tau_alpha = []
    for lag in tau_indices:
        disp_x = (unwrapped_x[lag:] - unwrapped_x[:-lag]).flatten()
        disp_y = (unwrapped_y[lag:] - unwrapped_y[:-lag]).flatten()
        disp_all = np.concatenate([disp_x, disp_y])
        std_disp = np.std(disp_all)
        if std_disp == 0:
            alphas.append(np.nan)
            tau_alpha.append(lag * dt)
            continue
        k_vals = np.logspace(-2, 1, 100) / std_disp
        phi_k = np.zeros_like(k_vals, dtype=float)
        for i, k in enumerate(k_vals):
            phi_k[i] = np.mean(np.cos(k * disp_all))
        valid = (phi_k > 0.1) & (phi_k < 0.9)
        if np.sum(valid) > 5:
            x_fit = np.log(k_vals[valid])
            y_fit = np.log(-np.log(phi_k[valid]))
            popt, _ = np.polyfit(x_fit, y_fit, 1, cov=False)
            alpha_est = popt[0]
            alpha_est = min(max(alpha_est, 0.0), 2.0)
        else:
            alpha_est = np.nan
        alphas.append(alpha_est)
        tau_alpha.append(lag * dt)
    alphas = np.array(alphas)
    tau_alpha = np.array(tau_alpha)
    plt.figure(figsize=(8, 6))
    plt.plot(tau_alpha, alphas, 'o-', linewidth=2)
    plt.xscale('log')
    plt.xlabel('Lag time tau [time]')
    plt.ylabel('Stable index alpha')
    plt.title('Stable index alpha vs Lag time')
    plt.ylim(0.5, 2.1)
    plt.grid(True)
    plt.tight_layout()
    filepath = 'data/stable_index_2_' + timestamp + '.png'
    plt.savefig(filepath, dpi=300)
    print('Plot saved to ' + filepath)
    plt.close()
    lag_stats = np.load('data/lagrangian_stats.npz')
    tau_Rv = lag_stats['tau_Rv']
    Rv = lag_stats['Rv']
    plt.figure(figsize=(8, 6))
    plt.plot(tau_Rv, Rv, linewidth=2)
    plt.xscale('log')
    plt.xlabel('Lag time tau [time]')
    plt.ylabel('Velocity Autocorrelation R_v')
    plt.title('Lagrangian Velocity Autocorrelation')
    plt.grid(True)
    plt.tight_layout()
    filepath = 'data/velocity_autocorr_3_' + timestamp + '.png'
    plt.savefig(filepath, dpi=300)
    print('Plot saved to ' + filepath)
    plt.close()
    eulerian_stats = np.load('data/eulerian_stats.npz')
    k_vals = eulerian_stats['k_vals']
    E_k = eulerian_stats['E_k']
    plt.figure(figsize=(8, 6))
    valid = (k_vals > 0) & (E_k > 0)
    plt.loglog(k_vals[valid], E_k[valid], 'b-', linewidth=2, label='E(k)')
    mask_ls = (k_vals >= 1) & (k_vals <= 3)
    if np.sum(mask_ls) > 1:
        popt_ls = np.polyfit(np.log(k_vals[mask_ls]), np.log(E_k[mask_ls]), 1)
        slope_ls = popt_ls[0]
        k_ls = np.linspace(1, 3, 10)
        E_ls = np.exp(popt_ls[1]) * k_ls**popt_ls[0]
        plt.loglog(k_ls, E_ls, 'r--', linewidth=2, label='Fit LS [1,3]: slope=' + str(round(slope_ls, 2)))
    mask_ss = (k_vals >= 6) & (k_vals <= 20)
    if np.sum(mask_ss) > 1:
        popt_ss = np.polyfit(np.log(k_vals[mask_ss]), np.log(E_k[mask_ss]), 1)
        slope_ss = popt_ss[0]
        k_ss = np.linspace(6, 20, 10)
        E_ss = np.exp(popt_ss[1]) * k_ss**popt_ss[0]
        plt.loglog(k_ss, E_ss, 'g--', linewidth=2, label='Fit SS [6,20]: slope=' + str(round(slope_ss, 2)))
    if len(E_k) > 2 and E_k[2] > 0:
        k_ref_inv = np.linspace(1, 4, 10)
        E_ref_53 = E_k[2] * (k_ref_inv / 2)**(-5/3)
        plt.loglog(k_ref_inv, E_ref_53, 'm:', linewidth=2, label='Ref k^-5/3')
    if len(E_k) > 10 and E_k[10] > 0:
        k_ref_ens = np.linspace(6, 30, 10)
        E_ref_3 = E_k[10] * (k_ref_ens / 10)**(-3)
        plt.loglog(k_ref_ens, E_ref_3, 'k:', linewidth=2, label='Ref k^-3')
    plt.xlabel('Wavenumber k [1/length]')
    plt.ylabel('Energy Spectrum E(k) [length^3/time^2]')
    plt.title('Isotropic Energy Spectrum')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filepath = 'data/energy_spectrum_4_' + timestamp + '.png'
    plt.savefig(filepath, dpi=300)
    print('Plot saved to ' + filepath)
    plt.close()
    vorticity = np.load('/home/node/work/projects/ns2d_levy_v1/data/vorticity_snapshots.npy')
    Q_fields = eulerian_stats['Q_fields']
    plt.figure(figsize=(8, 6))
    vmax = np.max(np.abs(vorticity[0]))
    plt.imshow(vorticity[0], extent=[0, 2*np.pi, 0, 2*np.pi], origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(label='Vorticity [1/time]')
    plt.contour(Q_fields[0], levels=[0], extent=[0, 2*np.pi, 0, 2*np.pi], colors='k', linewidths=1)
    plt.xlabel('x [length]')
    plt.ylabel('y [length]')
    plt.title('Vorticity Field with Q=0 Contours (Snapshot 0)')
    plt.tight_layout()
    filepath = 'data/vorticity_q_contours_5_' + timestamp + '.png'
    plt.savefig(filepath, dpi=300)
    print('Plot saved to ' + filepath)
    plt.close()
if __name__ == '__main__':
    plot_all()