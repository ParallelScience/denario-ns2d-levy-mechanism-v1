# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['text.usetex'] = False

def generate_plots():
    data_dir = "data/"
    timestamp = int(time.time())
    
    t_traj = np.load(os.path.join(data_dir, "t_traj.npy"))
    dt = t_traj[1] - t_traj[0]
    
    msd = np.load(os.path.join(data_dir, "msd.npy"))
    H_t = np.load(os.path.join(data_dir, "H_t.npy"))
    E_k_avg = np.load(os.path.join(data_dir, "E_k_avg.npy"))
    strain_percentage = np.load(os.path.join(data_dir, "strain_percentage.npy"))
    msd_strain = np.load(os.path.join(data_dir, "msd_strain.npy"))
    H_t_strain = np.load(os.path.join(data_dir, "H_t_strain.npy"))
    msd_vortex = np.load(os.path.join(data_dir, "msd_vortex.npy"))
    H_t_vortex = np.load(os.path.join(data_dir, "H_t_vortex.npy"))
    unwrapped_x = np.load(os.path.join(data_dir, "unwrapped_traj_x.npy"))
    unwrapped_y = np.load(os.path.join(data_dir, "unwrapped_traj_y.npy"))
    vorticity_snapshots = np.load(os.path.join(data_dir, "vorticity_snapshots.npy"))
    Rv_full = np.load(os.path.join(data_dir, "Rv_full.npy"))
    Rv_strain = np.load(os.path.join(data_dir, "Rv_strain.npy"))
    Rv_vortex = np.load(os.path.join(data_dir, "Rv_vortex.npy"))
    msd_surr = np.load(os.path.join(data_dir, "msd_surr.npy"))
    disp_10 = np.load(os.path.join(data_dir, "displacements_dx_lag_10.npy"))
    disp_100 = np.load(os.path.join(data_dir, "displacements_dx_lag_100.npy"))
    disp_500 = np.load(os.path.join(data_dir, "displacements_dx_lag_500.npy"))
    trapping_durations = np.load(os.path.join(data_dir, "trapping_durations.npy"))
    jump_sizes = np.load(os.path.join(data_dir, "jump_sizes.npy"))
    mu_exponents = np.load(os.path.join(data_dir, "mu_exponents.npy"))
    mu_trap, mu_jump = mu_exponents[0], mu_exponents[1]
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(t_traj, msd, label='MSD')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Time t')
        ax1.set_ylabel('MSD')
        ax1.set_title('Ensemble MSD vs Time')
        ax1.grid(True, which="both", ls="--")
        ax2.plot(t_traj, H_t, label='H(t)', color='orange')
        ax2.set_xscale('log')
        ax2.set_xlabel('Time t')
        ax2.set_ylabel('Hurst Exponent H(t)')
        ax2.set_title('Running Hurst Exponent vs Time')
        ax2.axhline(0.5, color='r', linestyle='--', label='Normal Diffusion (H=0.5)')
        ax2.grid(True, which="both", ls="--")
        ax2.legend()
        fig.tight_layout()
        filename = os.path.join(data_dir, "msd_and_hurst_1_" + str(timestamp) + ".png")
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        print("Plot saved to " + filename)
    except Exception as e:
        print("Failed to generate Plot 1: " + str(e))
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        k_vals = np.arange(len(E_k_avg))
        valid_k = k_vals > 0
        ax.plot(k_vals[valid_k], E_k_avg[valid_k], 'o-', label='E(k)')
        k_ref_inv = np.arange(1, 4)
        if len(k_ref_inv) > 0 and E_k_avg[1] > 0:
            ref_inv = E_k_avg[1] * (k_ref_inv / 1.0)**(-3)
            ax.plot(k_ref_inv, ref_inv, 'r--', label='k^-3 (Inverse Cascade)')
        k_ref_ens = np.arange(6, 20)
        if len(k_ref_ens) > 0 and len(E_k_avg) > 6 and E_k_avg[6] > 0:
            ref_ens = E_k_avg[6] * (k_ref_ens / 6.0)**(-5/3)
            ax.plot(k_ref_ens, ref_ens, 'g--', label='k^-5/3 (Enstrophy Cascade)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('Energy Spectrum E(k)')
        ax.set_title('Eulerian Energy Spectrum')
        ax.grid(True, which="both", ls="--")
        ax.legend()
        fig.tight_layout()
        filename = os.path.join(data_dir, "energy_spectrum_2_" + str(timestamp) + ".png")
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        print("Plot saved to " + filename)
    except Exception as e:
        print("Failed to generate Plot 2: " + str(e))
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(strain_percentage, bins=50, color='skyblue', edgecolor='black')
        ax.set_xlabel('Percentage of Time in Strain Regions (%)')
        ax.set_ylabel('Number of Tracers')
        ax.set_title('Distribution of Time Spent in Strain Regions')
        ax.grid(True, ls="--")
        fig.tight_layout()
        filename = os.path.join(data_dir, "strain_histogram_3_" + str(timestamp) + ".png")
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        print("Plot saved to " + filename)
    except Exception as e:
        print("Failed to generate Plot 3: " + str(e))
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(t_traj, msd, label='Full Ensemble', color='black')
        if len(msd_strain) > 0:
            ax1.plot(t_traj, msd_strain, label='Strain-dominated', color='red')
        if len(msd_vortex) > 0:
            ax1.plot(t_traj, msd_vortex, label='Vortex-dominated', color='blue')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Time t')
        ax1.set_ylabel('MSD')
        ax1.set_title('Sub-population MSD vs Time')
        ax1.grid(True, which="both", ls="--")
        ax1.legend()
        ax2.plot(t_traj, H_t, label='Full Ensemble', color='black')
        if len(H_t_strain) > 0:
            ax2.plot(t_traj, H_t_strain, label='Strain-dominated', color='red')
        if len(H_t_vortex) > 0:
            ax2.plot(t_traj, H_t_vortex, label='Vortex-dominated', color='blue')
        ax2.set_xscale('log')
        ax2.set_xlabel('Time t')
        ax2.set_ylabel('Hurst Exponent H(t)')
        ax2.set_title('Sub-population H(t) vs Time')
        ax2.axhline(0.5, color='gray', linestyle='--')
        ax2.grid(True, which="both", ls="--")
        ax2.legend()
        fig.tight_layout()
        filename = os.path.join(data_dir, "subpop_msd_hurst_4_" + str(timestamp) + ".png")
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        print("Plot saved to " + filename)
    except Exception as e:
        print("Failed to generate Plot 4: " + str(e))
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        omega_0 = vorticity_snapshots[0]
        im = ax.imshow(omega_0, extent=[0, 2*np.pi, 0, 2*np.pi], origin='lower', cmap='RdBu_r', alpha=0.8)
        plt.colorbar(im, ax=ax, label='Vorticity')
        strain_dom_idx = np.where(strain_percentage > 70)[0]
        vortex_dom_idx = np.where(strain_percentage < 30)[0]
        num_plot = min(5, len(strain_dom_idx))
        for i in range(num_plot):
            idx = strain_dom_idx[i]
            ax.plot(unwrapped_x[:, idx] % (2*np.pi), unwrapped_y[:, idx] % (2*np.pi), 'r-', linewidth=0.5, alpha=0.7, label='Strain-dom' if i==0 else "")
        num_plot_v = min(5, len(vortex_dom_idx))
        for i in range(num_plot_v):
            idx = vortex_dom_idx[i]
            ax.plot(unwrapped_x[:, idx] % (2*np.pi), unwrapped_y[:, idx] % (2*np.pi), 'b-', linewidth=0.5, alpha=0.7, label='Vortex-dom' if i==0 else "")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Selected Trajectories over Initial Vorticity Field')
        ax.legend()
        fig.tight_layout()
        filename = os.path.join(data_dir, "trajectories_overlay_5_" + str(timestamp) + ".png")
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        print("Plot saved to " + filename)
    except Exception as e:
        print("Failed to generate Plot 5: " + str(e))
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        tau_vals = np.arange(len(Rv_full)) * dt
        ax.plot(tau_vals, Rv_full, label='Full Ensemble', color='black')
        if not np.all(np.isnan(Rv_strain)):
            ax.plot(tau_vals, Rv_strain, label='Strain-dominated', color='red')
        if not np.all(np.isnan(Rv_vortex)):
            ax.plot(tau_vals, Rv_vortex, label='Vortex-dominated', color='blue')
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel('Lag Time tau')
        ax.set_ylabel('Velocity Autocorrelation Rv(tau)')
        ax.set_title('Lagrangian Velocity Autocorrelation')
        ax.grid(True, ls="--")
        ax.legend()
        fig.tight_layout()
        filename = os.path.join(data_dir, "velocity_autocorr_6_" + str(timestamp) + ".png")
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        print("Plot saved to " + filename)
    except Exception as e:
        print("Failed to generate Plot 6: " + str(e))
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(t_traj, msd, label='Original MSD', color='black')
        ax.plot(t_traj, msd_surr, label='Surrogate MSD (Phase-Randomized)', color='green', linestyle='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Time t')
        ax.set_ylabel('MSD')
        ax.set_title('Original vs Surrogate MSD')
        ax.grid(True, which="both", ls="--")
        ax.legend()
        fig.tight_layout()
        filename = os.path.join(data_dir, "surrogate_msd_7_" + str(timestamp) + ".png")
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        print("Plot saved to " + filename)
    except Exception as e:
        print("Failed to generate Plot 7: " + str(e))
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        for disp, lag, color in zip([disp_10, disp_100, disp_500], [10, 100, 500], ['blue', 'green', 'red']):
            tau = lag * dt
            counts, bin_edges = np.histogram(disp, bins=100, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            valid = counts > 0
            ax.plot(bin_centers[valid], counts[valid], marker='.', linestyle='none', color=color, label='tau = ' + str(np.round(tau, 1)))
        ax.set_yscale('log')
        ax.set_xlabel('Displacement delta x')
        ax.set_ylabel('PDF P(delta x, tau)')
        ax.set_title('Displacement PDFs at Multiple Lags')
        ax.grid(True, ls="--")
        ax.legend()
        fig.tight_layout()
        filename = os.path.join(data_dir, "displacement_pdfs_8_" + str(timestamp) + ".png")
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        print("Plot saved to " + filename)
    except Exception as e:
        print("Failed to generate Plot 8: " + str(e))
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        if len(trapping_durations) > 0:
            counts_t, bin_edges_t = np.histogram(trapping_durations, bins=np.logspace(np.log10(max(dt, np.min(trapping_durations))), np.log10(np.max(trapping_durations)), 50), density=True)
            bin_centers_t = (bin_edges_t[:-1] + bin_edges_t[1:]) / 2
            valid_t = counts_t > 0
            ax1.plot(bin_centers_t[valid_t], counts_t[valid_t], 'bo', label='Data')
            if not np.isnan(mu_trap):
                x_fit = bin_centers_t[valid_t]
                y_fit = x_fit**(-mu_trap)
                y_fit = y_fit * (counts_t[valid_t][0] / y_fit[0])
                ax1.plot(x_fit, y_fit, 'r--', label='Fit mu=' + str(np.round(mu_trap, 2)))
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_xlabel('Trapping Duration tau_trap')
            ax1.set_ylabel('PDF')
            ax1.set_title('Trapping Time Distribution')
            ax1.grid(True, which="both", ls="--")
            ax1.legend()
        if len(jump_sizes) > 0:
            counts_j, bin_edges_j = np.histogram(jump_sizes, bins=np.logspace(np.log10(max(1e-5, np.min(jump_sizes))), np.log10(np.max(jump_sizes)), 50), density=True)
            bin_centers_j = (bin_edges_j[:-1] + bin_edges_j[1:]) / 2
            valid_j = counts_j > 0
            ax2.plot(bin_centers_j[valid_j], counts_j[valid_j], 'go', label='Data')
            if not np.isnan(mu_jump):
                x_fit = bin_centers_j[valid_j]
                y_fit = x_fit**(-mu_jump)
                y_fit = y_fit * (counts_j[valid_j][0] / y_fit[0])
                ax2.plot(x_fit, y_fit, 'r--', label='Fit mu=' + str(np.round(mu_jump, 2)))
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.set_xlabel('Jump Size')
            ax2.set_ylabel('PDF')
            ax2.set_title('Jump Size Distribution')
            ax2.grid(True, which="both", ls="--")
            ax2.legend()
        fig.tight_layout()
        filename = os.path.join(data_dir, "trapping_jump_dists_9_" + str(timestamp) + ".png")
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        print("Plot saved to " + filename)
    except Exception as e:
        print("Failed to generate Plot 9: " + str(e))

if __name__ == '__main__':
    generate_plots()