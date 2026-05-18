# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['text.usetex'] = False

def plot_conditional_stats():
    timestamp = str(int(time.time()))
    data_dir = 'data'
    
    print('Loading conditional statistics...')
    cond_stats = np.load(os.path.join(data_dir, 'conditional_stats.npz'))
    lags = cond_stats['lags']
    msd_vortex = cond_stats['msd_vortex']
    msd_strain = cond_stats['msd_strain']
    trapping_times = cond_stats['trapping_times']
    
    dt_traj = 0.4
    tau_vals = lags * dt_traj
    
    print('Plotting Conditional MSD...')
    plt.figure(figsize=(8, 6))
    plt.loglog(tau_vals, msd_vortex, 'r-', linewidth=2, label='Vortex')
    plt.loglog(tau_vals, msd_strain, 'b-', linewidth=2, label='Strain')
    plt.xlabel('Lag time tau [time]')
    plt.ylabel('Mean Squared Displacement (MSD) [length^2]')
    plt.title('Conditional MSD: Vortex vs Strain')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filepath_msd = os.path.join(data_dir, 'conditional_msd_1_' + timestamp + '.png')
    plt.savefig(filepath_msd, dpi=300)
    print('Plot saved to ' + filepath_msd)
    plt.close()
    
    print('Plotting Conditional Displacement PDFs...')
    selected_lags = [10, 50, 100, 200]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, lag in enumerate(selected_lags):
        ax = axes[i]
        dx_v = cond_stats['vortex_x_' + str(lag)]
        dx_s = cond_stats['strain_x_' + str(lag)]
        dx_v = dx_v[~np.isnan(dx_v)]
        dx_s = dx_s[~np.isnan(dx_s)]
        if len(dx_v) > 0:
            dx_v_std = (dx_v - np.mean(dx_v)) / np.std(dx_v)
            ax.hist(dx_v_std, bins=100, density=True, alpha=0.5, color='red', label='Vortex', log=True)
        if len(dx_s) > 0:
            dx_s_std = (dx_s - np.mean(dx_s)) / np.std(dx_s)
            ax.hist(dx_s_std, bins=100, density=True, alpha=0.5, color='blue', label='Strain', log=True)
        x_vals = np.linspace(-5, 5, 200)
        ax.plot(x_vals, 1/np.sqrt(2*np.pi)*np.exp(-0.5*x_vals**2), 'k--', label='Gaussian')
        ax.set_title('Lag = ' + str(lag) + ' steps')
        ax.set_xlabel('Standardized Displacement X')
        ax.set_ylabel('PDF')
        ax.set_ylim(bottom=1e-5)
        ax.legend()
    plt.tight_layout()
    filepath_disp = os.path.join(data_dir, 'conditional_disp_pdfs_2_' + timestamp + '.png')
    plt.savefig(filepath_disp, dpi=300)
    print('Plot saved to ' + filepath_disp)
    plt.close()
    
    print('Plotting Trapping Times Distribution...')
    plt.figure(figsize=(8, 6))
    trapping_times = trapping_times[trapping_times > 0]
    if len(trapping_times) > 0 and min(trapping_times) < max(trapping_times):
        counts, bins = np.histogram(trapping_times, bins=np.logspace(np.log10(min(trapping_times)), np.log10(max(trapping_times)), 50), density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        valid = counts > 0
        plt.loglog(bin_centers[valid], counts[valid], 'ko', label='Data')
        tail_mask = (bin_centers > 10) & valid
        if np.sum(tail_mask) > 3:
            try:
                popt, _ = curve_fit(lambda x, a, mu: a * x**(-mu), bin_centers[tail_mask], counts[tail_mask], p0=[1.0, 1.5])
                mu_est = popt[1]
                plt.loglog(bin_centers[tail_mask], popt[0] * bin_centers[tail_mask]**(-mu_est), 'r-', linewidth=2, label='Fit: mu=' + str(round(mu_est, 2)))
                print('Estimated trapping time power-law exponent mu: ' + str(mu_est))
            except Exception as e:
                print('Could not fit trapping times: ' + str(e))
    plt.xlabel('Trapping Time tau [time]')
    plt.ylabel('PDF P(tau) [1/time]')
    plt.title('Trapping Time Distribution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filepath_trap = os.path.join(data_dir, 'trapping_times_pdf_3_' + timestamp + '.png')
    plt.savefig(filepath_trap, dpi=300)
    print('Plot saved to ' + filepath_trap)
    plt.close()
    
    print('Plotting Normalized Velocity Increment PDFs...')
    vel_lags = [1, 10, 50]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, lag in enumerate(vel_lags):
        ax = axes[i]
        dv_v = cond_stats['vortex_' + str(lag)]
        dv_s = cond_stats['strain_' + str(lag)]
        dv_v = dv_v[~np.isnan(dv_v)]
        dv_s = dv_s[~np.isnan(dv_s)]
        if len(dv_v) > 0:
            ax.hist(dv_v, bins=100, density=True, alpha=0.5, color='red', label='Vortex', log=True)
        if len(dv_s) > 0:
            ax.hist(dv_s, bins=100, density=True, alpha=0.5, color='blue', label='Strain', log=True)
        ax.set_title('Lag = ' + str(lag) + ' steps')
        ax.set_xlabel('Normalized Velocity Increment dv/v_rms')
        ax.set_ylabel('PDF')
        ax.set_ylim(bottom=1e-6)
        ax.legend()
    plt.tight_layout()
    filepath_vel = os.path.join(data_dir, 'normalized_vel_incs_4_' + timestamp + '.png')
    plt.savefig(filepath_vel, dpi=300)
    print('Plot saved to ' + filepath_vel)
    plt.close()
    
    print('Recomputing detailed statistics for large jumps and vortex exit events...')
    traj_x = np.load('/home/node/work/projects/ns2d_levy_v1/data/traj_x.npy')
    traj_y = np.load('/home/node/work/projects/ns2d_levy_v1/data/traj_y.npy')
    t_traj = np.load('/home/node/work/projects/ns2d_levy_v1/data/t_traj.npy')
    snap_times = np.load('/home/node/work/projects/ns2d_levy_v1/data/snap_times.npy')
    eulerian = np.load(os.path.join(data_dir, 'eulerian_stats.npz'))
    Q_fields = eulerian['Q_fields']
    n_steps, n_tracers = traj_x.shape
    N = 256
    L = 2 * np.pi
    dx = L / N
    nearest_snap_idx = np.array([np.argmin(np.abs(snap_times - t)) for t in t_traj])
    grid_x = np.floor(traj_x / dx).astype(int) % N
    grid_y = np.floor(traj_y / dx).astype(int) % N
    tracer_states = np.zeros((n_steps, n_tracers), dtype=int)
    for t_idx in range(n_steps):
        s_idx = nearest_snap_idx[t_idx]
        q_vals = Q_fields[s_idx, grid_y[t_idx], grid_x[t_idx]]
        tracer_states[t_idx] = (q_vals < 0).astype(int)
    unwrapped_x = np.load(os.path.join(data_dir, 'unwrapped_traj_x.npy'))
    unwrapped_y = np.load(os.path.join(data_dir, 'unwrapped_traj_y.npy'))
    dx_1 = unwrapped_x[1:] - unwrapped_x[:-1]
    dy_1 = unwrapped_y[1:] - unwrapped_y[:-1]
    dr_1 = np.sqrt(dx_1**2 + dy_1**2)
    jump_mean = np.mean(dr_1)
    jump_std = np.std(dr_1)
    large_jump_threshold = jump_mean + 3 * jump_std
    is_large_jump = dr_1 > large_jump_threshold
    is_exit_event = (tracer_states[:-1] == 1) & (tracer_states[1:] == 0)
    is_exit_window = np.zeros_like(is_exit_event, dtype=bool)
    for n in range(n_tracers):
        exits = np.where(is_exit_event[:, n])[0]
        for e in exits:
            is_exit_window[e:e+6, n] = True
    total_large_jumps = np.sum(is_large_jump)
    total_exit_events = np.sum(is_exit_event)
    large_jumps_at_exit = np.sum(is_large_jump & is_exit_event)
    large_jumps_near_exit = np.sum(is_large_jump & is_exit_window)
    print('Total large jumps (>3 std dev): ' + str(total_large_jumps))
    print('Total exit events (Vortex -> Strain): ' + str(total_exit_events))
    if total_large_jumps > 0:
        print('Fraction of large jumps exactly at exit events: ' + str(large_jumps_at_exit / total_large_jumps))
        print('Fraction of large jumps near (within 5 steps) exit events: ' + str(large_jumps_near_exit / total_large_jumps))
    if total_exit_events > 0:
        print('Fraction of exit events producing a large jump exactly at exit: ' + str(large_jumps_at_exit / total_exit_events))
        print('Fraction of exit events producing a large jump near exit: ' + str(large_jumps_near_exit / total_exit_events))
    print('Plotting Jump Size Distribution...')
    plt.figure(figsize=(8, 6))
    jumps_exit = dr_1[is_exit_event]
    jumps_strain_general = dr_1[tracer_states[:-1] == 0]
    jumps_exit = jumps_exit[jumps_exit > 0]
    jumps_strain_general = jumps_strain_general[jumps_strain_general > 0]
    if len(jumps_exit) > 0 and min(jumps_exit) < max(jumps_exit) and len(jumps_strain_general) > 0 and min(jumps_strain_general) < max(jumps_strain_general):
        counts_e, bins_e = np.histogram(jumps_exit, bins=np.logspace(np.log10(min(jumps_exit)), np.log10(max(jumps_exit)), 50), density=True)
        counts_s, bins_s = np.histogram(jumps_strain_general, bins=np.logspace(np.log10(min(jumps_strain_general)), np.log10(max(jumps_strain_general)), 50), density=True)
        bc_e = (bins_e[:-1] + bins_e[1:]) / 2
        bc_s = (bins_s[:-1] + bins_s[1:]) / 2
        valid_e = counts_e > 0
        valid_s = counts_s > 0
        plt.loglog(bc_e[valid_e], counts_e[valid_e], 'ro', alpha=0.6, label='Vortex-Exit Jumps')
        plt.loglog(bc_s[valid_s], counts_s[valid_s], 'bo', alpha=0.6, label='Strain Jumps')
        tail_e = (bc_e > np.percentile(jumps_exit, 90)) & valid_e
        tail_s = (bc_s > np.percentile(jumps_strain_general, 95)) & valid_s
        if np.sum(tail_e) > 3:
            try:
                popt_e, _ = curve_fit(lambda x, a, alpha: a * x**(-alpha), bc_e[tail_e], counts_e[tail_e], p0=[1.0, 3.0])
                plt.loglog(bc_e[tail_e], popt_e[0] * bc_e[tail_e]**(-popt_e[1]), 'r-', linewidth=2, label='Exit Fit: alpha=' + str(round(popt_e[1], 2)))
                print('Estimated power-law exponent for Vortex-Exit jumps: ' + str(popt_e[1]))
            except Exception as e:
                print('Could not fit exit jumps: ' + str(e))
        if np.sum(tail_s) > 3:
            try:
                popt_s, _ = curve_fit(lambda x, a, alpha: a * x**(-alpha), bc_s[tail_s], counts_s[tail_s], p0=[1.0, 3.0])
                plt.loglog(bc_s[tail_s], popt_s[0] * bc_s[tail_s]**(-popt_s[1]), 'b-', linewidth=2, label='Strain Fit: alpha=' + str(round(popt_s[1], 2)))
                print('Estimated power-law exponent for Strain jumps: ' + str(popt_s[1]))
            except Exception as e:
                print('Could not fit strain jumps: ' + str(e))
    plt.xlabel('Jump Size Delta x [length]')
    plt.ylabel('PDF P(Delta x) [1/length]')
    plt.title('Jump Size Distribution: Vortex-Exit vs Strain')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filepath_jump = os.path.join(data_dir, 'jump_size_dist_5_' + timestamp + '.png')
    plt.savefig(filepath_jump, dpi=300)
    print('Plot saved to ' + filepath_jump)
    plt.close()

if __name__ == '__main__':
    plot_conditional_stats()