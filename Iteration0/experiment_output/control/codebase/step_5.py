# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams['text.usetex'] = False

def fit_power_law(data, xmin=None):
    if xmin is None:
        xmin = np.percentile(data, 90)
    data_tail = data[data >= xmin]
    if len(data_tail) < 2:
        return np.nan, xmin
    alpha = 1 + len(data_tail) / np.sum(np.log(data_tail / xmin))
    return alpha, xmin

def plot_conditional_stats():
    timestamp = str(int(time.time()))
    print('Loading conditional statistics...')
    cond_stats = np.load('data/conditional_stats.npz')
    lags = cond_stats['lags']
    dt_traj = 0.4
    tau = lags * dt_traj
    msd_vortex = cond_stats['msd_vortex']
    msd_strain = cond_stats['msd_strain']
    print('Plotting Conditional MSD...')
    plt.figure(figsize=(8, 6))
    valid = (msd_vortex > 0) & (msd_strain > 0)
    plt.loglog(tau[valid], msd_vortex[valid], 'r-', linewidth=2, label='Vortex')
    plt.loglog(tau[valid], msd_strain[valid], 'b-', linewidth=2, label='Strain')
    plt.xlabel('Lag time tau [time]')
    plt.ylabel('Conditional MSD [length^2]')
    plt.title('Conditional Mean Squared Displacement')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filepath = 'data/conditional_msd_1_' + timestamp + '.png'
    plt.savefig(filepath, dpi=300)
    print('Plot saved to ' + filepath)
    plt.close()
    print('Plotting Conditional Displacement PDFs...')
    lag_to_plot = 100
    dx_v = cond_stats['vortex_x_' + str(lag_to_plot)]
    dx_s = cond_stats['strain_x_' + str(lag_to_plot)]
    plt.figure(figsize=(8, 6))
    plt.hist(dx_v, bins=100, density=True, alpha=0.5, color='red', label='Vortex', log=True)
    plt.hist(dx_s, bins=100, density=True, alpha=0.5, color='blue', label='Strain', log=True)
    plt.xlabel('Displacement Delta x [length]')
    plt.ylabel('PDF')
    plt.title('Conditional Displacement PDF (Lag = ' + str(lag_to_plot * dt_traj) + ' time)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filepath = 'data/conditional_disp_pdf_2_' + timestamp + '.png'
    plt.savefig(filepath, dpi=300)
    print('Plot saved to ' + filepath)
    plt.close()
    print('Plotting Trapping Times...')
    trapping_times = cond_stats['trapping_times']
    trapping_times = trapping_times[trapping_times > 0]
    plt.figure(figsize=(8, 6))
    counts, bins, _ = plt.hist(trapping_times, bins=np.logspace(np.log10(min(trapping_times)), np.log10(max(trapping_times)), 50), density=True, alpha=0.7, color='green', label='Data')
    mu, xmin = fit_power_law(trapping_times, xmin=np.percentile(trapping_times, 75))
    if not np.isnan(mu):
        x_fit = np.linspace(xmin, max(trapping_times), 100)
        y_fit = (mu - 1) / xmin * (x_fit / xmin)**(-mu)
        scale = np.sum(trapping_times >= xmin) / len(trapping_times)
        plt.plot(x_fit, y_fit * scale, 'k--', linewidth=2, label='Fit: tau^{-' + str(round(mu, 2)) + '}')
        print('Trapping time power-law exponent mu: ' + str(mu))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Trapping Time tau_trap [time]')
    plt.ylabel('PDF')
    plt.title('Trapping Time Distribution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filepath = 'data/trapping_times_3_' + timestamp + '.png'
    plt.savefig(filepath, dpi=300)
    print('Plot saved to ' + filepath)
    plt.close()
    print('Plotting Normalized Velocity Increment PDFs...')
    lag_v = 10
    dv_v = cond_stats['vortex_' + str(lag_v)]
    dv_s = cond_stats['strain_' + str(lag_v)]
    plt.figure(figsize=(8, 6))
    plt.hist(dv_v, bins=100, density=True, alpha=0.5, color='red', label='Vortex', log=True)
    plt.hist(dv_s, bins=100, density=True, alpha=0.5, color='blue', label='Strain', log=True)
    plt.xlabel('Normalized Velocity Increment |delta v| / v_rms')
    plt.ylabel('PDF')
    plt.title('Normalized Velocity Increment PDF (Lag = ' + str(lag_v * dt_traj) + ' time)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filepath = 'data/norm_vel_inc_pdf_4_' + timestamp + '.png'
    plt.savefig(filepath, dpi=300)
    print('Plot saved to ' + filepath)
    plt.close()
    print('Re-computing detailed statistics for large jumps and vortex exit events...')
    traj_x = np.load('/home/node/work/projects/ns2d_levy_v1/data/traj_x.npy')
    traj_y = np.load('/home/node/work/projects/ns2d_levy_v1/data/traj_y.npy')
    t_traj = np.load('/home/node/work/projects/ns2d_levy_v1/data/t_traj.npy')
    snap_times = np.load('/home/node/work/projects/ns2d_levy_v1/data/snap_times.npy')
    eulerian = np.load('data/eulerian_stats.npz')
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
    unwrapped_x = np.load('data/unwrapped_traj_x.npy')
    unwrapped_y = np.load('data/unwrapped_traj_y.npy')
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
    print('--- Detailed Statistics ---')
    print('Total large jumps (>3 std dev): ' + str(total_large_jumps))
    print('Total exit events (Vortex -> Strain): ' + str(total_exit_events))
    if total_large_jumps > 0:
        print('Fraction of large jumps exactly at exit events: ' + str(large_jumps_at_exit / total_large_jumps))
        print('Fraction of large jumps near (within 5 steps) exit events: ' + str(large_jumps_near_exit / total_large_jumps))
    if total_exit_events > 0:
        print('Fraction of exit events producing a large jump exactly at exit: ' + str(large_jumps_at_exit / total_exit_events))
        print('Fraction of exit events producing a large jump near exit: ' + str(large_jumps_near_exit / total_exit_events))
    print('---------------------------')
    print('Plotting Jump Size Distribution (Exit vs Strain)...')
    jumps_exit = dr_1[is_exit_window]
    jumps_strain_general = dr_1[tracer_states[:-1] == 0]
    jumps_exit = jumps_exit[jumps_exit > 0]
    jumps_strain_general = jumps_strain_general[jumps_strain_general > 0]
    plt.figure(figsize=(8, 6))
    counts_e, bins_e, _ = plt.hist(jumps_exit, bins=np.logspace(np.log10(min(jumps_exit)), np.log10(max(jumps_exit)), 50), density=True, alpha=0.5, color='orange', label='Vortex-Exit Jumps')
    counts_s, bins_s, _ = plt.hist(jumps_strain_general, bins=np.logspace(np.log10(min(jumps_strain_general)), np.log10(max(jumps_strain_general)), 50), density=True, alpha=0.5, color='blue', label='Strain Jumps')
    alpha_e, xmin_e = fit_power_law(jumps_exit, xmin=np.percentile(jumps_exit, 90))
    alpha_s, xmin_s = fit_power_law(jumps_strain_general, xmin=np.percentile(jumps_strain_general, 90))
    if not np.isnan(alpha_e):
        x_fit = np.linspace(xmin_e, max(jumps_exit), 100)
        y_fit = (alpha_e - 1) / xmin_e * (x_fit / xmin_e)**(-alpha_e)
        scale = np.sum(jumps_exit >= xmin_e) / len(jumps_exit)
        plt.plot(x_fit, y_fit * scale, 'r--', linewidth=2, label='Exit Fit: x^{-' + str(round(alpha_e, 2)) + '}')
        print('Vortex-exit jump power-law exponent: ' + str(alpha_e))
    if not np.isnan(alpha_s):
        x_fit = np.linspace(xmin_s, max(jumps_strain_general), 100)
        y_fit = (alpha_s - 1) / xmin_s * (x_fit / xmin_s)**(-alpha_s)
        scale = np.sum(jumps_strain_general >= xmin_s) / len(jumps_strain_general)
        plt.plot(x_fit, y_fit * scale, 'k--', linewidth=2, label='Strain Fit: x^{-' + str(round(alpha_s, 2)) + '}')
        print('Strain jump power-law exponent: ' + str(alpha_s))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Jump Size Delta x [length]')
    plt.ylabel('PDF')
    plt.title('Jump Size Distribution (Exit vs Strain)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filepath = 'data/jump_size_dist_exit_vs_strain_5_' + timestamp + '.png'
    plt.savefig(filepath, dpi=300)
    print('Plot saved to ' + filepath)
    plt.close()

if __name__ == '__main__':
    plot_conditional_stats()