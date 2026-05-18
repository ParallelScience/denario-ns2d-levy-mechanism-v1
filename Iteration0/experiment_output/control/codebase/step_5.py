# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False

def fit_power_law_tail(data, percentile=95):
    if len(data) == 0:
        return np.nan, np.nan
    threshold = np.percentile(data, percentile)
    tail_data = data[data > threshold]
    if len(tail_data) > 0:
        alpha = 1 + len(tail_data) / np.sum(np.log(tail_data / threshold))
        return alpha, threshold
    return np.nan, np.nan

def plot_conditional_stats():
    timestamp = str(int(time.time()))
    data_dir = 'data/'
    print('Loading conditional statistics...')
    cond_stats = np.load(os.path.join(data_dir, 'conditional_stats.npz'))
    lags = cond_stats['lags']
    msd_vortex = cond_stats['msd_vortex']
    msd_strain = cond_stats['msd_strain']
    dt_traj = 0.4
    tau = lags * dt_traj
    print('Plotting Conditional MSD...')
    plt.figure(figsize=(8, 6))
    plt.loglog(tau, msd_vortex, label='Vortex', linewidth=2)
    plt.loglog(tau, msd_strain, label='Strain', linewidth=2)
    plt.xlabel('Lag time tau [time]')
    plt.ylabel('MSD [length^2]')
    plt.title('Conditional Mean Squared Displacement')
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
        ax.hist(dx_v, bins=100, density=True, alpha=0.5, label='Vortex', log=True)
        ax.hist(dx_s, bins=100, density=True, alpha=0.5, label='Strain', log=True)
        ax.set_title('Lag = ' + str(lag) + ' steps (' + str(round(lag*dt_traj, 1)) + ' time)')
        ax.set_xlabel('Displacement dx [length]')
        ax.set_ylabel('PDF')
        ax.legend()
    plt.tight_layout()
    filepath_pdfs = os.path.join(data_dir, 'conditional_disp_pdfs_2_' + timestamp + '.png')
    plt.savefig(filepath_pdfs, dpi=300)
    print('Plot saved to ' + filepath_pdfs)
    plt.close()
    print('Plotting Trapping Times Distribution...')
    trapping_times = cond_stats['trapping_times']
    trapping_times = trapping_times[trapping_times > 0]
    xmin = np.percentile(trapping_times, 90)
    tail_times = trapping_times[trapping_times > xmin]
    mu = 1 + len(tail_times) / np.sum(np.log(tail_times / xmin)) if len(tail_times) > 0 else np.nan
    plt.figure(figsize=(8, 6))
    if len(trapping_times) > 1 and min(trapping_times) < max(trapping_times):
        bins_trap = np.logspace(np.log10(min(trapping_times)), np.log10(max(trapping_times)), 50)
        plt.hist(trapping_times, bins=bins_trap, density=True, alpha=0.7)
        if not np.isnan(mu):
            x_fit = np.logspace(np.log10(xmin), np.log10(max(trapping_times)), 100)
            frac_tail = len(tail_times) / len(trapping_times)
            y_fit = frac_tail * (mu - 1) / xmin * (x_fit / xmin)**(-mu)
            plt.plot(x_fit, y_fit, 'r--', linewidth=2, label='Fit: mu=' + str(round(mu, 2)))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Trapping time tau_trap [time]')
    plt.ylabel('PDF')
    plt.title('Trapping Time Distribution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filepath_trap = os.path.join(data_dir, 'trapping_times_3_' + timestamp + '.png')
    plt.savefig(filepath_trap, dpi=300)
    print('Plot saved to ' + filepath_trap)
    plt.close()
    print('Plotting Normalized Velocity Increment PDFs...')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    lags_vel = [1, 10, 50]
    for i, lag in enumerate(lags_vel):
        ax = axes[i]
        dv_v = cond_stats['vortex_' + str(lag)]
        dv_s = cond_stats['strain_' + str(lag)]
        ax.hist(dv_v, bins=100, density=True, alpha=0.5, label='Vortex', log=True)
        ax.hist(dv_s, bins=100, density=True, alpha=0.5, label='Strain', log=True)
        ax.set_title('Lag = ' + str(lag) + ' steps')
        ax.set_xlabel('Normalized Velocity Increment dv / v_rms')
        ax.set_ylabel('PDF')
        ax.legend()
    plt.tight_layout()
    filepath_vel = os.path.join(data_dir, 'norm_vel_incs_4_' + timestamp + '.png')
    plt.savefig(filepath_vel, dpi=300)
    print('Plot saved to ' + filepath_vel)
    plt.close()
    print('Re-calculating tracer states to extract Vortex-exit jumps...')
    traj_x = np.load('/home/node/work/projects/ns2d_levy_v1/data/traj_x.npy')
    traj_y = np.load('/home/node/work/projects/ns2d_levy_v1/data/traj_y.npy')
    t_traj = np.load('/home/node/work/projects/ns2d_levy_v1/data/t_traj.npy')
    snap_times = np.load('/home/node/work/projects/ns2d_levy_v1/data/snap_times.npy')
    eulerian = np.load(os.path.join(data_dir, 'eulerian_stats.npz'))
    Q_fields = eulerian['Q_fields']
    N = 256
    L = 2 * np.pi
    dx = L / N
    nearest_snap_idx = np.array([np.argmin(np.abs(snap_times - t)) for t in t_traj])
    grid_x = np.floor(traj_x / dx).astype(int) % N
    grid_y = np.floor(traj_y / dx).astype(int) % N
    n_steps, n_tracers = traj_x.shape
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
    is_exit_event = (tracer_states[:-1] == 1) & (tracer_states[1:] == 0)
    jumps_exit = dr_1[is_exit_event]
    jumps_strain = dr_1[tracer_states[:-1] == 0]
    alpha_exit, thresh_exit = fit_power_law_tail(jumps_exit)
    alpha_strain, thresh_strain = fit_power_law_tail(jumps_strain)
    print('Plotting Jump Size Distribution...')
    plt.figure(figsize=(8, 6))
    valid_exit = jumps_exit[jumps_exit > 0]
    valid_strain = jumps_strain[jumps_strain > 0]
    if len(valid_exit) > 1 and min(valid_exit) < max(valid_exit):
        bins_e = np.logspace(np.log10(min(valid_exit)), np.log10(max(valid_exit)), 50)
        plt.hist(valid_exit, bins=bins_e, density=True, alpha=0.5, label='Vortex-exit')
        if not np.isnan(alpha_exit):
            x_fit = np.logspace(np.log10(thresh_exit), np.log10(max(valid_exit)), 100)
            tail_exit = valid_exit[valid_exit > thresh_exit]
            frac_tail = len(tail_exit) / len(valid_exit)
            y_fit = frac_tail * (alpha_exit - 1) / thresh_exit * (x_fit / thresh_exit)**(-alpha_exit)
            plt.plot(x_fit, y_fit, 'b--', linewidth=2, label='Exit fit: alpha=' + str(round(alpha_exit, 2)))
    if len(valid_strain) > 1 and min(valid_strain) < max(valid_strain):
        bins_s = np.logspace(np.log10(min(valid_strain)), np.log10(max(valid_strain)), 50)
        plt.hist(valid_strain, bins=bins_s, density=True, alpha=0.5, label='Strain')
        if not np.isnan(alpha_strain):
            x_fit = np.logspace(np.log10(thresh_strain), np.log10(max(valid_strain)), 100)
            tail_strain = valid_strain[valid_strain > thresh_strain]
            frac_tail = len(tail_strain) / len(valid_strain)
            y_fit = frac_tail * (alpha_strain - 1) / thresh_strain * (x_fit / thresh_strain)**(-alpha_strain)
            plt.plot(x_fit, y_fit, 'r--', linewidth=2, label='Strain fit: alpha=' + str(round(alpha_strain, 2)))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Jump size dx [length]')
    plt.ylabel('PDF')
    plt.title('Jump Size Distribution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filepath_jump = os.path.join(data_dir, 'jump_size_dist_5_' + timestamp + '.png')
    plt.savefig(filepath_jump, dpi=300)
    print('Plot saved to ' + filepath_jump)
    plt.close()
    print('Computing detailed statistics...')
    jump_mean = np.mean(dr_1)
    jump_std = np.std(dr_1)
    large_jump_threshold = jump_mean + 3 * jump_std
    is_large_jump = dr_1 > large_jump_threshold
    is_exit_window = np.zeros_like(is_exit_event, dtype=bool)
    for n in range(n_tracers):
        exits = np.where(is_exit_event[:, n])[0]
        for e in exits:
            is_exit_window[e:min(e+6, n_steps-1), n] = True
    total_large_jumps = np.sum(is_large_jump)
    total_exit_events = np.sum(is_exit_event)
    large_jumps_at_exit = np.sum(is_large_jump & is_exit_event)
    large_jumps_near_exit = np.sum(is_large_jump & is_exit_window)
    print('\n--- Detailed Statistics ---')
    print('Total large jumps (>3 std dev): ' + str(total_large_jumps))
    print('Total exit events (Vortex -> Strain): ' + str(total_exit_events))
    if total_large_jumps > 0:
        print('Fraction of large jumps exactly at exit events: ' + str(large_jumps_at_exit / total_large_jumps))
        print('Fraction of large jumps near (within 5 steps) exit events: ' + str(large_jumps_near_exit / total_large_jumps))
    if total_exit_events > 0:
        print('Fraction of exit events producing a large jump exactly at exit: ' + str(large_jumps_at_exit / total_exit_events))
        print('Fraction of exit events producing a large jump near exit: ' + str(large_jumps_near_exit / total_exit_events))
    print('Power-law exponent for trapping times mu: ' + str(mu))
    print('Power-law exponent for Vortex-exit jumps: ' + str(alpha_exit))
    print('Power-law exponent for Strain jumps: ' + str(alpha_strain))

if __name__ == '__main__':
    plot_conditional_stats()