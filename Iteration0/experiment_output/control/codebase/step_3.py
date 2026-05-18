# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np

def fit_power_law_tail(data, percentile=95):
    if len(data) == 0:
        return np.nan
    threshold = np.percentile(data, percentile)
    tail_data = data[data > threshold]
    if len(tail_data) > 0:
        alpha = 1 + len(tail_data) / np.sum(np.log(tail_data / threshold))
        return alpha
    return np.nan

if __name__ == '__main__':
    print('Loading data...')
    traj_x = np.load('/home/node/work/projects/ns2d_levy_v1/data/traj_x.npy')
    traj_y = np.load('/home/node/work/projects/ns2d_levy_v1/data/traj_y.npy')
    t_traj = np.load('/home/node/work/projects/ns2d_levy_v1/data/t_traj.npy')
    unwrapped_x = np.load('data/unwrapped_traj_x.npy')
    unwrapped_y = np.load('data/unwrapped_traj_y.npy')
    snap_times = np.load('/home/node/work/projects/ns2d_levy_v1/data/snap_times.npy')
    eulerian = np.load('data/eulerian_stats.npz')
    Q_fields = eulerian['Q_fields']
    n_steps, n_tracers = traj_x.shape
    dt_traj = t_traj[1] - t_traj[0]
    dt_snap = snap_times[1] - snap_times[0] if len(snap_times) > 1 else 0
    print('Temporal resolution mismatch:')
    print('Snapshots are ' + str(dt_snap) + ' time units apart.')
    print('Trajectories are ' + str(dt_traj) + ' time units apart.')
    N = 256
    L = 2 * np.pi
    dx = L / N
    print('Mapping tracers to Eulerian grid and assigning states...')
    nearest_snap_idx = np.array([np.argmin(np.abs(snap_times - t)) for t in t_traj])
    grid_x = np.floor(traj_x / dx).astype(int) % N
    grid_y = np.floor(traj_y / dx).astype(int) % N
    tracer_states = np.zeros((n_steps, n_tracers), dtype=int)
    for t_idx in range(n_steps):
        s_idx = nearest_snap_idx[t_idx]
        q_vals = Q_fields[s_idx, grid_y[t_idx], grid_x[t_idx]]
        tracer_states[t_idx] = (q_vals < 0).astype(int)
    total_steps = n_steps * n_tracers
    vortex_steps = np.sum(tracer_states == 1)
    strain_steps = np.sum(tracer_states == 0)
    print('Fraction of total trajectory time-steps in Vortex: ' + str(vortex_steps / total_steps))
    print('Fraction of total trajectory time-steps in Strain: ' + str(strain_steps / total_steps))
    vortex_frac_per_tracer = np.mean(tracer_states == 1, axis=0)
    predominantly_vortex = vortex_frac_per_tracer > 0.5
    predominantly_strain = vortex_frac_per_tracer <= 0.5
    print('Fraction of tracers predominantly in Vortex: ' + str(np.mean(predominantly_vortex)))
    print('Fraction of tracers predominantly in Strain: ' + str(np.mean(predominantly_strain)))
    print('Computing conditional MSD and displacement PDFs...')
    vortex_tracers_idx = np.where(predominantly_vortex)[0]
    strain_tracers_idx = np.where(predominantly_strain)[0]
    max_lag = n_steps // 2
    lags = np.arange(1, max_lag)
    msd_vortex = np.zeros(len(lags))
    msd_strain = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if len(vortex_tracers_idx) > 0:
            disp_x_v = unwrapped_x[lag:, vortex_tracers_idx] - unwrapped_x[:-lag, vortex_tracers_idx]
            disp_y_v = unwrapped_y[lag:, vortex_tracers_idx] - unwrapped_y[:-lag, vortex_tracers_idx]
            msd_vortex[i] = np.mean(disp_x_v**2 + disp_y_v**2)
        if len(strain_tracers_idx) > 0:
            disp_x_s = unwrapped_x[lag:, strain_tracers_idx] - unwrapped_x[:-lag, strain_tracers_idx]
            disp_y_s = unwrapped_y[lag:, strain_tracers_idx] - unwrapped_y[:-lag, strain_tracers_idx]
            msd_strain[i] = np.mean(disp_x_s**2 + disp_y_s**2)
    selected_lags = [10, 50, 100, 200]
    cond_disp_pdfs = {}
    for lag in selected_lags:
        if len(vortex_tracers_idx) > 0:
            dx_v = (unwrapped_x[lag:, vortex_tracers_idx] - unwrapped_x[:-lag, vortex_tracers_idx]).flatten()
            dy_v = (unwrapped_y[lag:, vortex_tracers_idx] - unwrapped_y[:-lag, vortex_tracers_idx]).flatten()
            cond_disp_pdfs['vortex_x_' + str(lag)] = dx_v
            cond_disp_pdfs['vortex_y_' + str(lag)] = dy_v
        if len(strain_tracers_idx) > 0:
            dx_s = (unwrapped_x[lag:, strain_tracers_idx] - unwrapped_x[:-lag, strain_tracers_idx]).flatten()
            dy_s = (unwrapped_y[lag:, strain_tracers_idx] - unwrapped_y[:-lag, strain_tracers_idx]).flatten()
            cond_disp_pdfs['strain_x_' + str(lag)] = dx_s
            cond_disp_pdfs['strain_y_' + str(lag)] = dy_s
    print('Identifying trapping events...')
    trapping_times = []
    for n in range(n_tracers):
        states = tracer_states[:, n]
        padded = np.pad(states, (1, 1), mode='constant')
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        durations = (ends - starts) * dt_traj
        trapping_times.extend(durations)
    trapping_times = np.array(trapping_times)
    print('Computing jump size distributions...')
    dx_1 = unwrapped_x[1:] - unwrapped_x[:-1]
    dy_1 = unwrapped_y[1:] - unwrapped_y[:-1]
    dr_1 = np.sqrt(dx_1**2 + dy_1**2)
    state_t = tracer_states[:-1]
    jumps_vortex = dr_1[state_t == 1]
    jumps_strain = dr_1[state_t == 0]
    alpha_jump_vortex = fit_power_law_tail(jumps_vortex)
    alpha_jump_strain = fit_power_law_tail(jumps_strain)
    print('Power-law tail exponent for jumps in Vortex: ' + str(alpha_jump_vortex))
    print('Power-law tail exponent for jumps in Strain: ' + str(alpha_jump_strain))
    print('Computing normalized velocity increments...')
    vx = dx_1 / dt_traj
    vy = dy_1 / dt_traj
    rms_v_vortex = np.sqrt(np.mean(vx[state_t == 1]**2 + vy[state_t == 1]**2))
    rms_v_strain = np.sqrt(np.mean(vx[state_t == 0]**2 + vy[state_t == 0]**2))
    print('RMS velocity in Vortex: ' + str(rms_v_vortex))
    print('RMS velocity in Strain: ' + str(rms_v_strain))
    vel_incs_norm = {}
    for lag in [1, 10, 50]:
        if lag >= len(vx):
            continue
        dvx = vx[lag:] - vx[:-lag]
        dvy = vy[lag:] - vy[:-lag]
        dv_mag = np.sqrt(dvx**2 + dvy**2)
        state_v = state_t[:-lag]
        vel_incs_norm['vortex_' + str(lag)] = dv_mag[state_v == 1] / rms_v_vortex
        vel_incs_norm['strain_' + str(lag)] = dv_mag[state_v == 0] / rms_v_strain
    print('Identifying large jumps and correlating with exit events...')
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
    print('Saving conditional statistics to disk...')
    np.savez('data/conditional_stats.npz', lags=lags, msd_vortex=msd_vortex, msd_strain=msd_strain, trapping_times=trapping_times, jumps_vortex=jumps_vortex, jumps_strain=jumps_strain, **vel_incs_norm, **cond_disp_pdfs)
    print('Conditional statistics saved to data/conditional_stats.npz')