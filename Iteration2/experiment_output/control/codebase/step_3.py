# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
from scipy.ndimage import map_coordinates

def interpolate_q_and_extract_events():
    data_dir = "data/"
    unwrapped_x_path = os.path.join(data_dir, "unwrapped_traj_x.npy")
    unwrapped_y_path = os.path.join(data_dir, "unwrapped_traj_y.npy")
    Q_fields_path = os.path.join(data_dir, "Q_fields.npy")
    t_traj_path = '/home/node/work/projects/ns2d_levy_v1/data/t_traj.npy'
    snap_times_path = '/home/node/work/projects/ns2d_levy_v1/data/snap_times.npy'
    unwrapped_x = np.load(unwrapped_x_path)
    unwrapped_y = np.load(unwrapped_y_path)
    Q_fields = np.load(Q_fields_path)
    t_traj = np.load(t_traj_path)
    snap_times = np.load(snap_times_path)
    N = 256
    L = 2 * np.pi
    wrapped_x = unwrapped_x % L
    wrapped_y = unwrapped_y % L
    Q_interp = np.zeros_like(unwrapped_x)
    for i, t in enumerate(t_traj):
        if t <= snap_times[0]:
            k = 0
            w = 0.0
        elif t >= snap_times[-1]:
            k = len(snap_times) - 2
            w = 1.0
        else:
            idx = np.searchsorted(snap_times, t, side='right')
            k = idx - 1
            w = (t - snap_times[k]) / (snap_times[k+1] - snap_times[k])
        x_idx = (wrapped_x[i] / L) * N
        y_idx = (wrapped_y[i] / L) * N
        Q_k = map_coordinates(Q_fields[k], [y_idx, x_idx], mode='wrap', order=1)
        Q_kp1 = map_coordinates(Q_fields[k+1], [y_idx, x_idx], mode='wrap', order=1)
        Q_interp[i] = (1 - w) * Q_k + w * Q_kp1
    state_strain = Q_interp > 0
    vortex_state = ~state_strain
    strain_percentage = np.mean(state_strain, axis=0) * 100
    trapping_durations = []
    jump_sizes = []
    dt_traj = t_traj[1] - t_traj[0]
    n_steps, n_tracers = state_strain.shape
    for i in range(n_tracers):
        v_state = vortex_state[:, i]
        v_padded = np.pad(v_state.astype(int), (1, 1), 'constant', constant_values=0)
        diff = np.diff(v_padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        durations = (ends - starts) * dt_traj
        trapping_durations.extend(durations)
        s_state = state_strain[:, i]
        s_padded = np.pad(s_state.astype(int), (1, 1), 'constant', constant_values=0)
        diff_s = np.diff(s_padded)
        starts_s = np.where(diff_s == 1)[0]
        ends_s = np.where(diff_s == -1)[0]
        for st, en in zip(starts_s, ends_s):
            en_idx = min(en, n_steps - 1)
            dx = unwrapped_x[en_idx, i] - unwrapped_x[st, i]
            dy = unwrapped_y[en_idx, i] - unwrapped_y[st, i]
            jump = np.sqrt(dx**2 + dy**2)
            if jump > 0:
                jump_sizes.append(jump)
    trapping_durations = np.array(trapping_durations)
    jump_sizes = np.array(jump_sizes)
    print("--- State Classification and Event Statistics ---")
    print("Strain percentage: mean = " + str(np.round(np.mean(strain_percentage), 2)) + "%" + ", std = " + str(np.round(np.std(strain_percentage), 2)) + "%" + ", median = " + str(np.round(np.median(strain_percentage), 2)) + "%")
    print("Trapping durations: count = " + str(len(trapping_durations)) + ", mean = " + str(np.round(np.mean(trapping_durations), 2)) + ", max = " + str(np.round(np.max(trapping_durations), 2)))
    print("Jump sizes: count = " + str(len(jump_sizes)) + ", mean = " + str(np.round(np.mean(jump_sizes), 4)) + ", max = " + str(np.round(np.max(jump_sizes), 4)))
    np.save(os.path.join(data_dir, "Q_interp.npy"), Q_interp)
    np.save(os.path.join(data_dir, "strain_percentage.npy"), strain_percentage)
    np.save(os.path.join(data_dir, "trapping_durations.npy"), trapping_durations)
    np.save(os.path.join(data_dir, "jump_sizes.npy"), jump_sizes)
    print("Interpolated Q values, Strain percentages, trapping durations, and jump sizes saved to data/ directory.")

if __name__ == '__main__':
    interpolate_q_and_extract_events()