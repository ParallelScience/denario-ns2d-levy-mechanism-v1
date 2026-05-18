# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
from scipy.stats import linregress

def fit_power_law_tail(data, percentile=90):
    if data.size == 0:
        return np.nan
    threshold = np.percentile(data, percentile)
    tail_data = data[data > threshold]
    if len(tail_data) > 0:
        x_min = np.min(tail_data)
        log_vals = np.log(tail_data / x_min)
        sum_log = np.sum(log_vals)
        if sum_log > 0:
            mu = 1 + len(tail_data) / sum_log
            return mu
    return np.nan

def phase_randomize(v):
    V = np.fft.rfft(v, axis=0)
    phases = np.random.uniform(0, 2*np.pi, size=V.shape)
    phases[0, :] = 0.0
    if v.shape[0] % 2 == 0:
        phases[-1, :] = 0.0
    V_surr = V * np.exp(1j * phases)
    v_surr = np.fft.irfft(V_surr, n=v.shape[0], axis=0)
    return v_surr

if __name__ == '__main__':
    data_dir = "data/"
    trapping_durations = np.load(os.path.join(data_dir, "trapping_durations.npy"))
    jump_sizes = np.load(os.path.join(data_dir, "jump_sizes.npy"))
    vx = np.load(os.path.join(data_dir, "vx.npy"))
    vy = np.load(os.path.join(data_dir, "vy.npy"))
    t_traj = np.load("/home/node/work/projects/ns2d_levy_v1/data/t_traj.npy")
    mu_trap = fit_power_law_tail(trapping_durations, percentile=90)
    mu_jump = fit_power_law_tail(jump_sizes, percentile=90)
    print("--- Power-Law Fits ---")
    print("Trapping time exponent mu: " + str(np.round(mu_trap, 4)) + " (CTRW threshold: mu < 2)")
    print("Jump size exponent mu: " + str(np.round(mu_jump, 4)))
    np.random.seed(42)
    vx_surr = phase_randomize(vx)
    vy_surr = phase_randomize(vy)
    dt = t_traj[1] - t_traj[0]
    dx_surr = vx_surr * dt
    dy_surr = vy_surr * dt
    x_surr = np.zeros((vx.shape[0] + 1, vx.shape[1]))
    y_surr = np.zeros((vy.shape[0] + 1, vy.shape[1]))
    x_surr[1:] = np.cumsum(dx_surr, axis=0)
    y_surr[1:] = np.cumsum(dy_surr, axis=0)
    msd_surr = np.mean(x_surr**2 + y_surr**2, axis=1)
    window_size = 100
    H_t_surr = np.zeros_like(msd_surr)
    H_t_surr[:] = np.nan
    log_t = np.log(t_traj[1:])
    log_msd_surr = np.log(msd_surr[1:])
    for i in range(window_size//2, len(log_t) - window_size//2):
        start = i - window_size//2
        end = i + window_size//2
        slope, _, _, _, _ = linregress(log_t[start:end], log_msd_surr[start:end])
        H_t_surr[i+1] = 0.5 * slope
    late_regime_start = len(H_t_surr) * 2 // 3
    avg_H_surr = np.nanmean(H_t_surr[late_regime_start:])
    print("--- Surrogate Trajectory Analysis ---")
    print("Final Surrogate MSD at t=" + str(np.round(t_traj[-1], 1)) + ": " + str(np.round(msd_surr[-1], 4)))
    print("Average Surrogate H(t) in late regime: " + str(np.round(avg_H_surr, 4)))
    np.save(os.path.join(data_dir, "mu_exponents.npy"), np.array([mu_trap, mu_jump]))
    np.save(os.path.join(data_dir, "msd_surr.npy"), msd_surr)
    np.save(os.path.join(data_dir, "H_t_surr.npy"), H_t_surr)
    print("Fitted exponents and surrogate MSD saved to data/ directory.")