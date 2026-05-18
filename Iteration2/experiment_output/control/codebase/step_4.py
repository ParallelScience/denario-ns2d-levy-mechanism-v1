# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
from scipy.stats import linregress

def compute_msd_and_h(unwrapped_x, unwrapped_y, t_traj, window_size=100):
    dx = unwrapped_x - unwrapped_x[0]
    dy = unwrapped_y - unwrapped_y[0]
    msd = np.mean(dx**2 + dy**2, axis=1)
    H_t = np.zeros_like(msd)
    H_t[:] = np.nan
    log_t = np.log(t_traj[1:])
    log_msd = np.log(msd[1:])
    for i in range(window_size//2, len(log_t) - window_size//2):
        start = i - window_size//2
        end = i + window_size//2
        slope, _, _, _, _ = linregress(log_t[start:end], log_msd[start:end])
        H_t[i+1] = 0.5 * slope
    return msd, H_t

def compute_autocorr(vx, vy, max_lag=500):
    if vx.size == 0 or vy.size == 0:
        return np.full(max_lag, np.nan)
    autocorr = np.zeros(max_lag)
    vx_mean = np.mean(vx)
    vy_mean = np.mean(vy)
    vx_c = vx - vx_mean
    vy_c = vy - vy_mean
    var = np.mean(vx_c**2 + vy_c**2)
    if var == 0:
        return np.full(max_lag, np.nan)
    for lag in range(max_lag):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            cov = np.mean(vx_c[lag:] * vx_c[:-lag] + vy_c[lag:] * vy_c[:-lag])
            autocorr[lag] = cov / var
    return autocorr

def fit_power_law_tail(data, percentile=95):
    if data.size == 0:
        return np.nan
    threshold = np.percentile(data, percentile)
    tail_data = data[data > threshold]
    if len(tail_data) > 0 and threshold > 0:
        mu = 1 + len(tail_data) / np.sum(np.log(tail_data / threshold))
        return mu
    return np.nan

def decorr_time(Rv, dt):
    if np.all(np.isnan(Rv)):
        return np.nan
    idx = np.where(Rv < np.exp(-1))[0]
    if len(idx) > 0:
        return idx[0] * dt
    return np.nan

if __name__ == '__main__':
    data_dir = "data/"
    unwrapped_x = np.load(os.path.join(data_dir, "unwrapped_traj_x.npy"))
    unwrapped_y = np.load(os.path.join(data_dir, "unwrapped_traj_y.npy"))
    t_traj = np.load('/home/node/work/projects/ns2d_levy_v1/data/t_traj.npy')
    strain_percentage = np.load(os.path.join(data_dir, "strain_percentage.npy"))
    strain_dom_idx = np.where(strain_percentage > 70)[0]
    vortex_dom_idx = np.where(strain_percentage < 30)[0]
    late_regime_start = len(t_traj) * 2 // 3
    if len(strain_dom_idx) > 0:
        msd_strain, H_t_strain = compute_msd_and_h(unwrapped_x[:, strain_dom_idx], unwrapped_y[:, strain_dom_idx], t_traj)
        avg_H_strain = np.nanmean(H_t_strain[late_regime_start:])
    else:
        msd_strain = np.array([])
        H_t_strain = np.array([])
        avg_H_strain = np.nan
    if len(vortex_dom_idx) > 0:
        msd_vortex, H_t_vortex = compute_msd_and_h(unwrapped_x[:, vortex_dom_idx], unwrapped_y[:, vortex_dom_idx], t_traj)
        avg_H_vortex = np.nanmean(H_t_vortex[late_regime_start:])
    else:
        msd_vortex = np.array([])
        H_t_vortex = np.array([])
        avg_H_vortex = np.nan
    dt = t_traj[1] - t_traj[0]
    vx = np.diff(unwrapped_x, axis=0) / dt
    vy = np.diff(unwrapped_y, axis=0) / dt
    vx_strain = vx[:, strain_dom_idx] if len(strain_dom_idx) > 0 else np.array([])
    vy_strain = vy[:, strain_dom_idx] if len(strain_dom_idx) > 0 else np.array([])
    vx_vortex = vx[:, vortex_dom_idx] if len(vortex_dom_idx) > 0 else np.array([])
    vy_vortex = vy[:, vortex_dom_idx] if len(vortex_dom_idx) > 0 else np.array([])
    max_lag = 500
    Rv_full = compute_autocorr(vx, vy, max_lag)
    Rv_strain = compute_autocorr(vx_strain, vy_strain, max_lag)
    Rv_vortex = compute_autocorr(vx_vortex, vy_vortex, max_lag)
    dec_full = decorr_time(Rv_full, dt)
    dec_strain = decorr_time(Rv_strain, dt)
    dec_vortex = decorr_time(Rv_vortex, dt)
    lags = [10, 100, 500]
    alpha_full_dict = {}
    alpha_strain_dict = {}
    alpha_vortex_dict = {}
    save_dict = {}
    for lag in lags:
        dvx_full = np.abs((vx[lag:] - vx[:-lag]).flatten())
        alpha_full_dict[lag] = fit_power_law_tail(dvx_full)
        save_dict['dvx_full_lag_' + str(lag)] = dvx_full
        if len(strain_dom_idx) > 0:
            dvx_strain = np.abs((vx_strain[lag:] - vx_strain[:-lag]).flatten())
            alpha_strain_dict[lag] = fit_power_law_tail(dvx_strain)
            save_dict['dvx_strain_lag_' + str(lag)] = dvx_strain
        else:
            alpha_strain_dict[lag] = np.nan
        if len(vortex_dom_idx) > 0:
            dvx_vortex = np.abs((vx_vortex[lag:] - vx_vortex[:-lag]).flatten())
            alpha_vortex_dict[lag] = fit_power_law_tail(dvx_vortex)
            save_dict['dvx_vortex_lag_' + str(lag)] = dvx_vortex
        else:
            alpha_vortex_dict[lag] = np.nan
    print("--- Sub-population Analysis Results ---")
    print("Total tracers: " + str(len(strain_percentage)))
    print("Strain-dominated (>70%): " + str(len(strain_dom_idx)))
    print("Vortex-dominated (<30%): " + str(len(vortex_dom_idx)))
    print("Average H(t) in late regime - Strain-dominated: " + str(np.round(avg_H_strain, 4)))
    print("Average H(t) in late regime - Vortex-dominated: " + str(np.round(avg_H_vortex, 4)))
    print("Decorrelation time (1/e) - Full ensemble: " + str(np.round(dec_full, 2)))
    print("Decorrelation time (1/e) - Strain-dominated: " + str(np.round(dec_strain, 2)))
    print("Decorrelation time (1/e) - Vortex-dominated: " + str(np.round(dec_vortex, 2)))
    for lag in lags:
        print("Velocity increment tail exponent (lag=" + str(lag) + ") - Full: " + str(np.round(alpha_full_dict[lag], 4)) + ", Strain: " + str(np.round(alpha_strain_dict[lag], 4)) + ", Vortex: " + str(np.round(alpha_vortex_dict[lag], 4)))
    np.save(os.path.join(data_dir, "msd_strain.npy"), msd_strain)
    np.save(os.path.join(data_dir, "H_t_strain.npy"), H_t_strain)
    np.save(os.path.join(data_dir, "msd_vortex.npy"), msd_vortex)
    np.save(os.path.join(data_dir, "H_t_vortex.npy"), H_t_vortex)
    np.save(os.path.join(data_dir, "vx.npy"), vx)
    np.save(os.path.join(data_dir, "vy.npy"), vy)
    np.save(os.path.join(data_dir, "Rv_full.npy"), Rv_full)
    np.save(os.path.join(data_dir, "Rv_strain.npy"), Rv_strain)
    np.save(os.path.join(data_dir, "Rv_vortex.npy"), Rv_vortex)
    np.savez(os.path.join(data_dir, "vel_inc_pdfs.npz"), **save_dict)
    print("Sub-population MSDs, H(t), Lagrangian velocities, Rv(tau), and velocity increment PDFs saved to data/ directory.")