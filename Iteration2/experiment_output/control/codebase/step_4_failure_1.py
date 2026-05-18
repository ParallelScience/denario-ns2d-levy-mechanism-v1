# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
from scipy.stats import linregress

def compute_msd(x, y):
    dx = x - x[0]
    dy = y - y[0]
    return np.mean(dx**2 + dy**2, axis=1)

def compute_Ht(msd, t, window_size=100):
    H_t = np.full_like(msd, np.nan)
    log_t = np.log(t[1:])
    log_msd = np.log(msd[1:])
    for i in range(window_size//2, len(log_t) - window_size//2):
        start = i - window_size//2
        end = i + window_size//2
        valid = ~np.isnan(log_msd[start:end])
        if np.sum(valid) > 2:
            slope, _, _, _, _ = linregress(log_t[start:end][valid], log_msd[start:end][valid])
            H_t[i+1] = 0.5 * slope
    return H_t

def compute_Rv(vx, vy):
    n_steps = vx.shape[0]
    pad_len = 2 ** int(np.ceil(np.log2(2 * n_steps - 1)))
    n_tracers = vx.shape[1]
    batch_size = 1000
    Rv_sum = np.zeros(n_steps)
    for i in range(0, n_tracers, batch_size):
        vx_b = vx[:, i:i+batch_size]
        vy_b = vy[:, i:i+batch_size]
        vx_b = vx_b - np.mean(vx_b, axis=0)
        vy_b = vy_b - np.mean(vy_b, axis=0)
        fx = np.fft.fft(vx_b, n=pad_len, axis=0)
        fy = np.fft.fft(vy_b, n=pad_len, axis=0)
        rx = np.real(np.fft.ifft(fx * np.conj(fx), axis=0))[:n_steps, :]
        ry = np.real(np.fft.ifft(fy * np.conj(fy), axis=0))[:n_steps, :]
        counts = np.arange(n_steps, 0, -1)[:, None]
        rx /= counts
        ry /= counts
        Rv_sum += np.sum(rx + ry, axis=1)
    if Rv_sum[0] == 0:
        return np.zeros(n_steps)
    return Rv_sum / Rv_sum[0]

def decorrelation_time(Rv, dt):
    if np.all(np.isnan(Rv)):
        return np.nan
    neg_idx = np.where(Rv < 0)[0]
    if len(neg_idx) > 0:
        if neg_idx[0] == 0:
            return 0.0
        return np.trapz(Rv[:neg_idx[0]], dx=dt)
    else:
        return np.trapz(Rv, dx=dt)

def fit_tail_exponent(data, threshold_percentile=95):
    data = np.abs(data)
    threshold = np.percentile(data, threshold_percentile)
    tail_data = data[data > threshold]
    if len(tail_data) < 2:
        return np.nan
    tail_data = tail_data[tail_data > 0]
    if threshold <= 0 or len(tail_data) < 2:
        return np.nan
    alpha = 1.0 / np.mean(np.log(tail_data / threshold))
    return alpha

def compute_pdf(data, bins=200):
    if len(data) == 0:
        return np.array([]), np.array([])
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist

if __name__ == '__main__':
    data_dir = "data/"
    unwrapped_x = np.load(os.path.join(data_dir, "unwrapped_traj_x.npy"))
    unwrapped_y = np.load(os.path.join(data_dir, "unwrapped_traj_y.npy"))
    t_traj = np.load(os.path.join(data_dir, "t_traj.npy"))
    strain_percentage = np.load(os.path.join(data_dir, "strain_percentage.npy"))
    idx_strain = strain_percentage > 70
    idx_vortex = strain_percentage < 30
    if np.sum(idx_strain) > 0:
        msd_strain = compute_msd(unwrapped_x[:, idx_strain], unwrapped_y[:, idx_strain])
        Ht_strain = compute_Ht(msd_strain, t_traj)
    else:
        msd_strain = np.full(unwrapped_x.shape[0], np.nan)
        Ht_strain = np.full(unwrapped_x.shape[0], np.nan)
    if np.sum(idx_vortex) > 0:
        msd_vortex = compute_msd(unwrapped_x[:, idx_vortex], unwrapped_y[:, idx_vortex])
        Ht_vortex = compute_Ht(msd_vortex, t_traj)
    else:
        msd_vortex = np.full(unwrapped_x.shape[0], np.nan)
        Ht_vortex = np.full(unwrapped_x.shape[0], np.nan)
    late_regime_start = len(t_traj) * 2 // 3
    avg_H_strain = np.nanmean(Ht_strain[late_regime_start:]) if np.sum(idx_strain) > 0 else np.nan
    avg_H_vortex = np.nanmean(Ht_vortex[late_regime_start:]) if np.sum(idx_vortex) > 0 else np.nan
    dt = t_traj[1] - t_traj[0]
    vx = np.diff(unwrapped_x, axis=0) / dt
    vy = np.diff(unwrapped_y, axis=0) / dt
    Rv_full = compute_Rv(vx, vy)
    dec_time_full = decorrelation_time(Rv_full, dt)
    if np.sum(idx_strain) > 0:
        Rv_strain = compute_Rv(vx[:, idx_strain], vy[:, idx_strain])
        dec_time_strain = decorrelation_time(Rv_strain, dt)
    else:
        Rv_strain = np.full(vx.shape[0], np.nan)
        dec_time_strain = np.nan
    if np.sum(idx_vortex) > 0:
        Rv_vortex = compute_Rv(vx[:, idx_vortex], vy[:, idx_vortex])
        dec_time_vortex = decorrelation_time(Rv_vortex, dt)
    else:
        Rv_vortex = np.full(vx.shape[0], np.nan)
        dec_time_vortex = np.nan
    lags = [10, 100, 500]
    pdfs = {}
    tail_exponents = {}
    for lag in lags:
        dvx = vx[lag:] - vx[:-lag]
        dvx_full = dvx.flatten()
        alpha_full = fit_tail_exponent(dvx_full)
        bc_full, hist_full = compute_pdf(dvx_full)
        if np.sum(idx_strain) > 0:
            dvx_strain = dvx[:, idx_strain].flatten()
            alpha_strain = fit_tail_exponent(dvx_strain)
            bc_strain, hist_strain = compute_pdf(dvx_strain)
        else:
            alpha_strain = np.nan
            bc_strain, hist_strain = np.array([]), np.array([])
        if np.sum(idx_vortex) > 0:
            dvx_vortex = dvx[:, idx_vortex].flatten()
            alpha_vortex = fit_tail_exponent(dvx_vortex)
            bc_vortex, hist_vortex = compute_pdf(dvx_vortex)
        else:
            alpha_vortex = np.nan
            bc_vortex, hist_vortex = np.array([]), np.array([])
        tail_exponents[lag] = {'full': alpha_full, 'strain': alpha_strain, 'vortex': alpha_vortex}
        pdfs[str(lag)] = {'bc_full': bc_full, 'hist_full': hist_full, 'bc_strain': bc_strain, 'hist_strain': hist_strain, 'bc_vortex': bc_vortex, 'hist_vortex': hist_vortex}
    print("--- Sub-population Analysis Results ---")
    print("Total tracers: " + str(len(strain_percentage)))
    print("Strain-dominated (>70%): " + str(np.sum(idx_strain)))
    print("Vortex-dominated (<30%): " + str(np.sum(idx_vortex)))
    print("Average H(t) in late regime - Strain-dominated: " + str(np.round(avg_H_strain, 4)))
    print("Average H(t) in late regime - Vortex-dominated: " + str(np.round(avg_H_vortex, 4)))
    print("Decorrelation time - Full ensemble: " + str(np.round(dec_time_full, 4)))
    print("Decorrelation time - Strain-dominated: " + str(np.round(dec_time_strain, 4)))
    print("Decorrelation time - Vortex-dominated: " + str(np.round(dec_time_vortex, 4)))
    for lag in lags:
        alpha_full = tail_exponents[lag]['full']
        alpha_strain = tail_exponents[lag]['strain']
        alpha_vortex = tail_exponents[lag]['vortex']
        print("Velocity increment tail exponent (lag=" + str(lag) + ") - Full: " + str(np.round(alpha_full, 4)) + ", Strain: " + str(np.round(alpha_strain, 4)) + ", Vortex: " + str(np.round(alpha_vortex, 4)))
    np.save(os.path.join(data_dir, "msd_strain.npy"), msd_strain)
    np.save(os.path.join(data_dir, "msd_vortex.npy"), msd_vortex)
    np.save(os.path.join(data_dir, "Ht_strain.npy"), Ht_strain)
    np.save(os.path.join(data_dir, "Ht_vortex.npy"), Ht_vortex)
    np.save(os.path.join(data_dir, "vx.npy"), vx)
    np.save(os.path.join(data_dir, "vy.npy"), vy)
    np.save(os.path.join(data_dir, "Rv_full.npy"), Rv_full)
    np.save(os.path.join(data_dir, "Rv_strain.npy"), Rv_strain)
    np.save(os.path.join(data_dir, "Rv_vortex.npy"), Rv_vortex)
    save_dict = {}
    for lag_key, d in pdfs.items():
        for key, val in d.items():
            save_dict["lag_" + lag_key + "_" + key] = val
    np.savez(os.path.join(data_dir, "velocity_inc_pdfs.npz"), **save_dict)
    print("Sub-population MSDs, H(t), Lagrangian velocities, Rv(tau), and velocity increment PDFs saved to data/ directory.")