# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
import pickle

def linear_fit(x, a, b):
    return a * x + b

def unwrap_trajectory(traj, L):
    diffs = np.diff(traj, axis=0)
    diffs = diffs - L * np.round(diffs / L)
    unwrapped = np.zeros_like(traj)
    unwrapped[0] = traj[0]
    unwrapped[1:] = traj[0] + np.cumsum(diffs, axis=0)
    return unwrapped

def compute_msd(x, y):
    dx = x - x[0]
    dy = y - y[0]
    msd = np.mean(dx**2 + dy**2, axis=1)
    return msd

def compute_vacf(vx, vy):
    N = vx.shape[0]
    vx_pad = np.pad(vx, ((0, N), (0, 0)), mode='constant')
    vy_pad = np.pad(vy, ((0, N), (0, 0)), mode='constant')
    fx = np.fft.fft(vx_pad, axis=0)
    fy = np.fft.fft(vy_pad, axis=0)
    S = np.abs(fx)**2 + np.abs(fy)**2
    R = np.fft.ifft(S, axis=0).real
    overlap = np.arange(N, 0, -1)[:, None]
    R = R[:N, :] / overlap
    vacf = np.mean(R, axis=1)
    vacf = vacf / vacf[0]
    return vacf

def fit_alpha(data):
    std_disp = np.std(data)
    if std_disp == 0:
        return np.nan, np.nan
    k_test = np.logspace(-3, 1, 100) / std_disp
    cf = np.zeros_like(k_test, dtype=complex)
    if len(data) > 100000:
        data_sub = np.random.choice(data, 100000, replace=False)
    else:
        data_sub = data
    for i, k in enumerate(k_test):
        cf[i] = np.mean(np.exp(1j * k * data_sub))
    cf_mag = np.abs(cf)
    valid = (cf_mag > 0.5) & (cf_mag < 0.99)
    if np.sum(valid) > 5:
        y_fit = np.log(-np.log(cf_mag[valid]))
        x_fit = np.log(k_test[valid])
        popt, pcov = curve_fit(linear_fit, x_fit, y_fit)
        return popt[0], np.sqrt(pcov[0, 0])
    else:
        return np.nan, np.nan

if __name__ == '__main__':
    traj_x_path = '/home/node/work/projects/ns2d_levy_v1/data/traj_x.npy'
    traj_y_path = '/home/node/work/projects/ns2d_levy_v1/data/traj_y.npy'
    t_traj_path = '/home/node/work/projects/ns2d_levy_v1/data/t_traj.npy'
    data_dir = 'data/'
    traj_x = np.load(traj_x_path)
    traj_y = np.load(traj_y_path)
    t_traj = np.load(t_traj_path)
    L = 2 * np.pi
    unwrapped_x = unwrap_trajectory(traj_x, L)
    unwrapped_y = unwrap_trajectory(traj_y, L)
    msd = compute_msd(unwrapped_x, unwrapped_y)
    fit_mask = (t_traj > 10) & (msd > 0)
    popt, pcov = curve_fit(linear_fit, np.log(t_traj[fit_mask]), np.log(msd[fit_mask]))
    H = popt[0] / 2
    H_err = np.sqrt(pcov[0, 0]) / 2
    vx = np.gradient(unwrapped_x, t_traj, axis=0)
    vy = np.gradient(unwrapped_y, t_traj, axis=0)
    vacf = compute_vacf(vx, vy)
    T_L = np.trapz(vacf, t_traj)
    mean_vx = np.mean(vx, axis=1, keepdims=True)
    mean_vy = np.mean(vy, axis=1, keepdims=True)
    vx_sub = vx - mean_vx
    vy_sub = vy - mean_vy
    x_sub = unwrapped_x[0] + cumulative_trapezoid(vx_sub, t_traj, axis=0, initial=0)
    y_sub = unwrapped_y[0] + cumulative_trapezoid(vy_sub, t_traj, axis=0, initial=0)
    msd_sub = compute_msd(x_sub, y_sub)
    popt_sub, pcov_sub = curve_fit(linear_fit, np.log(t_traj[fit_mask]), np.log(msd_sub[fit_mask]))
    H_sub = popt_sub[0] / 2
    H_sub_err = np.sqrt(pcov_sub[0, 0]) / 2
    tau_indices = [25, 125, 250, 500, 1000]
    disp_pdfs = {}
    v_inc_pdfs = {}
    alpha_values = {}
    for lag in tau_indices:
        dx = unwrapped_x[lag:] - unwrapped_x[:-lag]
        dy = unwrapped_y[lag:] - unwrapped_y[:-lag]
        displacements = np.concatenate([dx.flatten(), dy.flatten()])
        hist, bin_edges = np.histogram(displacements, bins=1000, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        disp_pdfs[lag] = {'bin_centers': bin_centers, 'hist': hist, 'tau': t_traj[lag]}
        dvx = vx[lag:] - vx[:-lag]
        dvy = vy[lag:] - vy[:-lag]
        dv = np.concatenate([dvx.flatten(), dvy.flatten()])
        hist_v, bin_edges_v = np.histogram(dv, bins=1000, density=True)
        bin_centers_v = (bin_edges_v[:-1] + bin_edges_v[1:]) / 2
        v_inc_pdfs[lag] = {'bin_centers': bin_centers_v, 'hist': hist_v, 'tau': t_traj[lag]}
        alpha, alpha_err = fit_alpha(displacements)
        alpha_values[lag] = {'tau': t_traj[lag], 'alpha': alpha, 'alpha_err': alpha_err}
    np.savez(os.path.join(data_dir, 'step1_lagrangian_stats.npz'), t_traj=t_traj, msd=msd, H=H, H_err=H_err, vacf=vacf, T_L=T_L, msd_sub=msd_sub, H_sub=H_sub, H_sub_err=H_sub_err, tau_indices=tau_indices)
    with open(os.path.join(data_dir, 'step1_pdfs.pkl'), 'wb') as f:
        pickle.dump({'disp_pdfs': disp_pdfs, 'v_inc_pdfs': v_inc_pdfs, 'alpha_values': alpha_values}, f)
    np.save(os.path.join(data_dir, 'unwrapped_x.npy'), unwrapped_x)
    np.save(os.path.join(data_dir, 'unwrapped_y.npy'), unwrapped_y)
    np.save(os.path.join(data_dir, 'vx.npy'), vx)
    np.save(os.path.join(data_dir, 'vy.npy'), vy)