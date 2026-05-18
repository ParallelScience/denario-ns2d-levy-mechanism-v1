# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
from scipy.optimize import curve_fit

def neg_log_char_func(k, D, alpha):
    return D * k**alpha

if __name__ == '__main__':
    print("Loading trajectories...")
    traj_x = np.load('/home/node/work/projects/ns2d_levy_v1/data/traj_x.npy')
    traj_y = np.load('/home/node/work/projects/ns2d_levy_v1/data/traj_y.npy')
    t_traj = np.load('/home/node/work/projects/ns2d_levy_v1/data/t_traj.npy')
    n_steps, n_tracers = traj_x.shape
    dt = t_traj[1] - t_traj[0]
    L = 2 * np.pi
    print("Unwrapping trajectories...")
    dx = np.diff(traj_x, axis=0)
    dy = np.diff(traj_y, axis=0)
    dx = dx - L * np.round(dx / L)
    dy = dy - L * np.round(dy / L)
    unwrapped_x = np.zeros_like(traj_x)
    unwrapped_y = np.zeros_like(traj_y)
    unwrapped_x[0] = traj_x[0]
    unwrapped_y[0] = traj_y[0]
    unwrapped_x[1:] = traj_x[0] + np.cumsum(dx, axis=0)
    unwrapped_y[1:] = traj_y[0] + np.cumsum(dy, axis=0)
    np.save('data/unwrapped_traj_x.npy', unwrapped_x)
    np.save('data/unwrapped_traj_y.npy', unwrapped_y)
    print("Unwrapped trajectories saved to data/unwrapped_traj_x.npy and data/unwrapped_traj_y.npy")
    print("Computing MSD...")
    max_lag = n_steps // 2
    lags = np.arange(1, max_lag)
    msd = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        disp_x = unwrapped_x[lag:] - unwrapped_x[:-lag]
        disp_y = unwrapped_y[lag:] - unwrapped_y[:-lag]
        msd[i] = np.mean(disp_x**2 + disp_y**2)
    tau_vals = lags * dt
    fit_mask = (tau_vals > 10) & (tau_vals < tau_vals[-1]/2)
    popt, _ = np.polyfit(np.log(tau_vals[fit_mask]), np.log(msd[fit_mask]), 1)
    H = popt[0] / 2
    print("Estimated superdiffusive exponent H: " + str(H))
    print("Computing characteristic function and stable index alpha(tau)...")
    tau_indices = np.unique(np.logspace(0, np.log10(max_lag-1), 20).astype(int))
    alphas = []
    Ds = []
    selected_taus = []
    selected_tau_indices_for_dist = [tau_indices[0], tau_indices[len(tau_indices)//2], tau_indices[-1]]
    disp_dists_x = {}
    disp_dists_y = {}
    k_vals = np.logspace(-3, 1, 50)
    for lag in tau_indices:
        disp_x = (unwrapped_x[lag:] - unwrapped_x[:-lag]).flatten()
        disp_y = (unwrapped_y[lag:] - unwrapped_y[:-lag]).flatten()
        if lag in selected_tau_indices_for_dist:
            disp_dists_x[str(lag)] = disp_x
            disp_dists_y[str(lag)] = disp_y
        disp_all = np.concatenate([disp_x, disp_y])
        phi_k = np.zeros_like(k_vals, dtype=float)
        for i, k in enumerate(k_vals):
            phi_k[i] = np.mean(np.cos(k * disp_all))
        valid = (phi_k > 0.05) & (phi_k < 0.95)
        if np.sum(valid) > 3:
            try:
                popt_cf, _ = curve_fit(neg_log_char_func, k_vals[valid], -np.log(phi_k[valid]), p0=[1.0, 1.5], bounds=([0, 0], [np.inf, 2.0]))
                D_est, alpha_est = popt_cf
            except Exception:
                alpha_est = np.nan
                D_est = np.nan
        else:
            alpha_est = np.nan
            D_est = np.nan
        alphas.append(alpha_est)
        Ds.append(D_est)
        selected_taus.append(lag * dt)
    alphas = np.array(alphas)
    print("Computing Lagrangian velocities...")
    vx = (unwrapped_x[1:] - unwrapped_x[:-1]) / dt
    vy = (unwrapped_y[1:] - unwrapped_y[:-1]) / dt
    print("Computing velocity increments...")
    vel_inc_lags = [1, 10, 50, 100]
    vel_incs = {}
    for lag in vel_inc_lags:
        if lag >= len(vx):
            continue
        dvx = (vx[lag:] - vx[:-lag]).flatten()
        dvy = (vy[lag:] - vy[:-lag]).flatten()
        dv = np.concatenate([dvx, dvy])
        vel_incs[str(lag)] = dv
        abs_dv = np.abs(dv)
        threshold = np.percentile(abs_dv, 95)
        tail_dv = abs_dv[abs_dv > threshold]
        if len(tail_dv) > 0:
            beta = 1 + len(tail_dv) / np.sum(np.log(tail_dv / threshold))
            print("Velocity increment tail exponent for lag " + str(lag * dt) + ": " + str(beta))
    print("Computing Lagrangian velocity autocorrelation...")
    Rv = np.zeros(max_lag)
    vx_c = vx - np.mean(vx)
    vy_c = vy - np.mean(vy)
    var_v = np.mean(vx_c**2 + vy_c**2)
    for lag in range(max_lag):
        if lag == 0:
            Rv[lag] = 1.0
        else:
            cov = np.mean(vx_c[lag:] * vx_c[:-lag] + vy_c[lag:] * vy_c[:-lag])
            Rv[lag] = cov / var_v
    print("Saving results...")
    np.savez('data/lagrangian_stats.npz', tau_msd=tau_vals, msd=msd, H=H, tau_alpha=selected_taus, alphas=alphas, Ds=Ds, tau_Rv=np.arange(max_lag)*dt, Rv=Rv)
    np.savez('data/velocity_increments.npz', **vel_incs)
    disp_dists = {}
    for k, v in disp_dists_x.items():
        disp_dists["x_" + k] = v
    for k, v in disp_dists_y.items():
        disp_dists["y_" + k] = v
    np.savez('data/displacement_distributions.npz', **disp_dists)
    print("All computations completed and saved.")