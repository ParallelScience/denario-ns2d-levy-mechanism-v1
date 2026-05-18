# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
from scipy.stats import kurtosis
from scipy.stats import linregress

def unwrap_trajectories(traj, L=2*np.pi):
    """
    Unwrap trajectories in a periodic domain [0, L] to remove artificial jumps.
    
    Parameters:
    traj (numpy.ndarray): Array of shape (n_steps, n_tracers) containing the wrapped trajectories in length units.
    L (float): The size of the periodic domain in length units. Default is 2*pi.
    
    Returns:
    numpy.ndarray: Array of the same shape as traj containing the unwrapped trajectories in length units.
    """
    diffs = np.diff(traj, axis=0)
    diffs[diffs > L/2] -= L
    diffs[diffs < -L/2] += L
    
    unwrapped = np.zeros_like(traj)
    unwrapped[0] = traj[0]
    unwrapped[1:] = traj[0] + np.cumsum(diffs, axis=0)
    return unwrapped

if __name__ == '__main__':
    traj_x_path = '/home/node/work/projects/ns2d_levy_v1/data/traj_x.npy'
    traj_y_path = '/home/node/work/projects/ns2d_levy_v1/data/traj_y.npy'
    t_traj_path = '/home/node/work/projects/ns2d_levy_v1/data/t_traj.npy'
    
    traj_x = np.load(traj_x_path)
    traj_y = np.load(traj_y_path)
    t_traj = np.load(t_traj_path)
    
    L = 2 * np.pi
    unwrapped_x = unwrap_trajectories(traj_x, L)
    unwrapped_y = unwrap_trajectories(traj_y, L)
    
    dx = unwrapped_x - unwrapped_x[0]
    dy = unwrapped_y - unwrapped_y[0]
    msd = np.mean(dx**2 + dy**2, axis=1)
    
    window_size = 100
    H_t = np.zeros_like(msd)
    H_t[:] = np.nan
    
    log_t = np.log(t_traj[1:])
    log_msd = np.log(msd[1:])
    
    for i in range(window_size//2, len(log_t) - window_size//2):
        start = i - window_size//2
        end = i + window_size//2
        slope, _, _, _, _ = linregress(log_t[start:end], log_msd[start:end])
        H_t[i+1] = 0.5 * slope
        
    late_regime_start = len(H_t) * 2 // 3
    avg_H_late = np.nanmean(H_t[late_regime_start:])
    
    lags = [10, 100, 500]
    displacements = {}
    kurtosis_vals = {}
    
    for lag in lags:
        dx_lag = unwrapped_x[lag:] - unwrapped_x[:-lag]
        dx_lag_flat = dx_lag.flatten()
        displacements['lag_' + str(lag)] = dx_lag_flat
        
        kurt = kurtosis(dx_lag_flat, fisher=True)
        kurtosis_vals[lag] = kurt
        
    print("--- Trajectory Analysis Results ---")
    print("Final MSD at t=" + str(np.round(t_traj[-1], 1)) + ": " + str(np.round(msd[-1], 4)))
    print("Average H(t) in late regime (t > " + str(np.round(t_traj[late_regime_start], 1)) + "): " + str(np.round(avg_H_late, 4)))
    for lag in lags:
        tau = lag * (t_traj[1] - t_traj[0])
        print("Excess kurtosis of delta x at lag tau=" + str(np.round(tau, 1)) + " (" + str(lag) + " steps): " + str(np.round(kurtosis_vals[lag], 4)))
        
    data_dir = "data/"
    
    np.save(os.path.join(data_dir, "unwrapped_traj_x.npy"), unwrapped_x)
    np.save(os.path.join(data_dir, "unwrapped_traj_y.npy"), unwrapped_y)
    np.save(os.path.join(data_dir, "msd.npy"), msd)
    np.save(os.path.join(data_dir, "H_t.npy"), H_t)
    
    for lag in lags:
        np.save(os.path.join(data_dir, "displacements_dx_lag_" + str(lag) + ".npy"), displacements['lag_' + str(lag)])
        
    print("Data saved to data/ directory.")