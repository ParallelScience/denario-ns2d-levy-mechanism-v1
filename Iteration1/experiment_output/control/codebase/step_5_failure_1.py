# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os

def compute_q_criterion(vorticity):
    dx = 2 * np.pi / vorticity.shape[1]
    dy = 2 * np.pi / vorticity.shape[2]
    dudy, dudx = np.gradient(vorticity, axis=(2, 1))
    dvdy, dvdx = np.gradient(vorticity, axis=(2, 1))
    q = -0.5 * (dudx**2 + dvdy**2 + 2 * dudy * dvdx)
    return q

def main():
    data_dir = "data/"
    traj_x = np.load(os.path.join(data_dir, "traj_x.npy"))
    traj_y = np.load(os.path.join(data_dir, "traj_y.npy"))
    vorticity = np.load(os.path.join(data_dir, "vorticity_snapshots.npy"))
    snap_times = np.load(os.path.join(data_dir, "snap_times.npy"))
    
    q_fields = np.array([compute_q_criterion(vorticity[i]) for i in range(len(snap_times))])
    
    n_tracers = traj_x.shape[1]
    vortex_mask = np.zeros((len(snap_times), n_tracers), dtype=bool)
    
    for i in range(len(snap_times)):
        q_field = q_fields[i]
        x_idx = (traj_x[i * 100] / (2 * np.pi) * 256).astype(int) % 256
        y_idx = (traj_y[i * 100] / (2 * np.pi) * 256).astype(int) % 256
        vortex_mask[i] = q_field[x_idx, y_idx] > 0
        
    trapping_times = []
    for t in range(n_tracers):
        mask = vortex_mask[:, t]
        diffs = np.diff(mask.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        if len(starts) > 0 and len(ends) > 0:
            if starts[0] > ends[0]: ends = ends[1:]
            if len(starts) > len(ends): starts = starts[:len(ends)]
            trapping_times.extend(ends - starts)
            
    np.save(os.path.join(data_dir, "trapping_times.npy"), np.array(trapping_times))
    print("Saved trapping times to data/trapping_times.npy")

if __name__ == '__main__':
    main()