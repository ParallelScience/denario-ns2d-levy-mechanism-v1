# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np

def compute_eulerian_stats():
    print('Loading Eulerian vorticity fields and snapshot times...')
    vorticity = np.load('/home/node/work/projects/ns2d_levy_v1/data/vorticity_snapshots.npy')
    snap_times = np.load('/home/node/work/projects/ns2d_levy_v1/data/snap_times.npy')
    n_snaps, N, _ = vorticity.shape
    L = 2 * np.pi
    print('Computing spectral derivatives and Okubo-Weiss parameter Q...')
    kx = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0
    Q_fields = np.zeros_like(vorticity)
    masks_vortex = np.zeros_like(vorticity, dtype=bool)
    masks_strain = np.zeros_like(vorticity, dtype=bool)
    E_k_all = []
    for i in range(n_snaps):
        w = vorticity[i]
        w_hat = np.fft.fft2(w)
        psi_hat = w_hat / K2
        psi_hat[0, 0] = 0.0
        u_hat = 1j * KY * psi_hat
        v_hat = -1j * KX * psi_hat
        dx_u_hat = 1j * KX * u_hat
        dy_u_hat = 1j * KY * u_hat
        dx_v_hat = 1j * KX * v_hat
        dy_v_hat = 1j * KY * v_hat
        dx_u = np.fft.ifft2(dx_u_hat).real
        dy_u = np.fft.ifft2(dy_u_hat).real
        dx_v = np.fft.ifft2(dx_v_hat).real
        dy_v = np.fft.ifft2(dy_v_hat).real
        s2 = 4 * dx_u**2 + (dx_v + dy_u)**2
        w2 = (dx_v - dy_u)**2
        Q = s2 - w2
        Q_fields[i] = Q
        masks_vortex[i] = Q < 0
        masks_strain[i] = Q > 0
        u_hat_norm = u_hat / (N**2)
        v_hat_norm = v_hat / (N**2)
        E_2D = 0.5 * (np.abs(u_hat_norm)**2 + np.abs(v_hat_norm)**2)
        K_mag = np.sqrt(K2)
        K_mag[0, 0] = 0.0
        K_mag_int = np.round(K_mag).astype(int)
        E_1D = np.bincount(K_mag_int.ravel(), weights=E_2D.ravel())
        E_k_all.append(E_1D)
        frac_vortex = np.mean(masks_vortex[i])
        frac_strain = np.mean(masks_strain[i])
        print('Snapshot ' + str(i) + ' (t=' + str(snap_times[i]) + '): Vortex fraction = ' + str(frac_vortex) + ', Strain fraction = ' + str(frac_strain))
    print('Mean of Q threshold used: 0.0')
    print('Standard deviation of Q threshold used: 0.0')
    print('Mean of Q field across all snapshots: ' + str(np.mean(Q_fields)))
    print('Standard deviation of Q field across all snapshots: ' + str(np.std(Q_fields)))
    print('Averaging energy spectrum over snapshots...')
    max_len = max(len(e) for e in E_k_all)
    E_k_mean = np.zeros(max_len)
    for e in E_k_all:
        E_k_mean[:len(e)] += e
    E_k_mean /= n_snaps
    k_vals = np.arange(max_len)
    print('Fitting spectral slopes...')
    mask_ls = (k_vals >= 1) & (k_vals <= 3)
    if np.sum(mask_ls) > 1:
        popt_ls = np.polyfit(np.log(k_vals[mask_ls]), np.log(E_k_mean[mask_ls]), 1)
        slope_ls = popt_ls[0]
    else:
        slope_ls = np.nan
    mask_ss = (k_vals >= 10) & (k_vals <= 40)
    if np.sum(mask_ss) > 1:
        popt_ss = np.polyfit(np.log(k_vals[mask_ss]), np.log(E_k_mean[mask_ss]), 1)
        slope_ss = popt_ss[0]
    else:
        slope_ss = np.nan
    print('Spectral slope (large-scale, k in [1, 3]): ' + str(slope_ls))
    print('Spectral slope (small-scale, k in [10, 40]): ' + str(slope_ss))
    print('Saving Eulerian statistics to disk...')
    np.savez('data/eulerian_stats.npz', k_vals=k_vals, E_k=E_k_mean, Q_fields=Q_fields, masks_vortex=masks_vortex, masks_strain=masks_strain)
    print('Eulerian statistics saved to data/eulerian_stats.npz')

if __name__ == '__main__':
    compute_eulerian_stats()