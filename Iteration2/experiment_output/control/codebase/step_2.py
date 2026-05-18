# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
from scipy.stats import linregress

def compute_spectrum_and_q():
    vorticity_snapshots = np.load('/home/node/work/projects/ns2d_levy_v1/data/vorticity_snapshots.npy')
    N = 256
    L = 2 * np.pi
    kx = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0
    E_k_all = []
    Q_all = []
    urms_all = []
    for i in range(vorticity_snapshots.shape[0]):
        omega = vorticity_snapshots[i]
        omega_hat = np.fft.fft2(omega)
        psi_hat = omega_hat / K2
        psi_hat[0, 0] = 0.0
        u_hat = 1j * KY * psi_hat
        v_hat = -1j * KX * psi_hat
        u = np.real(np.fft.ifft2(u_hat))
        v = np.real(np.fft.ifft2(v_hat))
        urms_all.append(np.sqrt(np.mean(u**2 + v**2)))
        E_hat_density = 0.5 * K2 * np.abs(psi_hat)**2 / (N**4)
        K_mag = np.sqrt(K2)
        K_int = np.round(K_mag).astype(int)
        E_k = np.bincount(K_int.ravel(), weights=E_hat_density.ravel())
        E_k_all.append(E_k)
        s_n_hat = -2 * KX * KY * psi_hat
        s_s_hat = (KX**2 - KY**2) * psi_hat
        s_n = np.real(np.fft.ifft2(s_n_hat))
        s_s = np.real(np.fft.ifft2(s_s_hat))
        s2 = s_n**2 + s_s**2
        Q = s2 - omega**2
        Q_all.append(Q)
    max_k = max([len(e) for e in E_k_all])
    E_k_avg = np.zeros(max_k)
    for e in E_k_all:
        E_k_avg[:len(e)] += e
    E_k_avg /= len(E_k_all)
    Q_all = np.array(Q_all)
    k_inv = np.array([1, 2])
    E_inv = E_k_avg[k_inv]
    slope_inv, _, _, _, _ = linregress(np.log(k_inv), np.log(E_inv))
    k_ens = np.arange(7, 61)
    E_ens = E_k_avg[k_ens]
    slope_ens, _, _, _, _ = linregress(np.log(k_ens), np.log(E_ens))
    beta_inv = -slope_inv
    xi_inv = (beta_inv - 1) / 2
    alpha_inv = 2 / xi_inv if xi_inv != 0 else np.inf
    beta_ens = -slope_ens
    xi_ens = (beta_ens - 1) / 2
    alpha_ens = 2 / xi_ens if xi_ens != 0 else np.inf
    print("--- Eulerian Field Analysis Results ---")
    print("Mean U_rms: " + str(np.round(np.mean(urms_all), 4)))
    print("Spectral peak at k = " + str(np.argmax(E_k_avg)))
    print("Fitted spectral index (inverse cascade, k<3): " + str(np.round(slope_inv, 4)))
    print("Fitted spectral index (enstrophy cascade, k>6): " + str(np.round(slope_ens, 4)))
    print("Theoretical alpha from inverse cascade: " + str(np.round(alpha_inv, 4)))
    print("Theoretical alpha from enstrophy cascade: " + str(np.round(alpha_ens, 4)))
    print("Okubo-Weiss parameter Q: mean = " + str(np.round(np.mean(Q_all), 4)) + ", std = " + str(np.round(np.std(Q_all), 4)))
    data_dir = "data/"
    np.save(os.path.join(data_dir, "E_k_avg.npy"), E_k_avg)
    np.save(os.path.join(data_dir, "Q_fields.npy"), Q_all)
    print("Averaged E(k) and Q fields saved to data/ directory.")

if __name__ == '__main__':
    compute_spectrum_and_q()