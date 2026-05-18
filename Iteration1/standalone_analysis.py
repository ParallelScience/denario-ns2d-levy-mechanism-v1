"""
Standalone iteration 1 analysis: Surrogate test + Conditional R_v + Trapping statistics
All computed from raw NS2D simulation data.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq
import os, time, json

t0 = time.time()
DATA = '/home/node/work/projects/ns2d_levy_v1/data'
OUT  = '/home/node/work/projects/ns2d_levy_v1/Iteration1/experiment_output/control/data'
os.makedirs(OUT, exist_ok=True)

meta = json.load(open(f'{DATA}/meta.json'))
N = meta['N']; L = meta['L']; dt_save = meta['save_every'] * meta['dt']
print(f"N={N}, L={L:.3f}, dt_save={dt_save}")

# ── Load trajectories ────────────────────────────────────────────────────────
traj_x = np.load(f'{DATA}/traj_x.npy')   # (1500, 8000) float32
traj_y = np.load(f'{DATA}/traj_y.npy')
t_traj = np.load(f'{DATA}/t_traj.npy')   # (1500,) float32
n_t, n_tr = traj_x.shape
print(f"Trajectories: {n_t} x {n_tr}, t=[{t_traj[0]},{t_traj[-1]}]")

# ── Unwrap periodic trajectories ─────────────────────────────────────────────
def unwrap_periodic(traj, L=2*np.pi):
    diff = np.diff(traj.astype(np.float64), axis=0)
    diff -= L * np.round(diff / L)
    out = np.zeros_like(traj, dtype=np.float64)
    out[0] = traj[0]
    out[1:] = traj[0] + np.cumsum(diff, axis=0)
    return out

print("Unwrapping trajectories...")
ux_traj = unwrap_periodic(traj_x, L)
uy_traj = unwrap_periodic(traj_y, L)

# ── Lagrangian velocities ────────────────────────────────────────────────────
vx = np.diff(ux_traj, axis=0) / dt_save   # (1499, 8000)
vy = np.diff(uy_traj, axis=0) / dt_save
t_vel = 0.5 * (t_traj[:-1] + t_traj[1:])  # midpoint times

# ── MSD ──────────────────────────────────────────────────────────────────────
print("Computing MSD...")
dx = ux_traj - ux_traj[0:1]
dy = uy_traj - uy_traj[0:1]
msd = np.mean(dx**2 + dy**2, axis=1)
# Fit H in log-log
idx_fit = (t_traj > 10) & (t_traj < 400)
H_fit = np.polyfit(np.log(t_traj[idx_fit]), np.log(msd[idx_fit]+1e-20), 1)[0] / 2
print(f"  H = {H_fit:.4f}")

# ── Mean-subtracted MSD ──────────────────────────────────────────────────────
mean_vx = np.mean(vx, axis=1, keepdims=True)
mean_vy = np.mean(vy, axis=1, keepdims=True)
vx_sub = vx - mean_vx
vy_sub = vy - mean_vy
dx_sub = np.cumsum(np.vstack([[np.zeros(n_tr)], vx_sub * dt_save]), axis=0)
dy_sub = np.cumsum(np.vstack([[np.zeros(n_tr)], vy_sub * dt_save]), axis=0)
msd_sub = np.mean(dx_sub**2 + dy_sub**2, axis=1)
H_sub = np.polyfit(np.log(t_traj[idx_fit]), np.log(msd_sub[idx_fit]+1e-20), 1)[0] / 2
print(f"  H (mean-subtracted) = {H_sub:.4f}")

# ── Lagrangian velocity autocorrelation (global) ─────────────────────────────
print("Computing R_v(τ)...")
max_lag = 200  # 200 × dt_save = 80 time units
R_v = np.zeros(max_lag)
for tau in range(max_lag):
    if tau == 0:
        R_v[0] = np.mean(vx**2 + vy**2)
    else:
        R_v[tau] = np.mean(vx[:-tau]*vx[tau:] + vy[:-tau]*vy[tau:])
R_v /= R_v[0]
T_L = np.trapezoid(R_v, dx=dt_save)
print(f"  T_L = {T_L:.2f} (T_total = {t_traj[-1]:.1f})")

# ── Vortex/Strain classification from vorticity snapshots ───────────────────
print("Classifying tracers (Okubo-Weiss)...")
vort_snaps = np.load(f'{DATA}/vorticity_snapshots.npy')  # (15, 256, 256)
snap_times  = np.load(f'{DATA}/snap_times.npy')
kv = np.fft.fftfreq(N, d=1.0/N)
KX, KY = np.meshgrid(kv, kv, indexing='ij')
K2 = KX**2 + KY**2; K2[0,0] = 1.0

def compute_Q(omega):
    omega_hat = fft2(omega)
    psi_hat = -omega_hat / K2; psi_hat[0,0] = 0
    # strain: s = sxx^2 + sxy^2 where sxx = d²ψ/dxdy, sxy = (d²ψ/dy² - d²ψ/dx²)/2
    sxx_hat = KX * KY * psi_hat
    syy_hat = -(KX**2 - KY**2) * psi_hat / 2
    sxx = np.real(ifft2(sxx_hat))
    syy = np.real(ifft2(syy_hat))
    # simplified: Q = |∇u|^2 - ω^2 = 4*(sxx^2+syy^2) - ω^2
    Q = 4*(sxx**2 + syy**2) - omega**2
    return Q

Q_fields = np.array([compute_Q(vort_snaps[i]) for i in range(len(snap_times))])
# Threshold: median Q (more robust than 0)
Q_thresh = np.median(Q_fields)
print(f"  Q threshold = {Q_thresh:.6f}")

# Assign each tracer a state at each trajectory time via nearest-snapshot interpolation
state = np.zeros((n_t, n_tr), dtype=np.int8)  # 0=strain, 1=vortex
snap_idx = np.searchsorted(snap_times, t_traj)
snap_idx = np.clip(snap_idx, 0, len(snap_times)-1)
dx_grid = L / N

for ti in range(n_t):
    si = snap_idx[ti]
    Q_snap = Q_fields[si]
    # Bilinear interpolation
    ix = (traj_x[ti] / dx_grid % N).astype(int) % N
    iy = (traj_y[ti] / dx_grid % N).astype(int) % N
    Q_tracer = Q_snap[ix, iy]
    state[ti] = (Q_tracer < Q_thresh).astype(np.int8)

frac_vortex = state.mean()
print(f"  Fraction in vortex: {frac_vortex:.3f}")

# ── Conditional R_v ──────────────────────────────────────────────────────────
print("Computing conditional R_v...")
state_vel = state[:-1]  # aligned with velocity array (n_t-1, n_tr)
in_vortex = state_vel == 1
in_strain  = state_vel == 0

R_v_vortex = np.zeros(max_lag)
R_v_strain  = np.zeros(max_lag)
for tau in range(max_lag):
    if tau == 0:
        R_v_vortex[0] = np.mean((vx**2 + vy**2)[in_vortex]) if in_vortex.any() else 0
        R_v_strain[0]  = np.mean((vx**2 + vy**2)[in_strain])  if in_strain.any()  else 0
    else:
        mask_v = in_vortex[:-tau]
        mask_s = in_strain[:-tau]
        v_dot_tau = vx[:-tau]*vx[tau:] + vy[:-tau]*vy[tau:]
        R_v_vortex[tau] = np.mean(v_dot_tau[mask_v]) if mask_v.any() else 0
        R_v_strain[tau]  = np.mean(v_dot_tau[mask_s])  if mask_s.any()  else 0

R_v0_v = R_v_vortex[0] if R_v_vortex[0] != 0 else 1
R_v0_s = R_v_strain[0]  if R_v_strain[0]  != 0 else 1
R_v_vortex /= R_v0_v
R_v_strain  /= R_v0_s
tau_lags = np.arange(max_lag) * dt_save

# ── Trapping time distribution ────────────────────────────────────────────────
print("Computing trapping times...")
# For each tracer, find runs of state==1 (vortex)
all_trap_durations = []
all_free_durations = []

for tr_i in range(n_tr):
    s = state[:, tr_i]
    in_trap = False
    trap_start = 0
    for ti in range(n_t):
        if s[ti] == 1 and not in_trap:
            in_trap = True; trap_start = ti
        elif s[ti] == 0 and in_trap:
            in_trap = False
            dur = (ti - trap_start) * dt_save
            if dur > 0:
                all_trap_durations.append(dur)
    # free flights
    in_free = False; free_start = 0
    for ti in range(n_t):
        if s[ti] == 0 and not in_free:
            in_free = True; free_start = ti
        elif s[ti] == 1 and in_free:
            in_free = False
            dur = (ti - free_start) * dt_save
            if dur > 0:
                all_free_durations.append(dur)

trap_dur = np.array(all_trap_durations)
free_dur = np.array(all_free_durations)
print(f"  Trapping events: {len(trap_dur)}, mean={trap_dur.mean():.2f}, median={np.median(trap_dur):.2f}")
print(f"  Free events:     {len(free_dur)},   mean={free_dur.mean():.2f}")

# ── Q-threshold sensitivity ───────────────────────────────────────────────────
print("Q-threshold sensitivity...")
q_thresholds = [np.percentile(Q_fields, p) for p in [25, 40, 50, 60, 75]]
mu_vals = []
for qt in q_thresholds:
    durs = []
    for tr_i in range(0, n_tr, 10):  # subsample
        s = (Q_fields[snap_idx[:], tr_i//10 % N, tr_i % N] < qt) if False else None
        # Use precomputed state with new threshold
        ix = (traj_x[:, tr_i] / dx_grid % N).astype(int) % N
        iy = (traj_y[:, tr_i] / dx_grid % N).astype(int) % N
        s_tr = np.array([Q_fields[snap_idx[ti]][ix[ti], iy[ti]] < qt for ti in range(n_t)])
        in_trap = False; start = 0
        for ti in range(n_t):
            if s_tr[ti] and not in_trap:
                in_trap = True; start = ti
            elif not s_tr[ti] and in_trap:
                in_trap = False
                d = (ti - start) * dt_save
                if d > 0: durs.append(d)
    if len(durs) > 20:
        durs = np.array(durs)
        # Simple Hill estimator
        durs_sorted = np.sort(durs)
        k = max(10, int(0.1*len(durs_sorted)))
        mu_hill = 1 + k / np.sum(np.log(durs_sorted[-k:]/durs_sorted[-k]))
        mu_vals.append(mu_hill)
    else:
        mu_vals.append(np.nan)
print("  μ(Q_thresh percentile):", list(zip([25,40,50,60,75], [f'{m:.2f}' for m in mu_vals])))

# ── Surrogate trajectory test ─────────────────────────────────────────────────
print("Surrogate test (phase randomization)...")
# For each tracer, FFT the velocity time series, randomize phases, IFFT → surrogate velocities
# Then compute MSD of surrogate trajectories
n_surr = 500  # use subset for speed
rng = np.random.default_rng(42)
tr_idx = rng.choice(n_tr, n_surr, replace=False)

def compute_msd_from_vel(vx_s, vy_s, dt):
    disp_x = np.cumsum(np.vstack([[np.zeros(vx_s.shape[1])], vx_s * dt]), axis=0)
    disp_y = np.cumsum(np.vstack([[np.zeros(vy_s.shape[1])], vy_s * dt]), axis=0)
    return np.mean(disp_x**2 + disp_y**2, axis=1)

# Original MSD for these tracers
msd_orig = compute_msd_from_vel(vx[:, tr_idx], vy[:, tr_idx], dt_save)

# Surrogate: phase-randomize Fourier coefficients
vx_surr = np.zeros_like(vx[:, tr_idx])
vy_surr = np.zeros_like(vy[:, tr_idx])
for j, tri in enumerate(tr_idx):
    vxf = np.fft.rfft(vx[:, tri])
    vyf = np.fft.rfft(vy[:, tri])
    phases = rng.uniform(0, 2*np.pi, len(vxf))
    vx_surr[:, j] = np.fft.irfft(np.abs(vxf) * np.exp(1j * phases), n=n_t-1)
    vy_surr[:, j] = np.fft.irfft(np.abs(vyf) * np.exp(1j * phases), n=n_t-1)

msd_surr = compute_msd_from_vel(vx_surr, vy_surr, dt_save)

# Fit H for both
t_fit = t_traj[:n_t]
fit_mask = (t_fit > 10) & (t_fit < 400)
H_orig_surr = np.polyfit(np.log(t_fit[fit_mask]), np.log(msd_orig[fit_mask]+1e-20), 1)[0] / 2
H_surr_     = np.polyfit(np.log(t_fit[fit_mask]), np.log(msd_surr[fit_mask]+1e-20), 1)[0] / 2
print(f"  H (original, {n_surr} tracers) = {H_orig_surr:.4f}")
print(f"  H (surrogate, phase-randomised) = {H_surr_:.4f}")
print(f"  → Memory contribution: ΔH = {H_orig_surr - H_surr_:.4f}")

# ── Displacement PDF at multiple lags ────────────────────────────────────────
print("Computing displacement PDFs...")
lag_idxs = [5, 10, 25, 50, 100, 250]   # in steps
disp_pdfs = {}
alpha_vals = []
lag_times = []
for li in lag_idxs:
    if li >= n_t: continue
    dx_disp = ux_traj[li:] - ux_traj[:-li]
    dx_flat = dx_disp.flatten()
    disp_pdfs[li] = dx_flat
    lag_times.append(li * dt_save)
    # Characteristic function α estimate
    k_arr = np.linspace(0.1, 2.0, 30)
    phi = np.array([np.mean(np.exp(1j * k * dx_flat)) for k in k_arr])
    log_phi = -np.log(np.abs(phi) + 1e-20)
    pos = log_phi > 0
    if pos.sum() > 3:
        alpha_cf, _ = np.polyfit(np.log(k_arr[pos]), np.log(log_phi[pos]), 1)
        alpha_vals.append(float(np.clip(alpha_cf, 0.3, 2.5)))
    else:
        alpha_vals.append(2.0)

print(f"  α(τ) = {list(zip([f'{lt:.1f}' for lt in lag_times], [f'{a:.2f}' for a in alpha_vals]))}")

# ── Save numerical results ────────────────────────────────────────────────────
np.savez(f'{OUT}/iter1_results.npz',
    t_traj=t_traj, msd=msd, H_fit=H_fit, H_sub=H_sub,
    tau_lags=tau_lags, R_v=R_v, R_v_vortex=R_v_vortex, R_v_strain=R_v_strain,
    T_L=T_L, frac_vortex=frac_vortex,
    trap_durations=trap_dur, free_durations=free_dur,
    t_fit=t_fit, msd_orig_surr=msd_orig, msd_surr=msd_surr,
    H_orig_surr=H_orig_surr, H_surr=H_surr_,
    lag_times=lag_times, alpha_vals=alpha_vals,
    q_thresholds=q_thresholds, mu_q_sens=mu_vals,
)
print("Saved iter1_results.npz")

# ── Plots ─────────────────────────────────────────────────────────────────────
ts = str(int(time.time()))

# Plot 1: MSD comparison (original vs mean-subtracted vs surrogate)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.loglog(t_traj[1:], msd[1:], 'k-', lw=2, label=f'Original (H={H_fit:.3f})')
ax.loglog(t_traj[1:], msd_sub[1:], 'b--', lw=2, label=f'Mean-subtracted (H={H_sub:.3f})')
t_ref = np.array([10, 400])
ax.loglog(t_ref, msd[np.argmin(abs(t_traj-10))] * (t_ref/10)**(2*H_fit), 'k:', alpha=0.4)
ax.set_xlabel('Time τ', fontsize=12)
ax.set_ylabel('MSD', fontsize=12)
ax.set_title('MSD vs Time', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Conditional R_v
ax2 = axes[1]
ax2.plot(tau_lags[:100], R_v[:100], 'k-', lw=2, label='Global R_v')
ax2.plot(tau_lags[:100], R_v_vortex[:100], 'b-', lw=2, label='Vortex (Q<Q₀)')
ax2.plot(tau_lags[:100], R_v_strain[:100], 'r-', lw=2, label='Strain (Q>Q₀)')
ax2.axhline(0, color='gray', ls=':', lw=0.8)
ax2.set_xlabel('Lag τ', fontsize=12)
ax2.set_ylabel('Normalised R_v(τ)', fontsize=12)
ax2.set_title('Conditional Velocity Autocorrelation', fontsize=11)
ax2.legend(fontsize=9)
ax2.set_xlim(0, tau_lags[99])
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/iter1_msd_Rv_{ts}.png', dpi=150)
plt.close()
print(f"Saved iter1_msd_Rv_{ts}.png")

# Plot 2: Surrogate test MSD
fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(t_fit[1:], msd_orig[1:], 'k-', lw=2, label=f'Original (H={H_orig_surr:.3f})')
ax.loglog(t_fit[1:], msd_surr[1:], 'r--', lw=2, label=f'Phase-randomised surrogate (H={H_surr_:.3f})')
ax.set_xlabel('Time τ', fontsize=12)
ax.set_ylabel('MSD', fontsize=12)
ax.set_title('Surrogate Test: Memory vs Increment Mechanism', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/iter1_surrogate_{ts}.png', dpi=150)
plt.close()
print(f"Saved iter1_surrogate_{ts}.png")

# Plot 3: Trapping times
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
if len(trap_dur) > 20:
    bins = np.logspace(np.log10(trap_dur.min()+0.01), np.log10(trap_dur.max()), 40)
    cnt, edges = np.histogram(trap_dur, bins=bins, density=True)
    mids = np.sqrt(edges[:-1]*edges[1:])
    mask = cnt > 0
    ax.loglog(mids[mask], cnt[mask], 'bo', ms=5, label='Data')
    # Hill estimator
    td_sorted = np.sort(trap_dur)
    k = max(10, int(0.1*len(td_sorted)))
    mu_hill = 1 + k / np.sum(np.log(td_sorted[-k:]/td_sorted[-k]))
    ax.loglog(mids[mask], cnt[mask].max()*(mids[mask]/mids[mask].min())**(-mu_hill),
              'r--', lw=2, label=f'μ≈{mu_hill:.2f} (Hill)')
    ax.set_xlabel('Trapping duration τ_trap', fontsize=12)
    ax.set_ylabel('PDF', fontsize=12)
    ax.set_title(f'Trapping Time Distribution (n={len(trap_dur)})', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    print(f"  μ_Hill = {mu_hill:.3f}")

# Q-threshold sensitivity
ax2 = axes[1]
percs = [25, 40, 50, 60, 75]
mu_clean = [m if not np.isnan(m) else 0 for m in mu_vals]
ax2.bar(percs, mu_clean, color='steelblue', alpha=0.7)
ax2.axhline(2.0, color='r', ls='--', lw=1.5, label='μ=2 (finite mean)')
ax2.axhline(1.0, color='orange', ls='--', lw=1.5, label='μ=1 (diverging mean)')
ax2.set_xlabel('Q threshold percentile', fontsize=12)
ax2.set_ylabel('Trapping time exponent μ', fontsize=12)
ax2.set_title('Q-Threshold Sensitivity', fontsize=11)
ax2.legend(fontsize=9)
ax2.set_ylim(0, max(3, max(mu_clean)+0.5) if any(mu_clean) else 3)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/iter1_trapping_sensitivity_{ts}.png', dpi=150)
plt.close()
print(f"Saved iter1_trapping_sensitivity_{ts}.png")

# Plot 4: Displacement PDFs + α(τ)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0, 1, len(lag_idxs)))
for j, li in enumerate(lag_idxs):
    if li not in disp_pdfs: continue
    dx_flat = disp_pdfs[li]
    bins = np.linspace(np.percentile(dx_flat, 0.5), np.percentile(dx_flat, 99.5), 80)
    cnt, edges = np.histogram(dx_flat, bins=bins, density=True)
    mids = 0.5*(edges[:-1]+edges[1:])
    ax.semilogy(mids, cnt, '-', color=colors[j], alpha=0.8, label=f'τ={li*dt_save:.1f}')
    # Gaussian reference
    σ = dx_flat.std()
    x_g = np.linspace(mids[0], mids[-1], 200)
    ax.semilogy(x_g, np.exp(-x_g**2/(2*σ**2))/(σ*np.sqrt(2*np.pi)), '--',
                color=colors[j], alpha=0.3)
ax.set_xlabel('Displacement Δx', fontsize=12)
ax.set_ylabel('PDF', fontsize=12)
ax.set_title('Displacement PDFs (dashed=Gaussian fit)', fontsize=11)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(lag_times, alpha_vals, 'ko-', lw=2, ms=6)
ax2.axhline(2.0, color='r', ls='--', lw=1.5, label='α=2 (Gaussian)')
ax2.axhline(1.2, color='b', ls='--', lw=1.5, label='α=1.2 (Kolmogorov)')
ax2.set_xlabel('Lag time τ', fontsize=12)
ax2.set_ylabel('Stable index α(τ)', fontsize=12)
ax2.set_title('Evolution of Stable Index', fontsize=11)
ax2.legend(fontsize=10)
ax2.set_ylim(0.5, 2.5)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/iter1_pdfs_alpha_{ts}.png', dpi=150)
plt.close()
print(f"Saved iter1_pdfs_alpha_{ts}.png")

print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("\n=== SUMMARY OF KEY RESULTS ===")
print(f"H (global)             = {H_fit:.4f}")
print(f"H (mean-subtracted)    = {H_sub:.4f}")
print(f"H (original, subsample)= {H_orig_surr:.4f}")
print(f"H (surrogate)          = {H_surr_:.4f}")
print(f"ΔH (memory effect)     = {H_orig_surr-H_surr_:.4f}")
print(f"T_L / T_total          = {T_L/t_traj[-1]:.4f}")
print(f"Fraction in vortex     = {frac_vortex:.3f}")
print(f"α(τ) range             = {min(alpha_vals):.2f} – {max(alpha_vals):.2f}")
print(f"Trapping events        = {len(trap_dur)}")
print(f"μ_Hill (trapping)      = {mu_hill:.3f}")
print(f"μ sensitivity (Q 25–75%): {[(p,f'{m:.2f}') for p,m in zip(percs,mu_clean)]}")
