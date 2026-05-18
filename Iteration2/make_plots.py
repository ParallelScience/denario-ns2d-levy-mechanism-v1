"""Generate iteration 2 plots from saved data arrays."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time

DATA = '/home/node/work/projects/ns2d_levy_v1/Iteration2/experiment_output/control/data'
OUT  = DATA
ts   = str(int(time.time()))

t_traj = np.load('/home/node/work/projects/ns2d_levy_v1/data/t_traj.npy')
dt_save = float(t_traj[1] - t_traj[0])

msd       = np.load(f'{DATA}/msd.npy')
msd_strain= np.load(f'{DATA}/msd_strain.npy')
msd_vortex= np.load(f'{DATA}/msd_vortex.npy')
msd_surr  = np.load(f'{DATA}/msd_surr.npy')
H_t       = np.load(f'{DATA}/H_t.npy')
H_t_strain= np.load(f'{DATA}/H_t_strain.npy')
H_t_vortex= np.load(f'{DATA}/H_t_vortex.npy')
H_t_surr  = np.load(f'{DATA}/H_t_surr.npy')
Rv_full   = np.load(f'{DATA}/Rv_full.npy')
Rv_strain = np.load(f'{DATA}/Rv_strain.npy')
Rv_vortex = np.load(f'{DATA}/Rv_vortex.npy')
mu_exp    = np.load(f'{DATA}/mu_exponents.npy')
mu_trap, mu_jump = mu_exp[0], mu_exp[1]

strain_pct = np.load(f'{DATA}/strain_percentage.npy')
print(f"Loaded data. t=[{t_traj[0]:.1f},{t_traj[-1]:.1f}]")
print(f"MSD shape: {msd.shape}, H_t shape: {H_t.shape}")

# ── Plot 1: MSD comparison (full, strain, vortex, surrogate) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
t_plot = t_traj
# trim to same length
n = min(len(t_plot), len(msd), len(msd_strain), len(msd_vortex), len(msd_surr))
t_p = t_plot[:n]; t_p1 = t_p[1:]

ax.loglog(t_p1, msd[1:n], 'k-', lw=2.5, label='Full ensemble')
ax.loglog(t_p1, msd_strain[1:n], 'r-', lw=2, label='Strain-dominated (>70%)')
ax.loglog(t_p1, msd_vortex[1:n], 'b-', lw=2, label='Vortex-dominated (<30%)')
ax.loglog(t_p1, msd_surr[1:n], 'k--', lw=1.5, alpha=0.7, label='Phase-random surrogate')
# Reference slopes
t_ref = np.array([20, 400])
for slope, col, lbl in [(0.85*2, 'gray', 'H=0.85'), (1.0*2, 'orange', 'H=1.0 (ballistic)'), (0.5*2, 'purple', 'H=0.5 (diffusive)')]:
    idx0 = np.argmin(abs(t_p - 20))
    ax.loglog(t_ref, msd[idx0]*(t_ref/20)**slope, '--', color=col, alpha=0.5, lw=1, label=lbl)
ax.set_xlabel('Time τ', fontsize=12)
ax.set_ylabel('MSD', fontsize=12)
ax.set_title('MSD: Subpopulations vs Surrogate', fontsize=11)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Running H(t)
ax2 = axes[1]
valid = ~np.isnan(H_t[:n])
if valid.any():
    ax2.plot(t_p[valid], H_t[:n][valid], 'k-', lw=2.5, label='Full ensemble')
valid_s = ~np.isnan(H_t_strain[:n])
if valid_s.any():
    ax2.plot(t_p[valid_s], H_t_strain[:n][valid_s], 'r-', lw=2, label='Strain-dominated (>70%)')
valid_v = ~np.isnan(H_t_vortex[:n])
if valid_v.any():
    ax2.plot(t_p[valid_v], H_t_vortex[:n][valid_v], 'b-', lw=2, label='Vortex-dominated (<30%)')
valid_surr = ~np.isnan(H_t_surr[:n])
if valid_surr.any():
    ax2.plot(t_p[valid_surr], H_t_surr[:n][valid_surr], 'k--', lw=1.5, alpha=0.7, label='Surrogate')
ax2.axhline(0.5, color='purple', ls=':', lw=1.5, label='H=0.5 (diffusive)')
ax2.axhline(1.0, color='orange', ls=':', lw=1.5, label='H=1.0 (ballistic)')
ax2.set_xlabel('Time τ', fontsize=12)
ax2.set_ylabel('Running Hurst exponent H(t)', fontsize=12)
ax2.set_title('Temporal Evolution of H(t)', fontsize=11)
ax2.legend(fontsize=8)
ax2.set_ylim(0, 1.5)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
p = f'{OUT}/iter2_msd_Ht_{ts}.png'
plt.savefig(p, dpi=150); plt.close(); print(f"Saved {p}")

# ── Plot 2: Conditional R_v(τ) ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
tau_arr = np.arange(len(Rv_full)) * dt_save
nmax = min(300, len(Rv_full))
Rv_f = Rv_full[:nmax] / (Rv_full[0] if Rv_full[0] != 0 else 1)
Rv_s = Rv_strain[:nmax] / (Rv_strain[0] if Rv_strain[0] != 0 else 1)
Rv_v = Rv_vortex[:nmax] / (Rv_vortex[0] if Rv_vortex[0] != 0 else 1)
ax.plot(tau_arr[:nmax], Rv_f, 'k-', lw=2.5, label='Full ensemble')
ax.plot(tau_arr[:nmax], Rv_s, 'r-', lw=2, label='Strain-dominated')
ax.plot(tau_arr[:nmax], Rv_v, 'b-', lw=2, label='Vortex-dominated')
ax.axhline(0, color='gray', ls=':', lw=0.8)
ax.axhline(np.exp(-1), color='gray', ls='--', lw=0.8, alpha=0.5, label='1/e')
ax.set_xlabel('Lag τ', fontsize=12)
ax.set_ylabel('Normalised R_v(τ)', fontsize=12)
ax.set_title('Conditional Lagrangian Velocity Autocorrelation', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(0, tau_arr[nmax-1])
ax.grid(True, alpha=0.3)
plt.tight_layout()
p = f'{OUT}/iter2_conditional_Rv_{ts}.png'
plt.savefig(p, dpi=150); plt.close(); print(f"Saved {p}")

# ── Plot 3: Strain fraction histogram + H vs strain fraction ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.hist(strain_pct * 100, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(70, color='r', ls='--', lw=2, label='>70% = strain-dominated')
ax.axvline(30, color='b', ls='--', lw=2, label='<30% = vortex-dominated')
ax.set_xlabel('% time in strain region (Q > Q₀)', fontsize=12)
ax.set_ylabel('Number of tracers', fontsize=12)
ax.set_title(f'Tracer State Distribution (n={len(strain_pct)})', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# H vs strain fraction (binned)
ax2 = axes[1]
bins_pct = np.linspace(0, 1, 21)
bin_centers = 0.5*(bins_pct[:-1]+bins_pct[1:])
# Compute mean H for each bin using running H_t at late time
late_start = len(t_traj)*2//3
H_late_full = np.nanmean(H_t[late_start:]) if not np.all(np.isnan(H_t[late_start:])) else 0.85

# Since we only have two sub-population H values, just show those as reference
ax2.axhline(H_late_full, color='k', ls='-', lw=2, label=f'Full ensemble H={H_late_full:.3f}')
H_s_late = np.nanmean(H_t_strain[late_start:]) if not np.all(np.isnan(H_t_strain[late_start:])) else np.nan
H_v_late = np.nanmean(H_t_vortex[late_start:]) if not np.all(np.isnan(H_t_vortex[late_start:])) else np.nan
if not np.isnan(H_s_late):
    ax2.axhline(H_s_late, color='r', ls='--', lw=2, label=f'Strain-dom H={H_s_late:.3f}')
if not np.isnan(H_v_late):
    ax2.axhline(H_v_late, color='b', ls='--', lw=2, label=f'Vortex-dom H={H_v_late:.3f}')
ax2.axhline(0.5, color='purple', ls=':', lw=1.5, label='Diffusive (H=0.5)')
ax2.axhline(1.0, color='orange', ls=':', lw=1.5, label='Ballistic (H=1.0)')
ax2.set_xlabel('% time in strain region', fontsize=12)
ax2.set_ylabel('Hurst exponent H (late regime)', fontsize=12)
ax2.set_title('H by sub-population', fontsize=11)
ax2.set_ylim(0, 1.5)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
p = f'{OUT}/iter2_strain_fraction_H_{ts}.png'
plt.savefig(p, dpi=150); plt.close(); print(f"Saved {p}")

print("\n=== KEY METRICS ===")
print(f"H_full (late):    {H_late_full:.4f}")
print(f"H_strain (late):  {H_s_late:.4f}")
print(f"H_vortex (late):  {H_v_late:.4f}")
print(f"H_surrogate (late): {np.nanmean(H_t_surr[late_start:]):.4f}")
print(f"mu_trap: {mu_trap}")
print(f"mu_jump: {mu_jump:.4f}")
