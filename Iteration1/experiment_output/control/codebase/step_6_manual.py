"""
Manual step 6: Plot Conditional Lagrangian Analysis & Trapping/Jump Statistics
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, os, time

data_dir = '/home/node/work/projects/ns2d_levy_v1/Iteration1/experiment_output/control/data'
timestamp = str(int(time.time()))

# Load data
cond = np.load(f'{data_dir}/conditional_stats.npz', allow_pickle=True)
with open(f'{data_dir}/trapping_jump_dists.pkl','rb') as f:
    tjd = pickle.load(f)
lags = np.load(f'{data_dir}/lagrangian_stats.npz', allow_pickle=True)

print("Loaded files. Keys:")
print("  conditional_stats:", list(cond.keys()))
print("  lagrangian_stats:", list(lags.keys()))

# ── Figure 1: Conditional R_v(τ) for Vortex vs Strain ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

tau_lags = cond['tau_lags']
R_v_vortex = cond['R_v_vortex']
R_v_strain = cond['R_v_strain']

ax = axes[0]
ax.plot(tau_lags, R_v_vortex / R_v_vortex[0], label='Vortex (Q<0)', color='royalblue', lw=2)
ax.plot(tau_lags, R_v_strain / R_v_strain[0], label='Strain (Q>0)', color='firebrick', lw=2)
ax.axhline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel('Lag τ', fontsize=12)
ax.set_ylabel('Normalised R_v(τ)', fontsize=12)
ax.set_title('Conditional Lagrangian Velocity Autocorrelation', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(0, min(200, tau_lags[-1]))
ax.grid(True, alpha=0.3)

# ── Figure 1b: Cross-correlation of large jumps with vortex exit ─────────────
ax2 = axes[1]
corr_lags = cond['correlation_lags']
cross_corr = cond['cross_corr']
ax2.plot(corr_lags, cross_corr, color='purple', lw=2)
ax2.axhline(float(np.mean(cross_corr)), color='k', ls='--', lw=1, label='Baseline')
ax2.axvline(0, color='gray', ls=':', lw=1)
ax2.set_xlabel('Lag relative to vortex exit', fontsize=12)
ax2.set_ylabel('P(large jump)', fontsize=12)
ax2.set_title('Cross-correlation: Vortex Exit → Large Jump', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
path1 = f'{data_dir}/step_6_conditional_Rv_exit_xcorr_{timestamp}.png'
plt.savefig(path1, dpi=150)
plt.close()
print(f"Saved: {path1}")

# ── Figure 2: Trapping time distribution ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

trapping = np.array(tjd['trapping_durations'])
jumps    = np.array(tjd['jump_sizes'])

ax = axes[0]
if len(trapping) > 10:
    bins = np.logspace(np.log10(trapping.min()+1e-10), np.log10(trapping.max()), 40)
    counts, edges = np.histogram(trapping, bins=bins, density=True)
    mids = np.sqrt(edges[:-1]*edges[1:])
    mask = counts > 0
    ax.loglog(mids[mask], counts[mask], 'o', color='royalblue', ms=4, label='Data')
    # Annotate with μ
    mu = float(cond['mu_trap'])
    se = float(cond['se_trap'])
    xfit = mids[mask]
    ax.loglog(xfit, xfit**0*counts[mask].max() * (xfit/xfit.min())**(-mu),
              'r--', lw=1.5, label=f'μ={mu:.2f}±{se:.2f}')
    ax.set_xlabel('Trapping duration τ_trap', fontsize=12)
    ax.set_ylabel('PDF', fontsize=12)
    ax.set_title('Trapping Time Distribution', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Insufficient trapping events', ha='center', va='center', transform=ax.transAxes)

ax2 = axes[1]
if len(jumps) > 10:
    bins = np.logspace(np.log10(max(jumps.min(), 1e-10)), np.log10(jumps.max()), 40)
    counts, edges = np.histogram(jumps, bins=bins, density=True)
    mids = np.sqrt(edges[:-1]*edges[1:])
    mask = counts > 0
    ax2.loglog(mids[mask], counts[mask], 'o', color='firebrick', ms=4, label='Data')
    mu_j = float(cond['mu_jump'])
    se_j = float(cond['se_jump'])
    ax2.loglog(mids[mask], counts[mask].max() * (mids[mask]/mids[mask].min())**(-mu_j),
               'r--', lw=1.5, label=f'β={mu_j:.2f}±{se_j:.2f}')
    ax2.set_xlabel('Jump size |Δx|', fontsize=12)
    ax2.set_ylabel('PDF', fontsize=12)
    ax2.set_title('Jump Size Distribution', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'Insufficient jump data', ha='center', va='center', transform=ax2.transAxes)

plt.tight_layout()
path2 = f'{data_dir}/step_6_trapping_jump_dists_{timestamp}.png'
plt.savefig(path2, dpi=150)
plt.close()
print(f"Saved: {path2}")

print("\nStep 6 complete. Key metrics:")
print(f"  Trapping time exponent μ = {float(cond['mu_trap']):.3f} ± {float(cond['se_trap']):.3f}")
print(f"  Jump size exponent β     = {float(cond['mu_jump']):.3f} ± {float(cond['se_jump']):.3f}")
