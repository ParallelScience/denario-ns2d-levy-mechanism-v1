"""
2D Navier-Stokes simulation: vorticity-streamfunction, pseudo-spectral.
Optimised for speed: uses pyfftw (or numpy.fft fallback), GPU via CuPy if available.

Reduced grid N=256, AB2 time integration (cheaper than RK4), more tracers.
"""

import numpy as np
import os, json, time

# ── Try CuPy for GPU acceleration ─────────────────────────────────────────────
try:
    import cupy as cp
    from cupy.fft import fft2, ifft2, fftfreq
    xp = cp
    USE_GPU = True
    print("Using CuPy (GPU)")
except ImportError:
    from numpy.fft import fft2, ifft2, fftfreq
    xp = np
    USE_GPU = False
    print("Using NumPy (CPU)")

# ── Parameters ─────────────────────────────────────────────────────────────────
N          = 256           # grid NxN  (256 is fast enough, captures inverse cascade)
L          = 2 * np.pi
nu         = 2e-3          # viscosity (slightly higher → more stable at lower res)
dt         = 0.02          # time step
n_spinup   = 1000          # spin-up steps
n_steps    = 30000         # production steps  (T = 600)
n_tracers  = 8000          # Lagrangian tracers
save_every = 20            # save positions every N steps → 1500 snapshots
force_band = (3, 6)        # forcing wavenumbers
force_amp  = 1.0
seed       = 42

rng_np = np.random.default_rng(seed)

dx = L / N
kv = np.fft.fftfreq(N, d=1.0/N)
KX, KY = np.meshgrid(kv, kv, indexing='ij')
K2 = KX**2 + KY**2
K2[0, 0] = 1.0

# Dealiasing (2/3 rule)
kmax = N // 3
dealias_np = ((np.abs(KX) < kmax) & (np.abs(KY) < kmax)).astype(float)

# Forcing mask
K_mag = np.sqrt(K2)
force_mask_np = (K_mag >= force_band[0]) & (K_mag <= force_band[1])

if USE_GPU:
    KX_g      = cp.asarray(KX)
    KY_g      = cp.asarray(KY)
    K2_g      = cp.asarray(K2)
    dealias_g = cp.asarray(dealias_np)
    force_mask_g = cp.asarray(force_mask_np)
else:
    KX_g = KX; KY_g = KY; K2_g = K2
    dealias_g = dealias_np; force_mask_g = force_mask_np

print(f"N={N}, dt={dt}, T_spinup={n_spinup*dt:.0f}, T_prod={n_steps*dt:.0f}")
print(f"Tracers={n_tracers}, saves={n_steps//save_every}")

# ── Helpers ────────────────────────────────────────────────────────────────────

def make_forcing():
    phases = rng_np.uniform(0, 2*np.pi, (N, N))
    f_hat_np = np.zeros((N, N), dtype=complex)
    f_hat_np[force_mask_np] = force_amp * np.exp(1j * phases[force_mask_np])
    f_hat_np *= dealias_np
    if USE_GPU:
        return cp.asarray(f_hat_np)
    return f_hat_np

def rhs_hat(w_hat):
    psi_hat = -w_hat / K2_g
    psi_hat[0, 0] = 0.0
    ux_hat =  1j * KY_g * psi_hat
    uy_hat = -1j * KX_g * psi_hat
    ux = xp.real(ifft2(ux_hat * dealias_g))
    uy = xp.real(ifft2(uy_hat * dealias_g))
    # Spectral derivatives of vorticity
    dw_dx = xp.real(ifft2(1j * KX_g * w_hat * dealias_g))
    dw_dy = xp.real(ifft2(1j * KY_g * w_hat * dealias_g))
    adv   = fft2(ux * dw_dx + uy * dw_dy) * dealias_g
    diff  = -nu * K2_g * w_hat
    return -adv + diff

def get_uv(w_hat):
    psi_hat = -w_hat / K2_g
    psi_hat[0, 0] = 0.0
    ux = xp.real(ifft2(1j * KY_g * psi_hat * dealias_g))
    uy = xp.real(ifft2(-1j * KX_g * psi_hat * dealias_g))
    return ux, uy

def interp_uv_cpu(ux_cpu, uy_cpu, px, py):
    ix  = (px / dx) % N
    iy  = (py / dx) % N
    ix0 = np.floor(ix).astype(int) % N
    iy0 = np.floor(iy).astype(int) % N
    ix1 = (ix0 + 1) % N
    iy1 = (iy0 + 1) % N
    fx  = ix - np.floor(ix)
    fy  = iy - np.floor(iy)
    w00 = (1-fx)*(1-fy); w10 = fx*(1-fy)
    w01 = (1-fx)*fy;     w11 = fx*fy
    vx  = w00*ux_cpu[ix0,iy0] + w10*ux_cpu[ix1,iy0] + w01*ux_cpu[ix0,iy1] + w11*ux_cpu[ix1,iy1]
    vy  = w00*uy_cpu[ix0,iy0] + w10*uy_cpu[ix1,iy0] + w01*uy_cpu[ix0,iy1] + w11*uy_cpu[ix1,iy1]
    return vx, vy

# ── Initialise ─────────────────────────────────────────────────────────────────
w0 = rng_np.standard_normal((N, N)) * 0.1
if USE_GPU:
    w_hat = cp.asarray(fft2(w0)) * dealias_g
else:
    w_hat = fft2(w0) * dealias_g
w_hat_prev = w_hat.copy()

# ── Spin-up (Adams-Bashforth 1st order = forward Euler) ───────────────────────
print("Spinning up...")
t0 = time.time()
rhs_prev = None
for i in range(n_spinup):
    f_hat = make_forcing()
    r     = rhs_hat(w_hat) + f_hat
    if rhs_prev is None:
        w_hat = w_hat + dt * r          # Euler start
    else:
        w_hat = w_hat + dt * (1.5*r - 0.5*rhs_prev)   # AB2
    rhs_prev = r
    if i % 200 == 0:
        if USE_GPU:
            rms = float(cp.sqrt(cp.mean(cp.abs(ifft2(w_hat))**2)).get())
        else:
            rms = float(np.sqrt(np.mean(np.abs(ifft2(w_hat))**2)))
        print(f"  spin-up {i:4d}: rms(ω)={rms:.4f}  ({time.time()-t0:.0f}s)")

print(f"Spin-up done in {time.time()-t0:.1f}s")

# ── Seed tracers ───────────────────────────────────────────────────────────────
px = rng_np.uniform(0, L, n_tracers)
py = rng_np.uniform(0, L, n_tracers)

n_saves  = n_steps // save_every
traj_x   = np.zeros((n_saves, n_tracers), dtype=np.float32)
traj_y   = np.zeros((n_saves, n_tracers), dtype=np.float32)

vort_snaps = []
snap_times  = []
save_idx   = 0

# ── Production run ─────────────────────────────────────────────────────────────
print(f"Production: {n_steps} steps...")
t0 = time.time()
for i in range(n_steps):
    f_hat = make_forcing()
    r     = rhs_hat(w_hat) + f_hat
    if rhs_prev is None:
        w_hat = w_hat + dt * r
    else:
        w_hat = w_hat + dt * (1.5*r - 0.5*rhs_prev)
    rhs_prev = r

    # Tracer advection (forward Euler)
    ux_g, uy_g = get_uv(w_hat)
    if USE_GPU:
        ux_cpu = ux_g.get(); uy_cpu = uy_g.get()
    else:
        ux_cpu = ux_g; uy_cpu = uy_g
    vx, vy = interp_uv_cpu(ux_cpu, uy_cpu, px, py)
    px = (px + dt * vx) % L
    py = (py + dt * vy) % L

    if i % save_every == 0:
        traj_x[save_idx] = px.astype(np.float32)
        traj_y[save_idx] = py.astype(np.float32)
        save_idx += 1

    if i % 2000 == 0:
        if USE_GPU:
            w_now = cp.asnumpy(xp.real(ifft2(w_hat)))
        else:
            w_now = np.real(ifft2(w_hat))
        vort_snaps.append(w_now.astype(np.float32))
        snap_times.append(i * dt)
        rms  = float(np.sqrt(np.mean(w_now**2)))
        elapsed = time.time()-t0
        eta = elapsed/(i+1)*(n_steps-i-1)
        print(f"  step {i:5d} t={i*dt:6.1f}: rms(ω)={rms:.4f}  elapsed={elapsed:.0f}s ETA={eta:.0f}s")

print(f"Production done in {time.time()-t0:.1f}s")

# ── Save ───────────────────────────────────────────────────────────────────────
out = "/home/node/work/projects/ns2d_levy_v1/data"
np.save(f"{out}/traj_x.npy", traj_x)
np.save(f"{out}/traj_y.npy", traj_y)
np.save(f"{out}/vorticity_snapshots.npy", np.array(vort_snaps))
np.save(f"{out}/snap_times.npy", np.array(snap_times))
t_traj = np.arange(n_saves) * save_every * dt
np.save(f"{out}/t_traj.npy", t_traj.astype(np.float32))

meta = dict(N=N, L=L, nu=nu, dt=dt, n_steps=n_steps, n_tracers=n_tracers,
            save_every=save_every, force_band=list(force_band),
            force_amp=force_amp, seed=seed,
            t_max=n_steps*dt, n_saves=n_saves, USE_GPU=USE_GPU)
with open(f"{out}/meta.json","w") as f:
    json.dump(meta, f, indent=2)

print("Files:")
for fn in ["traj_x.npy","traj_y.npy","vorticity_snapshots.npy","snap_times.npy","t_traj.npy","meta.json"]:
    sz = os.path.getsize(f"{out}/{fn}")/1e6
    print(f"  {fn}: {sz:.1f} MB")
