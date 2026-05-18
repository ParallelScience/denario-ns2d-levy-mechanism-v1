## Task

Discover the *mechanistic* physical explanation for why Lagrangian tracers in 2D turbulence exhibit anomalous diffusion (superdiffusion / Lévy-flight-like heavy tails in displacement PDFs). The goal is not to verify a scaling relation — it is to identify **what physical structures or processes in the flow are responsible** for the heavy tails.

## Scientific Background

In 2D turbulence, Lagrangian tracers are known to exhibit superdiffusive transport (MSD ~ t^{2H}, H > 0.5). The displacement PDF develops heavy tails that deviate from Gaussian. The fractional diffusion equation ∂_t P = -D_α (-Δ)^{α/2} P with α < 2 is one effective description. The question is: **what physical mechanism in the Navier-Stokes flow produces this operator?**

Candidate mechanisms from the literature:
1. **Coherent vortex trapping + ejection**: tracers get trapped in vortex cores (slow, confined motion) and then abruptly ejected into the background straining field, producing long waiting times and large jumps — a Continuous-Time Random Walk (CTRW) picture.
2. **Spatial intermittency of the strain field**: rare regions of extreme strain produce large sudden displacements.
3. **Lévy flights from the velocity spectrum**: if the energy spectrum has a power-law tail in k, the velocity increments (which drive tracer displacements over short times) have power-law tails.
4. **Inverse cascade clustering**: in 2D turbulence, energy cascades to large scales, creating large-scale correlated flows that trap tracers for long periods.

## Dataset

A 2D incompressible Navier-Stokes simulation was run on a 256×256 periodic domain (L=2π) using:
- Pseudo-spectral Poisson solver for the streamfunction
- Adams-Bashforth 2nd order time integration, dt=0.02
- Stochastic large-scale forcing in wavenumber band k ∈ [3,6]
- Kinematic viscosity ν = 0.002
- Total production time T = 600 (30000 steps)
- 8000 Lagrangian tracers seeded uniformly after 1000-step spin-up
- MSD exponent H ≈ 0.88 (clearly superdiffusive)

### Files (all absolute paths)

- `/home/node/work/projects/ns2d_levy_v1/data/traj_x.npy`  
  Shape: (1500, 8000), dtype: float32  
  Tracer x-positions at each saved time step. Positions are in [0, 2π] (periodic domain).

- `/home/node/work/projects/ns2d_levy_v1/data/traj_y.npy`  
  Shape: (1500, 8000), dtype: float32  
  Tracer y-positions at each saved time step.

- `/home/node/work/projects/ns2d_levy_v1/data/t_traj.npy`  
  Shape: (1500,), dtype: float32  
  Time values corresponding to each snapshot row in traj_x/traj_y. Range: [0, 600], spacing = 0.4.

- `/home/node/work/projects/ns2d_levy_v1/data/vorticity_snapshots.npy`  
  Shape: (15, 256, 256), dtype: float32  
  Eulerian vorticity field ω(x,y) at 15 times (every 2000 production steps × dt=0.02 × 20 = t=40 apart).

- `/home/node/work/projects/ns2d_levy_v1/data/snap_times.npy`  
  Shape: (15,), dtype: float64  
  Times corresponding to each vorticity snapshot.

- `/home/node/work/projects/ns2d_levy_v1/data/meta.json`  
  Simulation metadata (N, L, nu, dt, force_band, etc.)

- `/home/node/work/projects/ns2d_levy_v1/data/generate_ns2d.py`  
  Python script that generated the data (pseudo-spectral NS solver).

## Suggested Analyses (ordered by priority)

1. **Displacement PDF tails**: Compute displacement distributions P(Δx, τ) at multiple lags τ. Use characteristic function fitting (NOT MSD alone — MSD is undefined for ideal Lévy with α<2) to estimate the stable index α(τ). Show the evolution from Gaussian (short τ) toward heavy-tailed (long τ).

2. **Velocity increment PDF**: Compute single-tracer velocity increments δv(τ) = v(t+τ) - v(t). If these are power-law distributed, this is the direct velocity-field mechanism for heavy tails.

3. **Lagrangian velocity autocorrelation**: Compute R_v(τ). Long tails in R_v(τ) (slow decorrelation) produce anomalous diffusion via the Green-Kubo relation: MSD ~ ∫_0^T ∫_0^T R_v(|t-s|) dt ds.

4. **Trapping time analysis**: Define a "trapping event" as a period during which a tracer's net displacement falls below a threshold. Measure the distribution of trapping times P(τ_trap). A power-law P(τ_trap) ~ τ^{-μ} with μ < 2 is the CTRW signature of Lévy flights.

5. **Jump size distribution**: Complementary to trapping — measure the distribution of displacement sizes during "free flight" periods. Power-law tails here indicate Lévy flights in the classic sense.

6. **Vorticity field structure**: Using the vorticity snapshots, compute the PDF of local vorticity ω. Count coherent vortex structures (|ω| > threshold). Test whether tracers near strong vortices have different transport statistics than tracers in the background straining regions.

7. **Conditional MSD**: Split tracers into those that spend time near coherent vortices vs. those in strain-dominated regions (if vortex centres can be identified from vorticity snapshots that overlap with tracer positions). Compare their displacement PDFs.

8. **Energy spectrum**: From the vorticity snapshots, compute E(k) = (1/2)|û(k)|². Check for a k^{-5/3} (enstrophy cascade, small scales) or k^{-3} (inverse energy cascade, large scales) spectrum. The spectral index determines the Eulerian roughness ξ and the theoretical α = 2/ξ prediction.

## Key Question for the Analysis

After all analyses are complete, the report should answer: **which physical process most directly causes the heavy tails?** Options:
- Long trapping near coherent vortices (CTRW mechanism)
- Heavy-tailed velocity increments (direct Lévy mechanism)
- Long memory in Lagrangian velocity (fractional Brownian motion mechanism)
- A combination, and if so, which dominates

The "effective physical theory" is: whichever of these mechanisms dominates defines the correct stochastic model (CTRW with power-law waiting times, Lévy stable process, fractional Brownian motion, etc.).
