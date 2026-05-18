# Results: Distinguishing Memory-Driven vs. Increment-Driven Superdiffusion in 2D Turbulence via Surrogate Lagrangian Analysis

## 1. Introduction

This iteration addresses three critical gaps identified in the Iteration 0 evaluation: (1) verification that H ≈ 0.976 is not a simulation artifact, (2) the surrogate trajectory test to distinguish CTRW from fBm, and (3) the Q-threshold sensitivity analysis for the trapping time exponent. All analyses are performed directly on the 2D Navier-Stokes tracer dataset (8000 tracers, T=600, N=256×256).

---

## 2. Corrected MSD and Hurst Exponent

The global MSD scaling exponent is **H = 0.847**, significantly lower than the H ≈ 0.976 reported in Iteration 0. The discrepancy was traced to a wrapping artefact in the displacement calculation used previously. With correct periodic unwrapping, the superdiffusion is genuine but more moderate (H ≈ 0.85), consistent with the known 2D turbulence regime.

Subtracting the instantaneous ensemble-mean velocity from each tracer (to remove any large-scale "conveyor belt" contribution) leaves H unchanged at **0.847**. This rules out the hypothesis that a single dominant vortex structure inflates H — the superdiffusion is a distributed property of the turbulent flow, not a finite-domain artefact.

---

## 3. Surrogate Trajectory Test

To distinguish whether the observed superdiffusion arises from long-range temporal correlations (fBm mechanism) or from the spatial distribution of velocity increments (Lévy/increment-driven mechanism), we constructed phase-randomised surrogate trajectories. For each of 500 tracers, the Fourier transform of the Lagrangian velocity time series was computed, the phases were randomised (destroying temporal memory while preserving the single-time velocity PDF and power spectrum), and the IFFT was used to generate surrogate velocities. The MSD was then computed for the original and surrogate populations.

| Quantity | Value |
|---|---|
| H (original, 500 tracers) | 0.846 |
| H (surrogate, phase-randomised) | 0.946 |
| ΔH = H_orig − H_surr | −0.100 |

**The surrogate test yields a counterintuitive result:** phase-randomising the velocity (destroying temporal memory) *increases* H from 0.846 to 0.946, rather than reducing it. This means the temporal correlations in the Lagrangian velocity field are *suppressing* superdiffusion at long times, not driving it. The naive fBm picture — where persistent velocity correlations drive superdiffusion — is not supported. Instead, the long-range velocity memory produces a regime of correlated motion that partially saturates (or plateaus) before reaching fully ballistic behaviour.

The integral Lagrangian autocorrelation time is **T_L = 76 time units**, representing 12.7% of the total simulation duration T = 600. This is long enough to produce significant non-Markovian dynamics, but not long enough to drive the system into the asymptotic Lévy regime.

---

## 4. Stable Index α(τ) — No Lévy Regime Detected

Characteristic function fitting of the displacement PDFs across lag times τ ∈ [2, 100] yields a stable index α(τ) that remains pinned at **α ≈ 2.0** at all accessible lags. The displacement PDFs show mild heavy tails relative to a Gaussian (visible as slight excess kurtosis at long lags), but not consistent with an α-stable Lévy distribution with α < 2. This is consistent with the surrogate test result: the transport is superdiffusive (H ≈ 0.85 > 0.5) but the displacement increments have finite variance (α = 2).

---

## 5. Conditional Velocity Autocorrelation: Vortex vs. Strain

The Lagrangian velocity autocorrelation R_v(τ) was computed separately for the two populations:
- **Vortex-trapped tracers** (Q < Q₀, where Q₀ = median(Q)): R_v decays rapidly, consistent with quasi-periodic orbital motion in vortex cores.
- **Strain-dominated tracers** (Q > Q₀): R_v decays more slowly, indicating persistent directional motion along strain-field streamlines.

The fraction of tracer time spent in vortex cores is **50.8%**, confirming balanced partitioning. The strain-field population exhibits the longer correlation time, consistent with the inverse-cascade "highway" picture proposed in Iteration 0.

---

## 6. Trapping Time Sensitivity Analysis

The trapping time exponent μ extracted via the Hill estimator varies strongly with the Q threshold:

| Q threshold percentile | Trapping exponent μ |
|---|---|
| 25th | 17.6 |
| 40th | 8.0 |
| 50th | 3.1 |
| 60th | 4.1 |
| 75th | 4.3 |

All values satisfy μ > 2, indicating a **finite mean trapping time** at every threshold. The CTRW criterion (μ < 2, diverging mean) is not satisfied. The Iteration 0 result of μ ≈ 0.57 was an artefact of a non-robust Q threshold choice. The vortex trapping in this simulation does not produce the scale-free waiting time distribution required for classical CTRW anomalous diffusion.

---

## 7. Mechanistic Synthesis

The combined evidence from this iteration substantially revises the Iteration 0 conclusions:

1. **CTRW mechanism: not supported.** The trapping time exponent μ > 2 at all robust Q thresholds. Vortex trapping does not produce scale-free waiting times.

2. **Direct Lévy flight mechanism: not supported.** The stable index α(τ) ≈ 2 at all lags. Displacement PDFs have finite variance.

3. **fBm (velocity memory) mechanism: not supported as primary driver.** The surrogate test shows that temporal memory *suppresses* rather than drives superdiffusion (ΔH = −0.10).

4. **What is actually happening:** The H ≈ 0.85 superdiffusion arises from a *pre-asymptotic ballistic regime*. The long Lagrangian autocorrelation time (T_L ≈ 76) means that tracers are still partially ballistic over much of the simulation window T = 600 ≈ 8 × T_L. The true asymptotic diffusion regime (where the Green-Kubo integral has converged) has not been reached. The "superdiffusion" is not a genuine anomalous diffusion fixed point — it is a finite-time crossover from ballistic to diffusive motion.

**Central conclusion:** In this 2D NS simulation, with forcing at wavenumbers k ∈ [3, 6] and ν = 0.002, the Lagrangian tracers do not enter a genuine Lévy-flight or CTRW regime within T = 600 time units. The observed H ≈ 0.85 reflects pre-asymptotic ballistic transport driven by long-lived large-scale vortices generated by the inverse energy cascade. To access the true anomalous diffusion regime (if it exists), either:
- A significantly longer simulation (T ≫ T_L) is needed, or
- A simulation with a shorter T_L (smaller-scale forcing, or a more turbulent regime with higher Re) is required.

The physical mechanism for Lévy flights in 2D turbulence — if present — would require demonstrating scale-free trapping times (μ < 2) at a robust vortex identification threshold, or a clear α < 2 in the displacement characteristic function. Neither is observed here.
