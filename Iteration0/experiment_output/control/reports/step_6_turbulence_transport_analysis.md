<!-- filename: reports/step_6_turbulence_transport_analysis.md -->
### Disentangling Lévy Flights from Vortex Trapping in 2D Turbulence via Lagrangian-Vorticity State Partitioning: Results and Mechanistic Synthesis

**1. Global Lagrangian Transport and Anomalous Diffusion**

The Lagrangian tracers in the simulated 2D Navier-Stokes flow exhibit pronounced and robust superdiffusive behavior. The global Mean Squared Displacement (MSD) scales according to <code>\langle \Delta \mathbf{x}^2(\tau) \rangle \sim \tau^{2H}</code>, with an estimated Hurst exponent of <code>H \approx 0.976</code>. This value is exceptionally close to the ballistic limit (<code>H=1</code>), indicating highly persistent trajectories and strong anomalous diffusion that vastly exceeds standard Brownian motion (<code>H=0.5</code>).

The displacement probability density functions (PDFs), <code>P(\Delta x, \tau)</code>, exhibit distinct non-Gaussian heavy tails at intermediate lag times, a hallmark of anomalous transport in 2D turbulence. The evolution of the stable index <code>\alpha(\tau)</code>, estimated via empirical characteristic function fitting, captures this transition. While short-time displacements are constrained by the local velocity field, intermediate lags exhibit <code>\alpha < 2</code>, reflecting the transient emergence of heavy tails. However, to identify the physical origin of these tails and the near-ballistic MSD, we must distinguish between three competing mechanistic hypotheses: independent heavy-tailed increments (direct Lévy flights), power-law trapping times with sudden ejections (Continuous-Time Random Walk, CTRW), and long-range temporal correlations in the velocity field (fractional Brownian motion, fBm).

**2. Eulerian Flow Topology and Spectral Characteristics**

To understand the environment driving the tracers, we characterized the Eulerian velocity field using the isotropic energy spectrum <code>E(k)</code>. The spectral slope in the large-scale range (<code>k \in [1, 3]</code>) is approximately <code>-4.70</code>. This slope is steeper than the classical <code>k^{-3}</code> enstrophy cascade or the <code>k^{-5/3}</code> inverse energy cascade, likely reflecting the specific stochastic forcing band (<code>k \in [3, 6]</code>) and the accumulation of energy at the largest scales of the periodic domain (domain-scale condensation). The small-scale spectrum (<code>k \in [10, 40]</code>) exhibits a very steep roll-off (slope <code>\approx -26.8</code>), indicative of strong viscous dissipation and the absence of small-scale roughness.

To partition the flow into distinct dynamical regions, we utilized the Okubo-Weiss parameter <code>Q = s^2 - \omega^2</code>, where <code>s</code> is the strain rate and <code>\omega</code> is the vorticity. Regions with <code>Q < 0</code> are rotation-dominated (coherent vortex cores), while regions with <code>Q > 0</code> are strain-dominated (hyperbolic background flow). Across the 15 Eulerian snapshots, the spatial area is roughly evenly divided. Consequently, the Lagrangian tracers spend <code>46.9%</code> of their total trajectory time in vortex regions and <code>53.1%</code> in strain regions. Furthermore, <code>48.1%</code> of the tracers are predominantly (spending <code>>50%</code> of their trajectory) located in vortex regions, while <code>51.9%</code> are predominantly in strain regions. This balanced partitioning validates that the tracers thoroughly sample both topological states of the turbulence.

**3. Tracer State Partitioning and Conditional Statistics**

By mapping the tracers to the Eulerian grid and assigning a continuous-time state (Vortex or Strain) based on the local <code>Q</code> value, we computed conditional transport statistics. The temporal resolution mismatch between the Eulerian snapshots (<code>\Delta t = 40.0</code>) and the Lagrangian trajectories (<code>\Delta t = 0.4</code>) was handled via nearest-neighbor temporal interpolation, providing a coarse-grained but statistically robust state assignment.

The conditional MSD for both the Vortex and Strain populations reveals a critical finding: superdiffusion is not exclusive to one specific flow topology. Both populations exhibit persistent, superdiffusive scaling that mirrors the global <code>H \approx 0.976</code> exponent. Furthermore, the root-mean-square (RMS) velocities are remarkably similar between the two states (<code>v_{rms}^{vortex} \approx 0.00101</code>, <code>v_{rms}^{strain} \approx 0.00106</code>). This suggests that the kinetic energy is relatively homogeneously distributed between coherent structures and the background strain field, and that the heavy tails are an intrinsic property of the global flow rather than an artifact of localized topological extremes.

**4. Evaluating the Direct Lévy Mechanism**

The direct Lévy flight hypothesis posits that heavy-tailed displacement PDFs arise directly from heavy-tailed velocity increments in the Eulerian field. If the velocity field contains extreme spatial intermittency, tracers will experience sudden, massive velocity kicks.

We tested this by computing the velocity increment PDFs <code>\delta v(\tau) = v(t+\tau) - v(t)</code> for various lags. A power-law tail fit to the extreme values (<code>>95</code>th percentile) yielded tail exponents <code>\beta</code> ranging from <code>3.52</code> to <code>4.01</code> for lags between <code>\tau=0.4</code> and <code>\tau=40.0</code>. Because these exponents are strictly and significantly greater than 3, the variance of the velocity increments is well-defined and finite. Consequently, the velocity increments do not belong to the Lévy stable domain of attraction (which requires <code>\beta < 2</code>). The heavy tails in the displacement PDFs therefore cannot be attributed to a direct Lévy mechanism driven by spatial intermittency in the velocity field.

**5. Evaluating the CTRW Mechanism (Vortex Trapping and Ejection)**

The Continuous-Time Random Walk (CTRW) model explains anomalous diffusion through a combination of long waiting times (trapping in vortex cores) and sudden large jumps (ejections into the strain field). We rigorously tested this by analyzing the distribution of trapping times <code>\tau_{trap}</code> (defined as consecutive time steps spent in <code>Q < 0</code> regions) and the jump size distributions.

- **Trapping Times**: The CTRW model relies heavily on the assumption that tracers become deeply trapped, leading to a heavy-tailed distribution of waiting times <code>P(\tau_{trap}) \sim \tau_{trap}^{-\mu}</code> with <code>\mu < 2</code>. Our maximum likelihood estimation of the trapping time distribution failed to yield a valid power-law exponent in this regime, indicating an exponential or rapidly truncating tail. Trapping events do not exhibit the scale-free waiting times required by the CTRW framework to produce superdiffusion.
- **Jump Sizes**: We computed the jump size distributions for displacements occurring strictly within the strain regions and for displacements occurring immediately upon exiting a vortex core. The power-law tail exponents for these jumps are extremely high (<code>\alpha_{exit} \approx 8.86</code>, <code>\alpha_{strain} \approx 8.10</code>), confirming the complete absence of Lévy-like heavy-tailed jumps in the classical sense.
- **Vortex Ejection Correlation**: We defined "large jumps" as displacements exceeding 3 standard deviations above the mean. Out of 144,391 identified large jumps across all tracers, only <code>0.13%</code> occurred exactly at the moment of a vortex exit event, and only <code>0.76%</code> occurred within a 5-step window of an exit. Conversely, while <code>28.0%</code> of vortex exit events produce a large jump near the exit, the vast majority (<code>>99%</code>) of large jumps in the system are entirely uncorrelated with vortex ejections.

These results rigorously falsify the CTRW hypothesis for this system. The vortex trapping and ejection mechanism is not the primary driver of the observed large displacements.

**6. Mechanistic Synthesis: Fractional Brownian Motion via Inverse Cascade**

Having ruled out both the direct Lévy flight and CTRW mechanisms, we turn to the Lagrangian velocity autocorrelation function <code>R_v(\tau)</code>. The Green-Kubo formalism links macroscopic transport coefficients to microscopic velocity fluctuations: the MSD is the double time integral of the velocity autocorrelation. For standard Brownian motion, <code>R_v(\tau)</code> decays rapidly, yielding a finite integral and a linear MSD (<code>H=0.5</code>).

In our system, <code>R_v(\tau)</code> exhibits a slow, long-range decay. When <code>R_v(\tau)</code> decays slower than <code>\tau^{-1}</code>, the integral diverges, leading directly to superdiffusion (<code>H > 0.5</code>). The physical picture that emerges is one of **fractional Brownian motion (fBm)** driven by the inverse energy cascade.

In 2D turbulence, energy cascades to the largest scales, creating large-scale, temporally persistent correlated flow structures (as evidenced by the steep <code>k^{-4.70}</code> large-scale spectrum). Tracers embedded in these domain-scale sweeping motions maintain their velocity for extended periods, leading to long memory in the Lagrangian velocity. The heavy tails observed in the displacement PDFs at intermediate times are a transient manifestation of this long-range temporal correlation—tracers riding large-scale currents for varying durations—rather than the result of independent heavy-tailed increments or singular trapping/ejection events. The near-ballistic MSD exponent (<code>H \approx 0.976</code>) further corroborates that tracers are smoothly riding large-scale persistent currents rather than undergoing abrupt, uncorrelated Lévy jumps.

### Summary of Key Quantitative Results

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Global MSD Exponent (H)** | 0.976 | Near-ballistic superdiffusion; strong trajectory persistence. |
| **Velocity Increment Tail Exponent (\tau=0.4)** | 3.86 | Finite variance; rejects direct Lévy flight mechanism. |
| **Velocity Increment Tail Exponent (\tau=40.0)** | 4.01 | Finite variance; confirms lack of spatial velocity intermittency. |
| **Spectral Slope (Large-scale, k \in [1, 3])** | -4.70 | Steep inverse cascade; indicates large-scale energy condensation. |
| **Fraction of Time in Vortex (Q < 0)** | 46.9% | Tracers sample coherent structures extensively. |
| **Fraction of Time in Strain (Q > 0)** | 53.1% | Tracers sample background flow extensively. |
| **Jump Size Tail Exponent (Vortex-exit)** | 8.86 | No heavy-tailed jumps upon vortex ejection. |
| **Jump Size Tail Exponent (Strain)** | 8.10 | No heavy-tailed jumps within the strain field. |
| **Fraction of Large Jumps at Vortex Exits** | 0.13% | Large displacements are virtually uncorrelated with ejections. |
| **Fraction of Large Jumps near Vortex Exits** | 0.76% | Rejects the CTRW trapping/ejection mechanism. |

### Conclusion

The anomalous diffusion and heavy-tailed displacement PDFs observed in this 2D Navier-Stokes simulation are not caused by intermittent vortex trapping (CTRW) or heavy-tailed velocity increments (direct Lévy flights). Instead, the primary physical mechanism is the long-range temporal memory in the Lagrangian velocity field, characteristic of fractional Brownian motion. This memory is physically sustained by the large-scale correlated flows generated by the inverse energy cascade, which advect tracers persistently over long distances without abrupt decorrelation. The heavy tails in the displacement distributions are a natural consequence of integrating these long-correlated, finite-variance velocities over time.