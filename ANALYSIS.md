# Detailed analysis and numerical reconstruction

## 1. What the paper establishes

The central result is not merely that polymer filaments neck. It is that a filament-stretching rheometer can still measure a steady extensional stress, and can also measure stress relaxation, when the plate motion is continually adjusted so that the **true Hencky strain history at the filament symmetry plane** follows the prescribed history.

The distinction between strain measures is essential:

\[
\epsilon_N(t)=\ln\!\left(\frac{L(t)}{L_0}\right),
\qquad
\epsilon(t)=2\ln\!\left(\frac{R_0}{R(t)}\right).
\]

The nominal strain is determined by the plate separation. The true strain is inferred from the mid-filament radius and, under incompressibility, is the local axial Hencky strain. In a homogeneous affine extension, the two are equal. In an unstable or end-affected filament, they need not be close and may even evolve in opposite directions.

The model sequence is deliberately chosen to span very different extensional rheology:

1. **Newtonian liquid:** no elastic stress, broad and relatively gentle necking.
2. **Oldroyd-B:** unlimited strain hardening and a steady extensional-stress singularity at `Wi=0.5` in the scalar toy formulation; the filament is nearly homogeneous at `Wi=1` but no steady stress exists.
3. **Non-stretching Rolie–Poly toy:** elastic stress saturation, rapid localized necking, and reverse plate motion under control.
4. **Non-stretching DE-toy:** an integral model with the same stress-saturation mechanism and essentially the same localized-neck behavior.
5. **Stretching DE-toy:** molecular stretch stabilizes startup when `Wi_R>1`; after cessation, molecular stretch relaxes and the plates must move strongly backward to keep the true strain fixed.

This progression separates two issues that can otherwise be conflated:

- whether a constitutive model has a homogeneous steady extensional stress;
- whether a controlled instrument can hold the local kinematics long enough to measure it.

For the stress-saturating models, the answer is “yes” because stress reaches its plateau at approximately one Hencky strain, while catastrophic localization develops closer to two Hencky strains. The measurement window exists even though the bridge is ultimately unstable.

---

## 2. Differential models in the Eulerian frame

### 2.1 Thin-filament equations

The physical one-dimensional equations are

\[
\frac{\partial A}{\partial t}+V\frac{\partial A}{\partial z}
=-\frac{\partial V}{\partial z}A,
\]

\[
\frac{\partial}{\partial z}\left(A\sigma_E\right)=0,
\]

with a constitutive axial-minus-radial stress. The transformation

\[
u=z e^{-\epsilon_N},\qquad
v=V e^{-\epsilon_N},\qquad
a=Ae^{\epsilon_N}
\]

maps the changing physical interval to `u in [0,1]`. The transformed equations implemented here are

\[
\frac{\partial a}{\partial t}
+(v-\dot\epsilon_Nu)\frac{\partial a}{\partial u}
=-\left(\frac{\partial v}{\partial u}-\dot\epsilon_N\right)a,
\]

\[
\frac{\partial}{\partial u}
\left[a\left(G_0Z+\eta_s\frac{\partial v}{\partial u}\right)\right]=0,
\]

\[
\frac{\partial Z}{\partial t}
+(v-\dot\epsilon_Nu)\frac{\partial Z}{\partial u}
=\frac{\partial v}{\partial u}f(Z)-\frac{Z}{\tau},
\]

with

\[
f(Z)=2+2Z-\beta Z^2.
\]

`beta=0` gives the Oldroyd-B toy. Positive `beta` gives the non-stretching Rolie–Poly toy used in the paper.

### 2.2 Force balance and reconstructed ideal control

The transmitted engineering tension

\[
T=a\left(G_0Z+\eta(u,t)v_u\right)
\]

is constant along the filament. The original Eulerian controller was not available for this reconstruction, which therefore enforces the desired local rate exactly at the midplane:

\[
v_u(u=1/2)=\dot\epsilon_{\rm target}.
\]

This gives

\[
T=a_m\left(G_0Z_m+\eta_m\dot\epsilon_{\rm target}\right),
\]

and then

\[
\dot\epsilon_N
=T\int_0^1\frac{du}{a\eta}
-G_0\int_0^1\frac{Z}{\eta}\,du.
\]

The complete velocity-gradient field follows from

\[
v_u=\frac{T/a-G_0Z}{\eta},
\]

and integration with `v(0)=0`.

This algebraic step avoids a separate linear solve for velocity and makes the control mechanism transparent: the plate rate is whatever value is needed to produce the prescribed rate at the center.

### 2.3 Spatial and temporal discretization

The authors' numerical notes use:

- forward Euler for time derivatives;
- forward spatial differences where `v-dot(epsilon_N)u<0`;
- backward spatial differences where the coefficient is positive;
- an implicit update for `-Z/tau`;
- optional artificial diffusion for suppressing oscillations in `Z`.

The code follows that scheme. The printed CFL expression in the notes is dimensionally reversed; the implemented condition is the standard

\[
\Delta t=C\frac{\Delta u}{\dot\epsilon_{\rm target}},\qquad C\le 1.
\]

The standard reproduction uses `C=0.5`, matching the stated Courant number.

### 2.4 Endplate correction

No slip is not imposed as a direct boundary condition in the one-dimensional model. Instead, a Stokes-type divergent apparent viscosity is used near both plates:

\[
\eta(u,t)=\eta_s\left[1+\frac{c_s}{32}
\left(\frac{R_0e^{-\epsilon_N}}{u}\right)^2
+\frac{c_s}{32}
\left(\frac{R_0e^{-\epsilon_N}}{1-u}\right)^2\right].
\]

This suppresses axial deformation adjacent to the plates and generates the familiar unit-radius shoulders in the computed profiles. The authors' MATLAB helpers use the same functional form, but the production Eulerian coefficient and driver were not available to the reconstruction. The preset coefficient is therefore exposed rather than buried (its sensitivity is tabulated in Section 10.2).

---

## 3. Analytical results for the scalar Rolie–Poly toy

For homogeneous constant-rate extension, the steady state satisfies

\[
0=Wi\left(2+2Z-\beta Z^2\right)-Z.
\]

For `beta>0`, the positive root is

\[
Z_{ss}=
\frac{2Wi-1+\sqrt{(1-2Wi)^2+8\beta Wi^2}}
{2\beta Wi}.
\]

For `beta=0`,

\[
Z_{ss}=\frac{2Wi}{1-2Wi},
\]

which diverges at `Wi=0.5`. At high rate and positive `beta`,

\[
Z_{ss}\rightarrow \frac{1+\sqrt{1+2\beta}}{\beta}.
\]

For the paper’s `beta=0.4`, the limit is `5.8541`. At `Wi=2`, the polymer contribution is `4.79315`. With a 20% solvent fraction under the transformed Eulerian convention, the total steady stress is

\[
4.79315+0.25\times2=5.29315,
\]

which is the horizontal level in Figure 4(b).

The startup curve is obtained by integrating

\[
\dot Z=\dot\epsilon(2+2Z-\beta Z^2)-Z/\tau.
\]

The reconstructed curve reaches the steady value at roughly one relaxation time, before the filament undergoes its most severe geometric localization.

---

## 4. Integral models in the Lagrangian frame

### 4.1 Why a material-coordinate method is natural

Integral constitutive equations require the deformation from every earlier time `t'` to the current time `t`. In a Lagrangian mesh, each material segment retains its complete length history, so its Finger tensor can be reconstructed directly.

The unknowns at each step are:

- all current particle positions `z_i`;
- one common engineering force `F/A0`;
- one molecular stretch per segment when stretch is enabled.

For `n_z` particles there are `n_z-1` segment-force equations, one lower-plate condition, and one global kinematic/control equation. With stretch, another `n_z-1` implicit stretch equations are added.

### 4.2 Oldroyd-B integral form

The integral Oldroyd-B model is

\[
\boldsymbol\sigma(t)=
\int_{-\infty}^{t}M(t-t')\mathbf B(t,t')\,dt'
+\eta_s\dot{\boldsymbol\gamma}.
\]

For a material segment with current length `d`, earlier length `d_k`, and initial length `d_0`, the axial-minus-radial Finger contribution to engineering force is

\[
q_{\rm OB}=
\frac{d d_0}{d_k^2}-\frac{d_kd_0}{d^2}.
\]

The Python implementation uses the same expression and its exact derivative with respect to the current segment length.

### 4.3 DE-toy orientation saturation

The DE-toy replaces `B` by

\[
\mathbf Q=\frac{\mathbf B}{\operatorname{tr}\mathbf B}.
\]

For a segment, the normalized axial-minus-radial engineering-force contribution is

\[
q_{\rm DE}=
\frac{d d_0/d_k^2-d_kd_0/d^2}
{(d/d_k)^2+2d_k/d}.
\]

The polymer stress is

\[
\boldsymbol\sigma_p=3G_0\lambda^2\mathbf S.
\]

Without molecular stretch, the steady uniaxial orientation difference for a single Maxwell mode is

\[
D(Wi)=\int_0^\infty e^{-x}
\frac{1-e^{-3Wi x}}{1+2e^{-3Wi x}}\,dx.
\]

The code evaluates the equivalent exact hypergeometric form

\[
D(Wi)=-\frac12+rac32\,{}_2F_1
\left(1,\frac{1}{3Wi};1+\frac{1}{3Wi};-2\right).
\]

The non-stretching stress is `3D(Wi)` and tends to `3G0`, reproducing Figure 6(a).

### 4.4 Molecular stretch

The stretch equation is

\[
\dot\lambda=\lambda(\boldsymbol\kappa:\mathbf S)
-\frac{f(\lambda)}{\tau_s}(\lambda-1),
\]

with the Wagner function

\[
f(\lambda)=1-\frac{2p}{3}
+\frac{2p}{9}(\lambda^4+\lambda^3+\lambda^2),
\qquad p=0.3.
\]

In homogeneous steady extension,

\[
\lambda D(Wi)Wi_R=f(\lambda)(\lambda-1).
\]

For the long-time protocol (`Wi=30`, `Wi_R=3`), the analytical solution gives `lambda=2.5067` and a total steady stress near `19.42 G0`. The full filament simulation reaches a midplane stretch of approximately `2.51`, in agreement with the mechanism discussed in the paper.

### 4.5 Sparse Newton translation

The original MATLAB program allocates a dense `2n_z x 2n_z` Jacobian even though each segment couples only to its two neighboring nodes, the common force, and its own stretch. The Python code assembles the same derivatives as a sparse matrix and solves it with `scipy.sparse.linalg.spsolve`.

This does not alter the equations. It changes only storage and linear algebra. The standard Figure 9–10 case (`201` nodes, `501` time levels) runs in seconds rather than requiring repeated dense solves.

---

## 5. Memory functions

### 5.1 Single-mode Maxwell memory

The original memory helpers define exactly

\[
M(s)=\frac{G_0}{\tau}e^{-s/\tau},
\qquad
\int_s^\infty M(q)dq=G_0e^{-s/\tau}.
\]

These are implemented as a one-mode Prony spectrum.

### 5.2 Continuous BSW spectrum

The paper writes

\[
M(t)=\int_0^\infty H(\tau)\frac{e^{-t/\tau}}{\tau^2}d\tau,
\]

with entanglement and glassy branches. The code converts this to positive Prony modes by quadrature in `ln(tau)`.

For the entanglement branch,

\[
H_e(\tau)=n_eG_0(\tau/\tau_m)^{n_e},\qquad \tau\le\tau_m,
\]

and

\[
\tau_d=\tau_m\frac{n_e}{1+n_e}.
\]

The exponent values `n_e=0.23` and `n_g=0.70` were recovered from the cited polystyrene study. Neither the paper nor the original MATLAB program gives `tau_c/tau_m`. The default `tau_c/tau_m=10^-3` is therefore an explicit reconstruction calibrated to the approximately `1.55 G0` steady level in Figure 8. Changing it is a one-line parameter edit.

The broad BSW spectrum changes the stress-growth details but does not remove the localized neck. This supports the paper’s conclusion that the severe instability is controlled mainly by nonlinear stress saturation rather than by whether the linear spectrum is narrow or broad.

---

## 6. Figure-by-figure interpretation and reproduction

### Figure 2 — Newtonian

The true strain follows `epsilon=t`. The nominal strain lags, ending near `1.58` in the standard reconstruction when the true strain is `2`. The profiles develop a broad neck rather than a sharply localized notch. This is the mild stress-curvature mode.

### Figure 3 — Oldroyd-B

At `Wi=1`, both Eulerian and Lagrangian methods produce nearly affine extension except for narrow end transitions. The reconstructed final nominal strains are `1.925` and `1.930`, respectively, while the true strain is `2`. The profile overlay is a strong independent check because the methods use different coordinates and constitutive implementations.

### Figure 4 — Rolie–Poly stress

The analytical steady curves show the Oldroyd-B singularity and finite plateaus for positive `beta`. At `beta=0.4`, `Wi=2`, and 20% solvent, the exact reconstructed steady stress is `5.29315 G0`.

### Figure 5 — Rolie–Poly necking

The true strain increases to `4`. The nominal strain first increases to about `1.57`, then falls to about `1.22`: the plates reverse while the center continues to stretch. The final center radius is approximately `exp(-4/2)=0.1353`, and the neck becomes very narrow.

### Figure 6 — DE-toy stress

The non-stretching stress saturates at `3G0`. With `Wi_R=0.1Wi`, molecular stretch produces strong strain hardening. At `Wi=2`, 6% solvent, and negligible stretch, the analytical steady stress is `2.70921 G0`.

### Figure 7 — Single-mode DE-toy filament

The geometry closely mirrors the non-stretching Rolie–Poly case. With the default second-order scheme (Section 9) the nominal strain peaks near `1.40` at `t≈1.05` and ends near `1.14` while the true strain reaches `4`; the dt-converged values are `1.42` and `1.22`. The paper shows a peak of about `1.4` and an end value of about `1.15`. Differences of a few hundredths are expected from the missing production controller and exact Stokes settings.

### Figure 8 — BSW DE-toy filament

The BSW reconstruction gives an analytical steady stress of `1.548 G0`, matching the plotted level; the simulated mid-filament stress reaches `1.545 G0` at `t=2` once sub-step modes are lumped (Section 9.4). The radial profiles are very similar to the single-mode case. The final nominal strain is about `1.16`.

### Figures 9–10 — Startup and relaxation with stretch

The protocol is `dot(epsilon)=0.03 s^-1` for `100 s`, followed by constant true strain to `500 s`, with `tau=1000 s`, `tau_s=100 s`, and 1% solvent.

The reconstruction gives:

- analytical steady stress: approximately `19.42 G0`;
- simulated peak stress: approximately `19.55 G0`;
- simulated stress at the end of stretching (`t=100 s`): `19.35 G0` — slightly below the fully steady value because the orientation of a `1000 s` mode is only 90% relaxed from its equilibrium pre-history after `100 s`;
- nominal strain at `100 s`: approximately `2.70`;
- nominal strain at `500 s`: approximately `1.41` (dt-converged `1.415`; the paper shows about `1.45`);
- maximum midplane molecular stretch: approximately `2.50`.

During startup the filament is comparatively uniform because molecular stretch hardens and stabilizes it. During relaxation, stretch collapses over roughly one stretch-relaxation time. The controller then pulls the plates backward so that the midplane strain remains exactly `3`, while the stress falls rapidly and the deformation redistributes.

---

## 7. Relation to the original MATLAB program

The Lagrangian solver is a Python translation of the authors' original MATLAB Lagrangian driver. That program is not part of this repository; the notes below record the implementation choices inherited from it, so that the Python code can be read on its own.

### 7.1 Feedback controller

The original driver closes the loop on the true Hencky strain with the update

\[
\Delta\epsilon_N=
\frac{\epsilon_{ideal}-\epsilon}
{1+|\epsilon-\epsilon_N|}.
\]

The Python solver retains this as `control="matlab_feedback"`, but uses `control="exact_midplane"` for clean paper reproduction, since the paper states that the detailed control loop is not its focus.

### 7.2 Extra half-step in the memory quadrature

The original driver evaluated the memory function half a step later than the deformation it multiplies (an "extra 0.5 dt" experiment). The translation retains this in `time_discretization="matlab_first_order"`. It is one of three first-order errors quantified in Section 9; the default scheme removes them and the stress-only panels are drawn from the filament simulations.

### 7.3 Dense versus sparse Newton matrix

The original program assembles a dense `2n_z x 2n_z` Jacobian even though each segment couples only to its two neighbouring nodes, the common force, and its own stretch. The Python code assembles the same derivatives as a sparse matrix (Section 4.5); this changes only storage and linear algebra.

### 7.4 Solvent discretization

The original solvent force uses the new segment length in the numerator but the old length squared in the denominator. This is preserved in the `matlab_first_order` scheme. At `Wi=2`, `dt=0.05` it overestimates the solvent force by about 16% (Section 9.2); the default scheme replaces it by a BDF2 Hencky rate at the new time level.

### 7.5 Factor-of-three inconsistency in Appendix A

The physical stress equation contains `3 eta_s V_z`, while the transformed equations display `eta_s v_u`. Figure 4(b) matches the latter convention exactly. The code therefore uses:

- solvent factor `1` for differential-model Figure 4;
- solvent factor `3` for the integral/Lagrangian models, matching the original Lagrangian residual and analytical stress calculation.

This is documented in configuration rather than silently reconciled.

## 8. Validation and limitations

The code validates four independent levels:

1. **Analytical limits:** Rolie high-rate saturation and DE saturation at `3G0`.
2. **Exact stress levels:** `5.29315 G0` for Figure 4(b), `2.70921 G0` for Figure 6(b), and approximately `19.42 G0` for Figure 9.
3. **Method comparison:** Eulerian and Lagrangian Oldroyd-B profiles nearly superpose.
4. **Kinematic control:** the exact Lagrangian controller tracks the prescribed midplane strain to machine precision.

Remaining limitations are explicit:

- surface tension, inertia, gravity, fracture, and non-axisymmetric modes are excluded, as in the paper’s slender-filament analysis;
- the production Eulerian code and control tuning are missing;
- the exact BSW crossover time is missing;
- the Stokes end correction is an approximation, not a resolved no-slip plate flow;
- first-order history and solvent discretizations prevent claims of bitwise identity with an unavailable original run.

Within those limits, the package reproduces all computational phenomena, the reported parameter regimes, the key stress plateaus, the plate reversal, the narrow-neck profiles, and the molecular-stretch relaxation mechanism.

---

## 9. Time-discretization audit (version 1.1)

### 9.1 Symptom

The v1.0 Lagrangian solver, run with the paper's `dt=0.05`, gave a mid-filament stress of `2.82 G0` at `t=2` for the Figure 6(b) preset against the analytical steady value `2.709 G0`, and `1.616 G0` against `1.548 G0` for Figure 8(a). The paper's blue curves lie on the red analytical lines. Halving `dt` halved the discrepancy (`+0.229, +0.110, +0.054, +0.027` for `dt = 0.1, 0.05, 0.025, 0.0125`), so the discrepancy is a first-order time-discretization error of the original scheme and not a modelling difference. Its three sources are listed below, each with its second-order replacement. The literal scheme is kept as `time_discretization="matlab_first_order"`.

### 9.2 Sources and replacements

1. **Memory quadrature.** The original driver evaluates `M` at ages `(j-k+1/2)dt` but multiplies it by the deformation from level `k`, i.e. from the *start* of that slice, so the memory lags the deformation by half a step. The pre-history survival term is likewise taken at age `(j+3/2)dt` instead of `(j+1)dt`. The replacement is the trapezoidal rule on the nodes `t'_k = k dt`, `k = 0..j+1`: the node `t'=t` contributes nothing because `Q(t,t)=0`, so the only change is a half weight on `k=0`, and the survival term is evaluated at age `t`. (An un-offset variant, ages `(j-k)dt`, is a left-endpoint rule and overshoots even more: `2.88 G0`.)

2. **Solvent force.** `3η d0 (d-d_old)/(dt d_old²)` evaluates the geometric factor `1/d` explicitly. Over one step at `Wi=2`, `dt=0.05` the ratio to the exact `3η ε̇ d0/d` is `(e^{0.1}-1)/0.1 · e^{0.1} ≈ 1.16`. A midpoint rate `ln(d/d_old)/dt` with the implicit `1/d` fixes the magnitude but sits half a step behind the quasi-static force balance at `t_{n+1}` and turned out to be less dissipative under the Considère instability; the adopted form is the second-order backward difference (BDF2) of `ln d` at the new level, with backward Euler on the first step from rest.

3. **Stretch equation.** The original backward-Euler update uses the Wagner function at the old stretch. The replacement is a θ-method with `θ=1/2` (Crank–Nicolson) when `dt ≤ τ_s` and `θ=1` when the equation is stiff (`τ_s ≪ dt`, where `λ = 1 + O(τ_s)` and the damped scheme is preferable).

Two supporting changes: the Stokes end-correction viscosity is evaluated at geometrically extrapolated positions (O(dt²)) rather than the old positions, and the Newton iteration starts from the same extrapolated predictor, which reduces the first correction from O(1) to O(dt²) and removes the convergence failures that the log-rate form otherwise showed at `dt=0.1`.

### 9.3 Convergence

`convergence_study.py` (Figure 6(b)/7 preset, 201 nodes):

| scheme | dt | σ(t=2)/G0 | error | peak ε_N | final ε_N |
|---|---|---|---|---|---|
| matlab_first_order | 0.1 | 2.939 | +0.230 | 1.339 | 1.180 |
| matlab_first_order | 0.05 | 2.819 | +0.110 | 1.376 | 1.197 |
| matlab_first_order | 0.025 | 2.763 | +0.054 | 1.397 | 1.209 |
| matlab_first_order | 0.0125 | 2.736 | +0.027 | 1.409 | 1.216 |
| second_order | 0.1 | 2.704 | −0.005 | 1.354 | 1.127 |
| second_order | 0.05 | 2.708 | −0.0011 | 1.405 | 1.195 |
| second_order | 0.025 | 2.709 | −0.00014 | 1.418 | 1.216 |
| second_order | 0.0125 | 2.709 | +0.0001 | 1.421 | 1.221 |

The stress error falls as `dt²`, and both schemes converge to the same geometry (`ε_N` peak ≈ 1.42 at `t ≈ 1.07`, final ≈ 1.22). The Lagrangian Oldroyd-B run of Figure 3 agrees with the homogeneous Oldroyd-B solution to five digits (`14.019 G0` at `t=2`) with the default scheme. For the relaxation protocol of Figures 9–10 the nominal strain at `500 s` is `1.407, 1.413, 1.415` for `dt = 1, 0.5, 0.25 s`.

### 9.4 Sub-step relaxation modes

The BSW quadrature contains modes with `τ_k` far below `dt`. No time quadrature can resolve them, and with the trapezoidal rule they contribute nothing, so the Figure 8 stress initially undershot the analytical value (`1.514` vs `1.548 G0`). As `τ_k → 0` a Maxwell mode of either integral model reduces exactly to a Newtonian contribution `3 G0 w_k τ_k ε̇` (for the DE-toy, `(B_zz-B_rr)/tr B → ε` at small strain). Modes with `τ_k < fast_mode_cutoff·dt` (default cutoff 1) are therefore lumped into the solvent viscosity and removed from the quadrature; the lumped viscosity is `0.0074 G0·τ_d` for the standard preset and is reported in `LagrangianResult.lumped_fast_mode_viscosity`. The BSW stress then converges to the analytical value (`1.5448, 1.5468, 1.5475` for `dt = 0.05, 0.025, 0.0125`).

### 9.5 Spatial resolution of the late-stage neck

With the second-order scheme at `dt = 0.05`, node refinement of the Figure 7 preset gives

| nodes | peak ε_N | ε_N(1.6) | ε_N(2) | σ(2)/G0 | neck segments at t=2 |
|---|---|---|---|---|---|
| 101 | 1.4049 | 1.2468 | 1.3175 | 2.7080 | 2 |
| 201 | 1.4047 | 1.2273 | 1.1954 | 2.7081 | 2 |
| 401 | 1.4046 | 1.2228 | 1.1384 | 2.7081 | 2 |
| 801 | 1.4046 | 1.2217 | 1.1170 | 2.7082 | 4 |
| 1601 | 1.4046 | 1.2214 | 1.1111 | 2.7082 | 8 |

Everything up to `t ≈ 1.6` is converged at the standard 401 nodes. The final value at `t = 2`, when the neck radius is `0.135` and only a few uniform material segments span it, converges only linearly (to about `1.10`); the standard preset is within `0.03` of that limit, the `fast` preset (101 nodes) is not, so the fast preset should be read as a smoke test only. This late stage is the one the paper describes as the point at which the simulations were terminated. `LagrangianSnapshot.neck_segments` and the Figure 7/8 metrics report the resolution.

### 9.6 Consequence for the figure panels (Lagrangian)

Figures 4(b), 6(b) and 8(a) are now drawn from the mid-filament stress of the filament simulations, as in the paper, with the homogeneous analytical transient overlaid as an independent check; the two coincide to the line width. The metrics file records both the simulated `t=2` stress and the analytical steady value for each.

---

## 10. Eulerian solver audit (version 1.2)

### 10.1 Grid and time-step convergence

The Eulerian solver is forward Euler with first-order upwind transport and a Courant number of 0.5, as in the authors' numerical notes. Halving the Courant number changes the Figures 2, 3 and 5 strain histories by less than `2×10⁻⁴`; grid refinement from 401 to 1601 points changes the Figure 2 and 3 end values by `10⁻³` and the Figure 5 end value by `7×10⁻³` (`1.223 → 1.230`), first order in `du` as expected. The stress in Figure 4(b) agrees with the homogeneous analytical curve to `0.2%` at all resolutions. The paper's Figure 2(a) endpoint, digitized from the PDF, is `ε_N(2) ≈ 1.56`; the reconstruction gives `1.58` with `c_s = 0.30`.

### 10.2 The Stokes coefficient

`c_s` is the one Eulerian parameter not fixed by the paper or the available notes. Its effect is systematic (standard grid):

| c_s | Fig. 2 ε_N(2) | Fig. 5 peak ε_N | Fig. 5 ε_N(2) | Fig. 3 ε_N(2) |
|---|---|---|---|---|
| 0.20 | 1.647 | 1.641 | 1.288 | 1.932 |
| 0.25 | 1.613 | 1.604 | 1.252 | 1.925 |
| **0.30** | **1.584** | **1.572** | **1.223** | **1.918** |
| 0.40 | 1.533 | 1.523 | 1.177 | 1.905 |
| paper | ≈1.56 | ≈1.57 | ≈1.22 | ≈1.92 |

`c_s = 0.30` reproduces Figures 3 and 5 to within `0.01`; Figure 2 alone would prefer `≈0.35`. The `0.02` residual is within the uncertainty of the missing production settings and is left as is.

### 10.3 Volume conservation and the mid-cell controller

The non-conservative upwind form of Eq. (A18) does not conserve the discrete filament volume: the drift is `0.6%` for the Figure 5 preset at 401 points and halves with each grid doubling. Version 1.2 adds `area_scheme="conservative"`, which advances `∂a/∂t + ∂(ca)/∂u = 0` in finite-volume flux form with upwind face values (volume conserved to round-off). In that form the mid cell exchanges volume with its neighbours, so the pointwise constraint `v_u(1/2) = ε̇` no longer controls the mid-cell strain (tracking error `0.16` at 401 points). The controller is therefore closed on the *discrete* mid-cell true Hencky strain — the tension is found by a bracketed root solve so that `ln(A0/a_mid) + ε_N` equals the target at the new time level — which is what an instrument closes its loop on. Tracking is then exact to round-off.

`convergence_study.py` (Rolie–Poly preset) compares the two:

| scheme | n | final ε_N | volume err | tracking err | σ(1.5)/G0 | σ(2)/G0 | neck points |
|---|---|---|---|---|---|---|---|
| pointwise | 401 | 1.2230 | 6×10⁻³ | 2×10⁻³ | 5.279 | 5.301 | 3 |
| pointwise | 1601 | 1.2299 | 1.5×10⁻³ | 5×10⁻⁴ | 5.273 | 5.296 | 17 |
| conservative | 401 | 1.2404 | 3×10⁻¹⁶ | 9×10⁻¹⁶ | 5.291 | 5.847 | 5 |
| conservative | 1601 | 1.2305 | 0 | 4×10⁻¹⁶ | 5.274 | 5.410 | 17 |
| analytical | | | | | 5.269 | 5.291 | |

Both converge to the same geometry (`ε_N(2) ≈ 1.23`) and agree to `<0.2%` in stress up to `t ≈ 1.5`, i.e. through the window in which the steady stress is established (`t ≈ 1.3`). Beyond that the neck spans only a few grid points ("neck points" is the number of nodes with `R < 1.5 R_min`). The pointwise scheme then still reports the homogeneous stress because its constraint fixes the local rate by construction; the conservative scheme has to over-drive the unresolved mid cell to hold the measured strain and its mid stress rises. That rise is a resolution warning, not physics — it disappears as the grid is refined — and it corresponds to the paper's remark that the simulations were terminated once the neck became this narrow.

The paper's figures were produced with the Hoyle–Fielding pointwise form, so `"pointwise"` remains the default and the figure presets use it; `"conservative"` is the recommended check whenever volume conservation or the resolution of a neck is in question. `EulerianSnapshot.neck_cells` and the Figure 5 metrics report the resolution.
