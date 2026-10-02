# Changelog

## 1.3.0 - 2026-10-02

- A front page with figures: a hero animation of the single-mode DE-toy filament (neck formation, reverse
  plate motion, the stress tracking the analytical transient) and a six-image gallery, all computed by the
  package from the reproduction's own data (`make_readme_images.py`, output in `docs/images/`).
- Continuous integration (`.github/workflows/tests.yml`): the test suite and the fast reproduction on
  Python 3.10 and 3.12 with NumPy 1.26 and 2.x.
- Packaging metadata (project URLs, keywords, README as the long description), `CITATION.cff` with the
  repository URL and ORCID, a `.gitignore`, and this changelog (the version notes below were moved out of the README).

## 1.2.0 - Eulerian solver audit

`EulerianConfig.area_scheme` selects `"pointwise"` (default; the Hoyle-Fielding non-conservative upwind form with
the rate constraint `v_u(1/2) = eps_dot`, used for the paper's figures) or `"conservative"` (finite-volume flux
form, volume conserved to round-off, controller closed on the discrete mid-cell true strain so tracking is exact to
round-off). The two agree to < 0.2 % while the neck is resolved; `EulerianSnapshot.neck_cells` reports the
resolution. Grid/CFL convergence and the sensitivity to the Stokes coefficient `c_s` (the one Eulerian parameter not
fixed by the paper; `c_s = 0.30` fits Figures 3 and 5 to 0.01 and Figure 2 to 0.02) are tabulated in `ANALYSIS.md`
§10, and `convergence_study.py` covers both solvers.

## 1.1.0 - time-discretisation audit of the Lagrangian solver

Checking the 1.0 output against the paper showed that the literal translation of the original MATLAB driver
overshoots the analytical steady stress by about 4 % at `dt = 0.05` (2.82 vs 2.709 G0 in the Figure 6(b) preset),
whereas the paper's Figures 6(b) and 8(a) show the simulated stress landing on the analytical line. The overshoot
is a first-order time-discretisation error (it halves when `dt` is halved) with three sources in the original
scheme: the memory function is sampled half a step later than the deformation it multiplies, the solvent force
uses the old segment length, and the stretch equation is backward Euler with an explicit Wagner function.

`LagrangianConfig.time_discretization` now selects between:

- `"matlab_first_order"` - the original program's scheme, kept for comparison;
- `"second_order"` (default) - trapezoidal history quadrature consistent with the pre-history survival term, BDF2
  Hencky rate in the solvent term, a theta-method (Crank-Nicolson when `dt <= tau_s`) for molecular stretch, the
  Stokes end correction evaluated at extrapolated positions, and relaxation modes with `tau_k < dt` lumped into the
  solvent viscosity (exact as `tau_k -> 0`; this matters for the broad BSW spectrum). A geometric Newton predictor
  keeps the iteration in its quadratic regime.

`convergence_study.py` shows the stress error falling as `dt^2` and both schemes converging to the same neck
geometry. With the default scheme the simulated stresses for Figures 4(b), 6(b) and 8(a) sit on the analytical
curves, as in the paper, and those panels are drawn from the filament simulations with the homogeneous analytical
transient overlaid as an independent check.

## 1.0.0

First release: analytical toy-model solutions, the Eulerian and Lagrangian solvers, memory functions, the
reproduction of Figures 1-10, tests and the notebook.
