"""Eulerian slender-filament solver for Appendix A of Hassager et al. (2021).

The implementation follows Eqs. (A18)-(A20), the sign-dependent first-order
upwind discretization of the authors' numerical notes, and implicit treatment
of ``-Z/tau``.  Two area schemes are offered.  ``area_scheme="pointwise"`` (default) is the
Hoyle-Fielding form used for the paper's figures: non-conservative upwind
transport of ``a`` with the pointwise rate constraint ``v_u(1/2)=target``;
its discrete volume drifts by O(du) (0.6% for the Figure 5 preset).
``area_scheme="conservative"`` advances ``a`` in finite-volume flux form,
conserving volume to round-off, with the controller closed on the discrete
mid-cell true Hencky strain (what an instrument measures) so tracking is
exact.  The two agree to <0.2% while the neck is resolved; once it spans
only a few cells (Figure 5 for t > 1.5) the conservative scheme's mid stress
departs from the homogeneous value, which the pointwise constraint hides by
construction.  ``EulerianResult.neck_cells`` reports the resolution.  The original Eulerian driver and controller were not available for this
reconstruction.  The controller is therefore reconstructed as an ideal
constraint: at every step the axial strain rate at the symmetry plane is set
to the requested true rate, and the plate/nominal rate follows from force
balance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from .analytical import solvent_viscosity_from_fraction

FloatArray = NDArray[np.float64]
# numpy renamed trapz -> trapezoid in 2.0; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
EulerianModel = Literal["newtonian", "oldroyd_b_toy", "rolie_poly_toy"]
AreaScheme = Literal["pointwise", "conservative"]


@dataclass(frozen=True)
class EulerianConfig:
    model: EulerianModel
    target_rate: float
    end_time: float = 2.0
    tau: float = 1.0
    beta: float = 0.0
    G0: float = 1.0
    eta_r: float = 0.2
    n_points: int = 401
    cfl: float = 0.25
    artificial_diffusion: float = 0.0
    stokes_strength: float = 0.25
    initial_radius: float = 1.0
    pre_stretch: float = 0.0
    solvent_factor: float = 1.0
    area_scheme: AreaScheme = "pointwise"
    snapshot_times: tuple[float, ...] = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0)

    def validate(self) -> None:
        if self.model not in {"newtonian", "oldroyd_b_toy", "rolie_poly_toy"}:
            raise ValueError(f"Unsupported Eulerian model: {self.model}")
        if self.target_rate <= 0.0 or self.end_time <= 0.0:
            raise ValueError("target_rate and end_time must be positive.")
        if self.tau <= 0.0 or self.G0 <= 0.0:
            raise ValueError("tau and G0 must be positive.")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if not (0.0 <= self.eta_r < 1.0):
            raise ValueError("eta_r must satisfy 0 <= eta_r < 1.")
        if self.model != "newtonian" and self.eta_r <= 0.0:
            raise ValueError("The quasi-static velocity solve requires eta_r > 0.")
        if self.n_points < 51 or self.n_points % 2 == 0:
            raise ValueError("n_points must be an odd integer of at least 51.")
        if not (0.0 < self.cfl <= 1.0):
            raise ValueError("cfl must lie in (0, 1].")
        if self.artificial_diffusion < 0.0 or self.stokes_strength < 0.0:
            raise ValueError("Diffusion and Stokes strength must be non-negative.")
        if self.initial_radius <= 0.0:
            raise ValueError("initial_radius must be positive.")
        if not (0.0 <= self.pre_stretch < 4.0):
            raise ValueError("pre_stretch must keep 1-alpha*u*(1-u) positive.")
        if self.solvent_factor <= 0.0:
            raise ValueError("solvent_factor must be positive.")
        if self.area_scheme not in {"pointwise", "conservative"}:
            raise ValueError(f"Unsupported area scheme: {self.area_scheme}")


@dataclass(frozen=True)
class EulerianSnapshot:
    time: float
    z: FloatArray
    radius: FloatArray
    transformed_area: FloatArray
    physical_area: FloatArray
    conformation: FloatArray

    @property
    def neck_cells(self) -> int:
        """Number of grid points with ``R < 1.5 min(R)``: a resolution check.

        Fewer than about ten means the neck shape is not resolved and pointwise
        mid-filament quantities should be read with care."""

        return int(np.count_nonzero(self.radius < 1.5 * float(np.min(self.radius))))


@dataclass(frozen=True)
class EulerianResult:
    config: EulerianConfig
    time: FloatArray
    true_strain: FloatArray
    nominal_strain: FloatArray
    mid_stress: FloatArray
    engineering_tension: FloatArray
    nominal_rate: FloatArray
    mid_rate: FloatArray
    volume: FloatArray
    u: FloatArray
    initial_area: FloatArray
    final_area: FloatArray
    final_conformation: FloatArray
    snapshots: dict[float, EulerianSnapshot] = field(default_factory=dict)

    @property
    def tracking_error(self) -> FloatArray:
        return self.true_strain - self.config.target_rate * self.time


def stokes_viscosity(
    u: FloatArray,
    *,
    nominal_strain: float,
    base_viscosity: float,
    initial_radius: float,
    strength: float,
) -> FloatArray:
    """Stokes-type end correction used to approximate no slip at the plates.

    This is the functional form used in the authors' MATLAB helper.  The singular
    endpoints are evaluated at a tiny positive distance; only the inverse
    viscosity enters the force-balance integrals, so the limit is regular.
    """

    if base_viscosity <= 0.0:
        raise ValueError("base_viscosity must be positive.")
    du = float(u[1] - u[0])
    safe = np.clip(u, 0.25 * du, 1.0 - 0.25 * du)
    contracted_radius = initial_radius * np.exp(-nominal_strain)
    correction = (strength / 32.0) * (
        (contracted_radius / safe) ** 2
        + (contracted_radius / (1.0 - safe)) ** 2
    )
    return base_viscosity * (1.0 + correction)


def _upwind_derivative(q: FloatArray, velocity: FloatArray, du: float) -> FloatArray:
    derivative = np.zeros_like(q)
    # Authors' notes: forward difference when c<0 (left half), backward when c>0.
    neg = np.flatnonzero(velocity[:-1] < 0.0)
    derivative[neg] = (q[neg + 1] - q[neg]) / du
    pos = np.flatnonzero(velocity[1:] > 0.0) + 1
    derivative[pos] = (q[pos] - q[pos - 1]) / du
    return derivative


def _conservative_area_rhs(
    area: FloatArray, convective_velocity: FloatArray, du: float
) -> FloatArray:
    """Finite-volume upwind form of Eq. (A18).

    Equation (A18) is ``da/dt + d(c a)/du = 0`` with ``c = v - epsdot_N u``
    and ``c(0) = c(1) = 0``.  Nodes are cell centres (half cells at the
    plates), face velocities are averaged, and the face value of ``a`` is
    taken from the upwind side.  The discrete filament volume, the
    trapezoidal sum of ``a``, is then conserved to round-off, whereas the
    non-conservative upwind form drifts by O(du).
    """

    c_face = 0.5 * (convective_velocity[:-1] + convective_velocity[1:])
    upwind = np.where(c_face > 0.0, area[:-1], area[1:])
    flux = c_face * upwind
    rhs = np.zeros_like(area)
    rhs[1:-1] = -(flux[1:] - flux[:-1]) / du
    rhs[0] = -flux[0] / (0.5 * du)
    rhs[-1] = flux[-1] / (0.5 * du)
    return rhs


def _neumann_laplacian(q: FloatArray, du: float) -> FloatArray:
    lap = np.empty_like(q)
    lap[1:-1] = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / du**2
    lap[0] = 2.0 * (q[1] - q[0]) / du**2
    lap[-1] = 2.0 * (q[-2] - q[-1]) / du**2
    return lap


def _make_snapshot(
    *,
    requested_time: float,
    u: FloatArray,
    nominal_strain: float,
    transformed_area: FloatArray,
    conformation: FloatArray,
) -> EulerianSnapshot:
    physical_area = transformed_area * np.exp(-nominal_strain)
    z = u * np.exp(nominal_strain)
    radius = np.sqrt(physical_area / np.pi)
    return EulerianSnapshot(
        time=float(requested_time),
        z=z.copy(),
        radius=radius.copy(),
        transformed_area=transformed_area.copy(),
        physical_area=physical_area.copy(),
        conformation=conformation.copy(),
    )


def simulate_eulerian(config: EulerianConfig) -> EulerianResult:
    """Run a controlled slender-filament simulation."""

    config.validate()
    u = np.linspace(0.0, 1.0, config.n_points)
    du = float(u[1] - u[0])
    mid = config.n_points // 2

    if config.model == "newtonian":
        G0 = 0.0
        eta_s = 1.0  # the magnitude cancels out of the kinematics
        beta = 0.0
    else:
        G0 = config.G0
        eta_s = solvent_viscosity_from_fraction(
            config.eta_r, G0=G0, polymer_zero_shear_time=config.tau
        )
        beta = 0.0 if config.model == "oldroyd_b_toy" else config.beta

    initial_area = (
        np.pi
        * config.initial_radius**2
        * (1.0 - config.pre_stretch * u * (1.0 - u))
    )
    area = initial_area.copy()
    conformation = np.zeros_like(u)

    dt_guess = config.cfl * du / config.target_rate
    n_steps = int(np.ceil(config.end_time / dt_guess))
    dt = config.end_time / n_steps
    time = np.linspace(0.0, config.end_time, n_steps + 1)

    true_strain = np.zeros(n_steps + 1)
    nominal_strain = np.zeros(n_steps + 1)
    mid_stress = np.zeros(n_steps + 1)
    tension = np.zeros(n_steps + 1)
    nominal_rate = np.zeros(n_steps + 1)
    mid_rate = np.zeros(n_steps + 1)
    volume = np.zeros(n_steps + 1)
    volume[0] = _trapezoid(area, u)

    requested = sorted(set(float(x) for x in config.snapshot_times))
    snapshots: dict[float, EulerianSnapshot] = {}
    if requested and requested[0] <= 0.5 * dt:
        snapshots[requested[0]] = _make_snapshot(
            requested_time=requested[0],
            u=u,
            nominal_strain=0.0,
            transformed_area=area,
            conformation=conformation,
        )
        next_snapshot = 1
    else:
        next_snapshot = 0

    eps_n = 0.0
    for step in range(n_steps):
        eta = stokes_viscosity(
            u,
            nominal_strain=eps_n,
            base_viscosity=eta_s,
            initial_radius=config.initial_radius,
            strength=config.stokes_strength,
        )

        # Constant engineering tension and boundary velocity determine the
        # nominal rate.  Enforcing v_u(mid)=target_rate gives T directly.
        integral_aeta = _trapezoid(1.0 / (area * eta), u)
        integral_zeta = (
            _trapezoid(conformation / eta, u) if G0 != 0.0 else 0.0
        )

        def kinematics(tension: float):
            """Velocity field, nominal rate and area update for a tension."""

            epsdot = tension * integral_aeta - G0 * integral_zeta
            grad = (tension / area - G0 * conformation) / eta
            vel = np.concatenate(
                (np.array([0.0]), cumulative_trapezoid(grad, u))
            )
            conv = vel - epsdot * u
            conv[0] = 0.0
            conv[-1] = 0.0
            if config.area_scheme == "conservative":
                rhs = _conservative_area_rhs(area, conv, du)
            else:
                rhs = (
                    -conv * _upwind_derivative(area, conv, du)
                    - (grad - epsdot) * area
                )
            return epsdot, grad, vel, conv, rhs

        current_tension = area[mid] * (
            G0 * conformation[mid] + eta[mid] * config.target_rate
        )
        if config.area_scheme == "conservative":
            # The rate constraint v_u(mid)=target only controls the mid cell
            # exactly in the non-conservative form.  In the finite-volume
            # form the mid cell also exchanges volume with its neighbours, so
            # the controller is closed on the discrete true Hencky strain of
            # the mid cell, which is what a real instrument measures.
            target_strain = config.target_rate * time[step + 1]

            def tracking_residual(tension: float) -> float:
                epsdot, _, _, _, rhs = kinematics(tension)
                a_mid_new = area[mid] + dt * rhs[mid]
                if a_mid_new <= 0.0:
                    return -np.inf
                # physical area = a exp(-eps_N)  =>  eps = ln(A0/a) + eps_N
                strain = np.log(initial_area[mid] / a_mid_new) + (
                    eps_n + epsdot * dt
                )
                return strain - target_strain

            lower, upper = 0.5 * current_tension, 2.0 * current_tension
            f_lower, f_upper = tracking_residual(lower), tracking_residual(upper)
            expansions = 0
            while f_lower > 0.0 and expansions < 60:
                lower *= 0.5
                f_lower = tracking_residual(lower)
                expansions += 1
            while f_upper < 0.0 and expansions < 60:
                upper *= 2.0
                f_upper = tracking_residual(upper)
                expansions += 1
            if not (f_lower <= 0.0 <= f_upper):
                raise RuntimeError(
                    f"Could not bracket the controller tension at step {step}."
                )
            current_tension = brentq(
                tracking_residual, lower, upper, xtol=1.0e-14, rtol=1.0e-13
            )

        epsdot_n, velocity_gradient, velocity, convective_velocity, area_rhs = (
            kinematics(current_tension)
        )
        next_area = area + dt * area_rhs
        if np.any(next_area <= 0.0) or np.any(~np.isfinite(next_area)):
            raise RuntimeError(
                f"Eulerian area became non-positive at step {step}; "
                "reduce CFL or increase solvent regularization."
            )

        if G0 != 0.0:
            fz = 2.0 + 2.0 * conformation - beta * conformation**2
            z_rhs = conformation + dt * (
                -convective_velocity
                * _upwind_derivative(conformation, convective_velocity, du)
                + velocity_gradient * fz
                + config.artificial_diffusion
                * _neumann_laplacian(conformation, du)
            )
            next_conformation = z_rhs / (1.0 + dt / config.tau)
        else:
            next_conformation = conformation

        eps_n += epsdot_n * dt
        area = next_area
        conformation = next_conformation
        physical_area_mid = area[mid] * np.exp(-eps_n)
        eps_true = np.log(initial_area[mid] / physical_area_mid)

        true_strain[step + 1] = eps_true
        nominal_strain[step + 1] = eps_n
        nominal_rate[step + 1] = epsdot_n
        mid_rate[step + 1] = velocity_gradient[mid]
        tension[step + 1] = current_tension
        mid_stress[step + 1] = current_tension / area[mid]
        volume[step + 1] = _trapezoid(area, u)

        while (
            next_snapshot < len(requested)
            and time[step + 1] >= requested[next_snapshot] - 0.5 * dt
        ):
            target_time = requested[next_snapshot]
            snapshots[target_time] = _make_snapshot(
                requested_time=target_time,
                u=u,
                nominal_strain=eps_n,
                transformed_area=area,
                conformation=conformation,
            )
            next_snapshot += 1

    return EulerianResult(
        config=config,
        time=time,
        true_strain=true_strain,
        nominal_strain=nominal_strain,
        mid_stress=mid_stress,
        engineering_tension=tension,
        nominal_rate=nominal_rate,
        mid_rate=mid_rate,
        volume=volume,
        u=u,
        initial_area=initial_area,
        final_area=area.copy(),
        final_conformation=conformation.copy(),
        snapshots=snapshots,
    )
