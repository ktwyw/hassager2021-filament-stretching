"""Sparse Lagrangian history-integral filament solver.

This module is a mathematical translation of the authors' original MATLAB
Lagrangian program (not included in this repository), with two important
engineering changes:

1. The dense Newton matrix is assembled as a sparse matrix.  This reduces the
   cost enough to reproduce Figures 3 and 6--10 on a laptop.
2. The user may select either the original MATLAB feedback law or an ideal
   midplane constraint.  The latter exactly imposes the requested true Hencky
   strain and is used for the paper-reproduction presets because the paper says
   the detailed controller is not its focus.
3. Two time discretizations are offered.  ``"matlab_first_order"`` is the
   literal translation of the original driver (midpoint memory ages with
   left-endpoint deformation, and a solvent force with the old segment length
   in the denominator).  That scheme overshoots the analytical steady stress by
   O(dt) (about 4% at Wi=2, dt=0.05), whereas the paper's Figs. 6(b) and 8(a)
   land on the analytical line.  ``"second_order"`` (default) uses a
   trapezoidal history quadrature consistent with the pre-history survival
   term, and the exact logarithmic segment rate in the solvent and stretch
   equations.  Its error is O(dt^2) and it converges to the same solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .analytical import solvent_viscosity_from_fraction
from .spectra import PronySpectrum

FloatArray = NDArray[np.float64]
IntegralModel = Literal["de_toy", "oldroyd_b"]
ControlMode = Literal["exact_midplane", "matlab_feedback"]
TimeDiscretization = Literal["second_order", "matlab_first_order"]


@dataclass(frozen=True)
class LagrangianConfig:
    model: IntegralModel = "de_toy"
    startup_rate: float = 2.0
    relaxation_rate: float = 0.0
    startup_time: float = 2.0
    relaxation_time: float = 0.0
    dt: float = 0.05
    n_nodes: int = 201
    G0: float = 1.0
    eta_r: float = 0.06
    include_molecular_stretch: bool = True
    stretch_relaxation_time: float = 1.0e-6
    wagner_p: float = 0.3
    stokes_coefficient: float = 0.25
    control: ControlMode = "exact_midplane"
    time_discretization: TimeDiscretization = "second_order"
    fast_mode_cutoff: float = 1.0
    matlab_initial_gain: float = 0.4
    max_newton_iterations: int = 10
    newton_tolerance: float = 1.0e-9
    strict_convergence: bool = True

    def validate(self) -> None:
        if self.model not in {"de_toy", "oldroyd_b"}:
            raise ValueError(f"Unsupported integral model: {self.model}")
        if self.startup_rate < 0.0 or self.relaxation_rate < 0.0:
            raise ValueError("Rates must be non-negative.")
        if self.startup_time < 0.0 or self.relaxation_time < 0.0:
            raise ValueError("Durations must be non-negative.")
        if self.startup_time + self.relaxation_time <= 0.0:
            raise ValueError("The total simulation time must be positive.")
        if self.dt <= 0.0 or self.G0 <= 0.0:
            raise ValueError("dt and G0 must be positive.")
        if self.n_nodes < 21 or self.n_nodes % 2 == 0:
            raise ValueError("n_nodes must be an odd integer of at least 21.")
        if not (0.0 < self.eta_r < 1.0):
            raise ValueError("The force solve requires 0 < eta_r < 1.")
        if self.model == "oldroyd_b" and self.include_molecular_stretch:
            raise ValueError("Molecular stretch is defined only for the DE-toy model.")
        if self.include_molecular_stretch and self.stretch_relaxation_time <= 0.0:
            raise ValueError("stretch_relaxation_time must be positive.")
        if self.wagner_p < 0.0 or self.stokes_coefficient < 0.0:
            raise ValueError("Wagner p and Stokes coefficient must be non-negative.")
        if self.control not in {"exact_midplane", "matlab_feedback"}:
            raise ValueError(f"Unsupported control mode: {self.control}")
        if self.fast_mode_cutoff < 0.0:
            raise ValueError("fast_mode_cutoff must be non-negative.")
        if self.time_discretization not in {"second_order", "matlab_first_order"}:
            raise ValueError(
                f"Unsupported time discretization: {self.time_discretization}"
            )
        if self.max_newton_iterations < 2 or self.newton_tolerance <= 0.0:
            raise ValueError("Invalid Newton settings.")


@dataclass(frozen=True)
class LagrangianSnapshot:
    time: float
    segment_midpoints: FloatArray
    radius: FloatArray
    molecular_stretch: FloatArray
    local_hencky_strain: FloatArray

    @property
    def neck_segments(self) -> int:
        """Number of material segments with ``R < 1.5 min(R)``: a resolution
        check.  Fewer than about ten means the neck shape is not resolved."""

        return int(np.count_nonzero(self.radius < 1.5 * float(np.min(self.radius))))


@dataclass(frozen=True)
class LagrangianResult:
    config: LagrangianConfig
    spectrum: PronySpectrum
    time: FloatArray
    target_true_strain: FloatArray
    true_strain: FloatArray
    nominal_strain: FloatArray
    true_stress: FloatArray
    engineering_force: FloatArray
    positions: FloatArray
    molecular_stretch: FloatArray
    local_stretch_ratio: FloatArray
    newton_errors: FloatArray
    newton_iterations: NDArray[np.int64]
    lumped_fast_mode_viscosity: float = 0.0

    @property
    def segment_midpoints(self) -> FloatArray:
        return 0.5 * (self.positions[:-1, :] + self.positions[1:, :])

    @property
    def radius(self) -> FloatArray:
        return np.sqrt(1.0 / self.local_stretch_ratio)

    @property
    def tracking_error(self) -> FloatArray:
        return self.true_strain - self.target_true_strain

    def snapshot(self, requested_time: float) -> LagrangianSnapshot:
        idx = int(np.argmin(np.abs(self.time - requested_time)))
        return LagrangianSnapshot(
            time=float(self.time[idx]),
            segment_midpoints=self.segment_midpoints[:, idx].copy(),
            radius=self.radius[:, idx].copy(),
            molecular_stretch=self.molecular_stretch[:, idx].copy(),
            local_hencky_strain=np.log(self.local_stretch_ratio[:, idx]),
        )


def _orientation_force_and_derivative(
    current_length: FloatArray,
    history_length: FloatArray,
    initial_length: FloatArray,
    model: IntegralModel,
) -> tuple[FloatArray, FloatArray]:
    """Return orientation contribution to engineering force and its d/dlength.

    ``history_length`` may be one-dimensional (pre-history term) or a matrix
    whose columns are earlier time levels.  The formulas are the vectorized
    form of the segment-wise expressions in the original MATLAB driver.
    """

    d = current_length
    d0 = initial_length
    h = history_length
    matrix_history = h.ndim == 2

    if matrix_history:
        d_b = d[:, None]
        d0_b = d0[:, None]
        trace_b = (d_b / h) ** 2 + 2.0 * h / d_b
        numerator = d_b * d0_b / h**2 - h * d0_b / d_b**2
        d_numerator = d0_b / h**2 + 2.0 * d0_b * h / d_b**3
    else:
        trace_b = (d / h) ** 2 + 2.0 * h / d
        numerator = d * d0 / h**2 - h * d0 / d**2
        d_numerator = d0 / h**2 + 2.0 * d0 * h / d**3

    if model == "de_toy":
        if matrix_history:
            d_trace = 2.0 * d[:, None] / h**2 - 2.0 * h / d[:, None] ** 2
        else:
            d_trace = 2.0 * d / h**2 - 2.0 * h / d**2
        value = numerator / trace_b
        derivative = d_numerator / trace_b - numerator * d_trace / trace_b**2
    elif model == "oldroyd_b":
        value = numerator
        derivative = d_numerator
    else:  # defensive; config validation catches this
        raise ValueError(model)
    return value, derivative


def wagner_function(lam: FloatArray, p: float) -> FloatArray:
    """Wagner stretch-relaxation function, Eq. (19)."""

    return 1.0 - 2.0 * p / 3.0 + (2.0 * p / 9.0) * (lam**4 + lam**3 + lam**2)


def wagner_derivative(lam: FloatArray, p: float) -> FloatArray:
    """Derivative of :func:`wagner_function` with respect to ``lam``."""

    return (2.0 * p / 9.0) * (4.0 * lam**3 + 3.0 * lam**2 + 2.0 * lam)


def _stokes_segment_viscosity(
    old_positions: FloatArray,
    *,
    eta_s: float,
    coefficient: float,
) -> FloatArray:
    """Position-dependent Stokes end correction of the original MATLAB driver."""

    length = float(old_positions[-1])
    mid = 0.5 * (old_positions[:-1] + old_positions[1:])
    left = np.maximum(mid, 1.0e-14)
    right = np.maximum(length - mid, 1.0e-14)
    return eta_s * (
        1.0
        + coefficient
        * (1.0 / (32.0 * left**2) + 1.0 / (32.0 * right**2))
        / max(length, 1.0e-14) ** 2
    )


def simulate_lagrangian(
    config: LagrangianConfig,
    spectrum: PronySpectrum,
) -> LagrangianResult:
    """Run the sparse Newton/history simulation."""

    config.validate()
    n = config.n_nodes
    m = n - 1
    mid_segment = m // 2

    n_startup = int(round(config.startup_time / config.dt))
    n_relax = int(round(config.relaxation_time / config.dt))
    if not np.isclose(n_startup * config.dt, config.startup_time):
        raise ValueError("startup_time must be an integer multiple of dt.")
    if not np.isclose(n_relax * config.dt, config.relaxation_time):
        raise ValueError("relaxation_time must be an integer multiple of dt.")
    n_steps = n_startup + n_relax
    time = np.arange(n_steps + 1, dtype=float) * config.dt
    imposed_rates = np.concatenate(
        (
            np.full(n_startup, config.startup_rate),
            np.full(n_relax, config.relaxation_rate),
        )
    )
    target_true_strain = np.concatenate(
        (np.array([0.0]), np.cumsum(imposed_rates) * config.dt)
    )

    eta_s = solvent_viscosity_from_fraction(
        config.eta_r,
        G0=config.G0,
        polymer_zero_shear_time=spectrum.zero_shear_time,
    )

    # Relaxation modes much faster than the time step cannot be resolved by
    # any quadrature of the history integral.  In the limit tau_k -> 0 a
    # Maxwell mode of the DE-toy or Oldroyd-B model is exactly a Newtonian
    # contribution 3*G0*w_k*tau_k*epsdot, so in the second-order scheme such
    # modes are lumped into the solvent viscosity and removed from the
    # quadrature.  This matters for the broad BSW spectrum (Figure 8), whose
    # sub-step modes carry a few per cent of the steady stress.  The MATLAB
    # scheme keeps every mode in the quadrature for source fidelity.
    quadrature_spectrum = spectrum
    lumped_viscosity = 0.0
    if config.time_discretization == "second_order" and config.fast_mode_cutoff > 0.0:
        fast = spectrum.taus < config.fast_mode_cutoff * config.dt
        if np.any(fast) and not np.all(fast):
            lumped_viscosity = float(
                config.G0 * np.dot(spectrum.weights[fast], spectrum.taus[fast])
            )
            quadrature_spectrum = PronySpectrum(
                weights=spectrum.weights[~fast],
                taus=spectrum.taus[~fast],
                name=spectrum.name + " (sub-step modes lumped)",
                metadata=spectrum.metadata,
            )
    eta_s_effective = eta_s + lumped_viscosity

    positions = np.zeros((n, n_steps + 1))
    positions[:, 0] = np.linspace(0.0, 1.0, n)
    segment_history = np.zeros((m, n_steps + 1))
    segment_history[:, 0] = np.diff(positions[:, 0])
    initial_segment = segment_history[:, 0].copy()
    molecular_stretch = np.ones((m, n_steps + 1))
    engineering_force = np.zeros(n_steps + 1)
    true_stress = np.zeros(n_steps + 1)
    true_strain = np.zeros(n_steps + 1)
    nominal_strain = np.zeros(n_steps + 1)
    max_newton = config.max_newton_iterations
    newton_errors = np.full((n_steps, max_newton), np.nan)
    newton_iterations = np.zeros(n_steps, dtype=np.int64)

    # Initial target of the original feedback controller.
    plate_position = np.exp(
        config.matlab_initial_gain * config.startup_rate * config.dt
    )

    second_order = config.time_discretization == "second_order"
    # Right-hand side of the stretch equation at the previous time level
    # (Crank-Nicolson).  The material is at rest for t<0, so it starts at 0.
    stretch_rhs_old = np.zeros(m)
    for step in range(n_steps):
        old_positions = positions[:, step]
        old_segment = segment_history[:, step]

        # Newton predictor: extrapolate each segment length geometrically
        # from the previous step (exact for a segment stretched at constant
        # rate).  The MATLAB driver starts from the old positions, which is
        # adequate for its linear solvent term but leaves a large first
        # correction at high strain; the predictor keeps Newton in its
        # quadratic regime for both discretizations.
        if step == 0:
            ratio = np.full(m, np.exp(imposed_rates[0] * config.dt))
        else:
            ratio = old_segment / segment_history[:, step - 1]
            ratio = np.clip(ratio, 0.5, 2.0)
        predicted_positions = np.concatenate(
            (np.array([0.0]), np.cumsum(old_segment * ratio))
        )

        # Stokes end-correction viscosity.  The MATLAB driver evaluates it at
        # the old positions (explicit, O(dt)); the second-order scheme uses
        # the extrapolated positions, which are O(dt^2) accurate, so the
        # Jacobian is unchanged in both cases.
        eta_star = _stokes_segment_viscosity(
            predicted_positions if second_order else old_positions,
            eta_s=eta_s_effective,
            coefficient=config.stokes_coefficient,
        )

        use_stretch = config.include_molecular_stretch and config.model == "de_toy"
        if use_stretch:
            n_unknown = 2 * n
            force_index = 2 * n - 1
            x = np.empty(n_unknown)
            x[:n] = predicted_positions
            x[n : n + m] = molecular_stretch[:, step]
            x[force_index] = engineering_force[step]
        else:
            n_unknown = n + 1
            force_index = n
            x = np.empty(n_unknown)
            x[:n] = predicted_positions
            x[force_index] = engineering_force[step]

        converged = False
        for iteration in range(max_newton):
            current_positions = x[:n]
            current_segment = np.diff(current_positions)
            if np.any(current_segment <= 0.0):
                raise RuntimeError(
                    f"A segment became non-positive at step {step}, Newton iteration {iteration}."
                )

            # Integral over t'<0 (equilibrium pre-history).
            pre_value, pre_derivative = _orientation_force_and_derivative(
                current_segment,
                initial_segment,
                initial_segment,
                config.model,
            )
            if second_order:
                # The new time level is t=(step+1)*dt; the pre-history term
                # is the exact survival G(t)/G0 of the equilibrium state.
                survival = float(quadrature_spectrum.survival((step + 1) * config.dt))
            else:
                # The +0.5*dt offset is deliberately retained from the
                # original driver's 'test version with extra 0.5*dt'.
                survival = float(quadrature_spectrum.survival((step + 1.5) * config.dt))
            orientation_force = survival * pre_value
            orientation_derivative = survival * pre_derivative

            # Memory quadrature over 0 <= t' <= t.  The stored history levels
            # k=0..step give the deformation from t'=k*dt to the new time.
            previous_segments = segment_history[:, : step + 1]
            hist_value, hist_derivative = _orientation_force_and_derivative(
                current_segment,
                previous_segments,
                initial_segment,
                config.model,
            )
            levels = np.arange(step + 1, dtype=float)
            if second_order:
                # Trapezoidal rule on nodes t'=k*dt, k=0..step+1.  The node
                # k=step+1 (t'=t) contributes nothing because Q(t,t)=0, so
                # only the k=0 end weight is halved.  This is globally
                # second-order accurate and consistent with the survival
                # term evaluated at age t.
                ages = config.dt * (step + 1.0 - levels)
                memory_weights = np.asarray(quadrature_spectrum.memory(ages)) * config.dt
                memory_weights[0] *= 0.5
            else:
                # MATLAB midpoint ages with left-endpoint deformation: the
                # memory is sampled half a step later than the deformation it
                # multiplies, which gives a first-order stress overshoot.
                ages = config.dt * (step - levels + 0.5)
                memory_weights = np.asarray(quadrature_spectrum.memory(ages)) * config.dt
            orientation_force += hist_value @ memory_weights
            orientation_derivative += hist_derivative @ memory_weights

            if config.model == "de_toy":
                true_orientation = (
                    orientation_force * current_segment / initial_segment
                )
                d_true_orientation = (
                    orientation_derivative * current_segment / initial_segment
                    + orientation_force / initial_segment
                )
            else:
                true_orientation = np.zeros(m)
                d_true_orientation = np.zeros(m)

            if second_order:
                # Engineering solvent force F/A0 = 3*eta*epsdot*d0/d, with
                # the Hencky rate of the segment at the new time level from a
                # second-order backward difference of ln(d) (BDF2; backward
                # Euler on the first step, where the material starts from
                # rest).  This keeps the solvent stress at the same time level
                # as the quasi-static force balance and the polymer stress.
                log_new = np.log(current_segment)
                if step == 0:
                    local_rate_s = (log_new - np.log(old_segment)) / config.dt
                    d_rate_s = 1.0 / (config.dt * current_segment)
                else:
                    older_segment = segment_history[:, step - 1]
                    local_rate_s = (
                        1.5 * log_new
                        - 2.0 * np.log(old_segment)
                        + 0.5 * np.log(older_segment)
                    ) / config.dt
                    d_rate_s = 1.5 / (config.dt * current_segment)
                viscous_force = (
                    3.0 * eta_star * initial_segment * local_rate_s / current_segment
                )
                d_viscous_force = (
                    3.0
                    * eta_star
                    * initial_segment
                    * (d_rate_s / current_segment - local_rate_s / current_segment**2)
                )
            else:
                # MATLAB solvent residual (new length in the numerator but the
                # previous segment length squared in the denominator).
                viscous_force = (
                    3.0
                    * eta_star
                    * (initial_segment / config.dt)
                    * (current_segment - old_segment)
                    / old_segment**2
                )
                d_viscous_force = (
                    3.0
                    * eta_star
                    * (initial_segment / config.dt)
                    / old_segment**2
                )

            if config.model == "de_toy":
                stretch_now = x[n : n + m] if use_stretch else np.ones(m)
                polymer_force = (
                    3.0
                    * config.G0
                    * orientation_force
                    * stretch_now**2
                )
                d_polymer_force = (
                    3.0
                    * config.G0
                    * orientation_derivative
                    * stretch_now**2
                )
                d_polymer_d_stretch = (
                    6.0
                    * config.G0
                    * orientation_force
                    * stretch_now
                )
            else:
                polymer_force = config.G0 * orientation_force
                d_polymer_force = config.G0 * orientation_derivative
                d_polymer_d_stretch = np.zeros(m)

            residual = np.zeros(n_unknown)
            rows: list[int] = []
            cols: list[int] = []
            data: list[float] = []

            # Fixed lower plate.
            residual[0] = current_positions[0]
            rows.append(0)
            cols.append(0)
            data.append(1.0)

            # Equal force in each segment.
            force_rows = np.arange(1, m + 1)
            residual[force_rows] = (
                polymer_force + viscous_force - x[force_index]
            )
            total_length_derivative = d_polymer_force + d_viscous_force
            rows.extend(force_rows.tolist())
            cols.extend(np.arange(m).tolist())
            data.extend((-total_length_derivative).tolist())
            rows.extend(force_rows.tolist())
            cols.extend(np.arange(1, n).tolist())
            data.extend(total_length_derivative.tolist())
            rows.extend(force_rows.tolist())
            cols.extend([force_index] * m)
            data.extend([-1.0] * m)
            if use_stretch:
                rows.extend(force_rows.tolist())
                cols.extend((n + np.arange(m)).tolist())
                data.extend(d_polymer_d_stretch.tolist())

            # One global kinematic equation closes the quasi-static system.
            control_row = n
            if config.control == "matlab_feedback":
                residual[control_row] = current_positions[-1] - plate_position
                rows.append(control_row)
                cols.append(n - 1)
                data.append(1.0)
            else:
                target_mid_length = initial_segment[mid_segment] * np.exp(
                    target_true_strain[step + 1]
                )
                residual[control_row] = (
                    current_segment[mid_segment] - target_mid_length
                )
                rows.extend([control_row, control_row])
                cols.extend([mid_segment, mid_segment + 1])
                data.extend([-1.0, 1.0])

            # Molecular stretch equation (Eq. 15 with the Wagner function).
            if use_stretch:
                old_stretch = molecular_stretch[:, step]
                stretch_now = x[n : n + m]
                tau_s = config.stretch_relaxation_time
                stretch_rows = n + 1 + np.arange(m)
                if second_order:
                    local_rate = local_rate_s
                    d_local_rate = d_rate_s
                    # theta-method for d(lambda)/dt = lambda*epsdot*S
                    # - f(lambda)(lambda-1)/tau_s: Crank-Nicolson (theta=1/2)
                    # when the stretch dynamics are resolved, backward Euler
                    # (theta=1) when dt > tau_s, where lambda is slaved to 1
                    # + O(tau_s) and the damped scheme is preferable.
                    theta = 0.5 if config.dt <= tau_s else 1.0
                    f_new = wagner_function(stretch_now, config.wagner_p)
                    df_new = wagner_derivative(stretch_now, config.wagner_p)
                    rhs_new = (
                        stretch_now * local_rate * true_orientation
                        - f_new * (stretch_now - 1.0) / tau_s
                    )
                    residual[stretch_rows] = (
                        stretch_now
                        - old_stretch
                        - config.dt * (theta * rhs_new + (1.0 - theta) * stretch_rhs_old)
                    )
                    d_residual_d_stretch = 1.0 - config.dt * theta * (
                        local_rate * true_orientation
                        - (df_new * (stretch_now - 1.0) + f_new) / tau_s
                    )
                    d_residual_d_segment = -config.dt * theta * stretch_now * (
                        true_orientation * d_local_rate
                        + local_rate * d_true_orientation
                    )
                else:
                    polynomial = (
                        old_stretch**4 + old_stretch**3 + old_stretch**2
                    )
                    f_wagner = (
                        1.0
                        - 2.0 * config.wagner_p / 3.0
                        + 2.0 * config.wagner_p * polynomial / 9.0
                    )
                    local_rate = (
                        current_segment - old_segment
                    ) / (config.dt * old_segment)
                    d_local_rate = 1.0 / (config.dt * old_segment)
                    residual[stretch_rows] = (
                        (1.0 + config.dt * f_wagner / tau_s) * stretch_now
                        - old_stretch * (1.0 + local_rate * true_orientation * config.dt)
                        - config.dt * f_wagner / tau_s
                    )
                    d_residual_d_stretch = 1.0 + config.dt * f_wagner / tau_s
                    d_residual_d_segment = -old_stretch * config.dt * (
                        true_orientation * d_local_rate
                        + local_rate * d_true_orientation
                    )
                rows.extend(stretch_rows.tolist())
                cols.extend(np.arange(m).tolist())
                data.extend((-d_residual_d_segment).tolist())
                rows.extend(stretch_rows.tolist())
                cols.extend(np.arange(1, n).tolist())
                data.extend(d_residual_d_segment.tolist())
                rows.extend(stretch_rows.tolist())
                cols.extend((n + np.arange(m)).tolist())
                data.extend(np.broadcast_to(d_residual_d_stretch, (m,)).tolist())

            jacobian = coo_matrix(
                (data, (rows, cols)), shape=(n_unknown, n_unknown)
            ).tocsr()
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                try:
                    increment = spsolve(jacobian, -residual)
                except MatrixRankWarning as exc:
                    raise RuntimeError(
                        f"Newton Jacobian became singular at step {step}."
                    ) from exc
            if np.any(~np.isfinite(increment)):
                raise RuntimeError(f"Non-finite Newton increment at step {step}.")

            raw_error = float(np.linalg.norm(increment))
            newton_errors[step, iteration] = raw_error
            damping = 1.0
            accepted = False
            while damping >= 1.0e-5:
                trial = x + damping * increment
                positions_ok = np.all(np.diff(trial[:n]) > 0.0)
                stretch_ok = (
                    not use_stretch or np.all(trial[n : n + m] > 0.0)
                )
                if positions_ok and stretch_ok:
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                raise RuntimeError(
                    f"Newton damping could not preserve positive segments at step {step}."
                )
            x = trial
            if raw_error * damping < config.newton_tolerance:
                converged = True
                newton_iterations[step] = iteration + 1
                break

        if not converged:
            newton_iterations[step] = max_newton
            message = (
                f"Newton iteration did not reach {config.newton_tolerance:g} "
                f"at step {step}; last correction was "
                f"{newton_errors[step, max_newton - 1]:.3e}."
            )
            if config.strict_convergence:
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        positions[:, step + 1] = x[:n]
        segment_history[:, step + 1] = np.diff(x[:n])
        if use_stretch:
            molecular_stretch[:, step + 1] = x[n : n + m]
            if second_order:
                lam = x[n : n + m]
                stretch_rhs_old = (
                    lam * local_rate * true_orientation
                    - wagner_function(lam, config.wagner_p)
                    * (lam - 1.0)
                    / config.stretch_relaxation_time
                )
        else:
            molecular_stretch[:, step + 1] = 1.0
        engineering_force[step + 1] = x[force_index]

        local_lambda = segment_history[:, step + 1] / initial_segment
        true_strain[step + 1] = np.log(local_lambda[mid_segment])
        nominal_strain[step + 1] = np.log(positions[-1, step + 1])
        true_stress[step + 1] = (
            engineering_force[step + 1] * local_lambda[mid_segment]
        )

        if config.control == "matlab_feedback":
            error = target_true_strain[step + 1] - true_strain[step + 1]
            correction = error / (
                1.0
                + abs(true_strain[step + 1] - nominal_strain[step + 1])
            )
            plate_position = np.exp(nominal_strain[step + 1] + correction)

    local_stretch_ratio = segment_history / initial_segment[:, None]
    return LagrangianResult(
        config=config,
        spectrum=spectrum,
        time=time,
        target_true_strain=target_true_strain,
        true_strain=true_strain,
        nominal_strain=nominal_strain,
        true_stress=true_stress,
        engineering_force=engineering_force,
        positions=positions,
        molecular_stretch=molecular_stretch,
        local_stretch_ratio=local_stretch_ratio,
        newton_errors=newton_errors,
        newton_iterations=newton_iterations,
        lumped_fast_mode_viscosity=lumped_viscosity,
    )
