"""Homogeneous-flow and steady-state formulas used in Figures 4, 6, 8, and 9."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import brentq
from scipy.special import hyp2f1

from .spectra import PronySpectrum

FloatArray = NDArray[np.float64]


def solvent_viscosity_from_fraction(
    eta_r: float, *, G0: float, polymer_zero_shear_time: float
) -> float:
    """Convert relative solvent fraction into a dimensional solvent viscosity.

    ``eta_r = eta_s / (G0 * polymer_zero_shear_time + eta_s)``.
    """

    if not (0.0 <= eta_r < 1.0):
        raise ValueError("eta_r must satisfy 0 <= eta_r < 1.")
    if G0 < 0.0 or polymer_zero_shear_time <= 0.0:
        raise ValueError("G0 must be non-negative and the polymer time positive.")
    if eta_r == 0.0:
        return 0.0
    return eta_r * G0 * polymer_zero_shear_time / (1.0 - eta_r)


def rolie_steady_z(wi: ArrayLike, beta: float) -> FloatArray | float:
    """Steady conformation difference for the scalar toy model.

    The homogeneous steady equation is

    ``0 = Wi * (2 + 2 Z - beta Z**2) - Z``.

    For ``beta=0`` it reduces to ``Z=2 Wi/(1-2 Wi)`` and has the
    Oldroyd-B toy singularity at ``Wi=1/2``.
    """

    w = np.asarray(wi, dtype=float)
    if np.any(w < 0.0):
        raise ValueError("Wi must be non-negative.")
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")
    if beta == 0.0:
        out = 2.0 * w / (1.0 - 2.0 * w)
    else:
        # This form is faithful to Eq. (6) and stable except at exactly Wi=0.
        disc = (1.0 - 2.0 * w) ** 2 + 8.0 * beta * w**2
        numerator = 2.0 * w - 1.0 + np.sqrt(disc)
        denominator = 2.0 * beta * w
        out = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(w),
            where=w != 0.0,
        )
    return float(out) if out.ndim == 0 else out


def rolie_high_rate_limit(beta: float) -> float:
    """Equation (6): high-rate stress/conformation saturation."""

    if beta <= 0.0:
        raise ValueError("beta must be positive for a finite high-rate limit.")
    return float((1.0 + np.sqrt(1.0 + 2.0 * beta)) / beta)


@dataclass(frozen=True)
class StartupCurve:
    time: FloatArray
    polymer_state: FloatArray
    stress: FloatArray
    steady_stress: float


def rolie_startup(
    time: ArrayLike,
    *,
    rate: float,
    tau: float = 1.0,
    beta: float = 0.4,
    G0: float = 1.0,
    eta_r: float = 0.2,
    solvent_factor: float = 1.0,
) -> StartupCurve:
    """Homogeneous startup for the scalar Rolie-Poly/Oldroyd-B toy.

    The solvent factor is set to one for the paper's transformed Eulerian
    convention.  Appendix A3 displays a factor of three, but A11/A19 and the
    Figure 4(b) stress level are consistent with a factor of one.
    """

    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) < 0.0):
        raise ValueError("time must be a sorted one-dimensional array.")
    if rate < 0.0 or tau <= 0.0 or beta < 0.0 or G0 <= 0.0:
        raise ValueError("rate/beta must be non-negative and tau/G0 positive.")

    eta_s = solvent_viscosity_from_fraction(
        eta_r, G0=G0, polymer_zero_shear_time=tau
    )

    def rhs(_: float, state: FloatArray) -> FloatArray:
        z = state[0]
        return np.array([rate * (2.0 + 2.0 * z - beta * z * z) - z / tau])

    sol = solve_ivp(
        rhs,
        (float(t[0]), float(t[-1])),
        y0=np.array([0.0]),
        t_eval=t,
        rtol=2.0e-10,
        atol=1.0e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Rolie startup integration failed: {sol.message}")
    z = sol.y[0]
    stress = G0 * z + solvent_factor * eta_s * rate
    z_ss = float(rolie_steady_z(rate * tau, beta))
    steady = G0 * z_ss + solvent_factor * eta_s * rate
    return StartupCurve(t, z, stress, float(steady))


def de_orientation_kernel(strain: ArrayLike) -> FloatArray | float:
    """Axial-minus-radial component of ``B/tr(B)`` in uniaxial extension."""

    e = np.asarray(strain, dtype=float)
    # q=(1-exp(-3e))/(1+2exp(-3e)); expm1 improves small-strain accuracy.
    em = np.exp(-3.0 * e)
    q = (1.0 - em) / (1.0 + 2.0 * em)
    return float(q) if q.ndim == 0 else q


def de_mode_steady_orientation(wi: ArrayLike) -> FloatArray | float:
    """Steady DE-toy orientation difference of a single Maxwell mode.

    This evaluates the two integrals used in the original MATLAB program:

    ``D(Wi)=integral exp(-x) [1-exp(-3 Wi x)]/[1+2 exp(-3 Wi x)] dx``.
    """

    w = np.asarray(wi, dtype=float)
    if np.any(w < 0.0):
        raise ValueError("Wi must be non-negative.")
    # With y=exp(-x), D=int_0^1 (1-y^(3Wi))/(1+2y^(3Wi)) dy.
    # The integral is -1/2 + (3/2) 2F1(1,b;1+b;-2), b=1/(3Wi).
    values = np.zeros_like(w)
    positive = w > 0.0
    b = np.divide(1.0, 3.0 * w, out=np.zeros_like(w), where=positive)
    values[positive] = -0.5 + 1.5 * hyp2f1(
        1.0, b[positive], 1.0 + b[positive], -2.0
    )
    return float(values) if values.ndim == 0 else values


def wagner_stretch_function(lam: ArrayLike, p: float = 0.3) -> FloatArray | float:
    """Wagner stretch-relaxation function from Eq. (19)."""

    x = np.asarray(lam, dtype=float)
    if np.any(x <= 0.0):
        raise ValueError("Molecular stretch must be positive.")
    if p < 0.0:
        raise ValueError("p must be non-negative.")
    value = 1.0 - 2.0 * p / 3.0 + (2.0 * p / 9.0) * (
        x**4 + x**3 + x**2
    )
    return float(value) if value.ndim == 0 else value


def de_steady_stretch(
    orientation: float, wi_r: float, *, p: float = 0.3
) -> float:
    """Solve the homogeneous steady molecular-stretch equation.

    The original MATLAB program solves

    ``lambda * orientation * Wi_R = f(lambda) * (lambda - 1)``.
    """

    if orientation < 0.0 or wi_r < 0.0:
        raise ValueError("orientation and Wi_R must be non-negative.")
    if orientation == 0.0 or wi_r == 0.0:
        return 1.0

    def residual(lam: float) -> float:
        return (
            lam * orientation * wi_r
            - float(wagner_stretch_function(lam, p=p)) * (lam - 1.0)
        )

    lower = 1.0
    upper = 2.0
    while residual(upper) > 0.0 and upper < 1.0e4:
        upper *= 2.0
    if upper >= 1.0e4 and residual(upper) > 0.0:
        raise RuntimeError("Could not bracket the steady molecular stretch.")
    return float(brentq(residual, lower, upper, xtol=1.0e-12, rtol=1.0e-12))


def de_spectrum_steady_orientation(rate: float, spectrum: PronySpectrum) -> float:
    """Steady orientation difference integrated over a Prony spectrum."""

    if rate < 0.0:
        raise ValueError("rate must be non-negative.")
    per_mode = np.asarray(de_mode_steady_orientation(rate * spectrum.taus))
    return float(np.dot(spectrum.weights, per_mode))


@dataclass(frozen=True)
class DESteadyState:
    orientation: float
    molecular_stretch: float
    polymer_stress: float
    solvent_stress: float
    total_stress: float


def de_steady_state(
    *,
    rate: float,
    spectrum: PronySpectrum,
    G0: float = 1.0,
    eta_r: float = 0.0,
    tau_s: float | None = None,
    p: float = 0.3,
) -> DESteadyState:
    """Analytical homogeneous steady state of the DE-toy model."""

    if rate < 0.0 or G0 <= 0.0:
        raise ValueError("rate must be non-negative and G0 positive.")
    orientation = de_spectrum_steady_orientation(rate, spectrum)
    if tau_s is None:
        lam = 1.0
    else:
        if tau_s <= 0.0:
            raise ValueError("tau_s must be positive.")
        lam = de_steady_stretch(orientation, rate * tau_s, p=p)
    eta_s = solvent_viscosity_from_fraction(
        eta_r, G0=G0, polymer_zero_shear_time=spectrum.zero_shear_time
    )
    polymer = 3.0 * G0 * lam**2 * orientation
    solvent = 3.0 * eta_s * rate
    return DESteadyState(
        orientation=orientation,
        molecular_stretch=lam,
        polymer_stress=float(polymer),
        solvent_stress=float(solvent),
        total_stress=float(polymer + solvent),
    )


def de_homogeneous_startup(
    time: ArrayLike,
    *,
    rate: float,
    spectrum: PronySpectrum,
    G0: float = 1.0,
    eta_r: float = 0.0,
) -> StartupCurve:
    """Controlled constant-rate startup of the non-stretching DE-toy model.

    The initial material is assumed equilibrated for all ``t'<0``.  For each
    Prony mode, the orientation is the surviving pre-history plus the integral
    over deformation created after startup.  This is the homogeneous analogue
    of the history sum in the original MATLAB program.
    """

    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("time must be a strictly increasing one-dimensional array.")
    if t[0] != 0.0:
        raise ValueError("time must begin at zero for startup calculations.")
    if rate < 0.0 or G0 <= 0.0:
        raise ValueError("rate must be non-negative and G0 positive.")

    q = np.asarray(de_orientation_kernel(rate * t))
    orientation = np.zeros_like(t)
    for weight, tau in zip(spectrum.weights, spectrum.taus, strict=True):
        integrand = (weight / tau) * np.exp(-t / tau) * q
        created = cumulative_trapezoid(integrand, t, initial=0.0)
        prehistory = weight * np.exp(-t / tau) * q
        orientation += prehistory + created

    eta_s = solvent_viscosity_from_fraction(
        eta_r, G0=G0, polymer_zero_shear_time=spectrum.zero_shear_time
    )
    stress = 3.0 * G0 * orientation + 3.0 * eta_s * rate
    # At exactly t=0 the instrument/model is conventionally shown from zero,
    # before the idealized instantaneous Newtonian jump.
    stress[0] = 0.0
    steady = de_steady_state(
        rate=rate, spectrum=spectrum, G0=G0, eta_r=eta_r, tau_s=None
    ).total_stress
    return StartupCurve(t, orientation, stress, float(steady))
