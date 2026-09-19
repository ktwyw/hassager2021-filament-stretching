"""Relaxation spectra and memory functions used by the reproduction code.

The paper formulates integral constitutive models through the memory function

    M(t) = -dG(t)/dt.

A Prony representation writes the relaxation modulus as

    G(t) / G0 = sum_k w_k exp(-t/tau_k),

which gives

    M(t) / G0 = sum_k (w_k/tau_k) exp(-t/tau_k).

The single-mode Maxwell memory is exact.  The BSW spectrum is represented by
positive logarithmic quadrature modes so that the same Lagrangian solver can
handle both cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PronySpectrum:
    """Positive Prony-series representation normalized by ``G0``.

    Parameters
    ----------
    weights:
        Dimensionless relaxation-modulus weights ``w_k``.
    taus:
        Positive relaxation times ``tau_k`` in the same time unit used by the
        simulation.
    name:
        Human-readable description used in plot metadata and diagnostics.
    metadata:
        Optional reconstruction notes and original parameter values.
    """

    weights: FloatArray
    taus: FloatArray
    name: str = "Prony spectrum"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float).reshape(-1)
        taus = np.asarray(self.taus, dtype=float).reshape(-1)
        if weights.size == 0:
            raise ValueError("At least one relaxation mode is required.")
        if weights.shape != taus.shape:
            raise ValueError("weights and taus must have identical one-dimensional shapes.")
        if np.any(~np.isfinite(weights)) or np.any(~np.isfinite(taus)):
            raise ValueError("weights and taus must be finite.")
        if np.any(weights < 0.0):
            raise ValueError("Prony weights must be non-negative.")
        if np.any(taus <= 0.0):
            raise ValueError("Relaxation times must be strictly positive.")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "taus", taus)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_modes(self) -> int:
        return int(self.weights.size)

    @property
    def zero_shear_time(self) -> float:
        """Polymeric zero-shear viscosity divided by ``G0``.

        For a single Maxwell mode this is exactly the relaxation time.  For a
        BSW quadrature it is the numerical approximation to
        ``integral H(tau) d tau / G0``.
        """

        return float(np.dot(self.weights, self.taus))

    @property
    def finite_instantaneous_modulus(self) -> float:
        """Return ``G(0+)/G0`` for the finite quadrature representation.

        The ideal BSW glassy branch has a short-time divergence.  A numerical
        lower cut-off makes the returned value finite; it should not be
        interpreted as a physical glassy modulus unless that cut-off is known.
        """

        return float(np.sum(self.weights))

    def memory(self, age: ArrayLike) -> FloatArray | float:
        """Evaluate ``M(age)/G0``.

        ``age`` may be a scalar or an arbitrary NumPy-compatible array.  Ages
        must be non-negative.
        """

        x = np.asarray(age, dtype=float)
        if np.any(x < 0.0):
            raise ValueError("Memory ages must be non-negative.")
        values = np.sum(
            (self.weights / self.taus) * np.exp(-x[..., None] / self.taus),
            axis=-1,
        )
        return float(values) if values.ndim == 0 else values

    def survival(self, age: ArrayLike) -> FloatArray | float:
        """Evaluate the pre-history survival integral ``G(age)/G0``."""

        x = np.asarray(age, dtype=float)
        if np.any(x < 0.0):
            raise ValueError("Memory ages must be non-negative.")
        values = np.sum(
            self.weights * np.exp(-x[..., None] / self.taus), axis=-1
        )
        return float(values) if values.ndim == 0 else values

    @classmethod
    def single_mode(cls, tau: float = 1.0) -> "PronySpectrum":
        """Exact single-mode Maxwell spectrum with modulus ``G0``."""

        if tau <= 0.0:
            raise ValueError("tau must be positive.")
        return cls(
            weights=np.array([1.0]),
            taus=np.array([float(tau)]),
            name=f"single-mode Maxwell (tau={tau:g})",
            metadata={"tau": float(tau), "source_status": "exact from Eq. (7)"},
        )

    @classmethod
    def bsw(
        cls,
        *,
        tau_d: float = 1.0,
        n_e: float = 0.23,
        n_g: float = 0.70,
        tau_c_over_tau_m: float = 1.0e-3,
        n_entanglement_modes: int = 160,
        n_glassy_modes: int = 160,
        glassy_lower_ratio: float = 1.0e-12,
        include_glassy: bool = True,
    ) -> "PronySpectrum":
        """Construct a positive quadrature of the continuous BSW spectrum.

        The paper defines

        ``H_e(tau)=n_e G0 (tau/tau_m)**n_e`` for ``tau <= tau_m`` and
        ``H_g(tau)=n_e G0 (tau/tau_c)**(-n_g)`` for ``tau <= tau_c``.

        It scales time with ``tau_d=tau_m*n_e/(1+n_e)``.  The exponent values
        0.23 and 0.70 are those stated for nearly monodisperse polystyrene in
        the cited Huang et al. work.  Neither the paper nor the original MATLAB program
        gives ``tau_c/tau_m``.  The default 1e-3 is therefore an
        explicit reconstruction parameter; it gives the stress level shown in
        Figure 8 while leaving the long-time dynamics dominated by ``H_e``.

        The entanglement branch is discretized by equal probability in
        ``y=(tau/tau_m)**n_e``.  This makes its weights sum to one exactly and
        reproduces ``tau_d`` accurately.  The glassy branch uses midpoint
        logarithmic quadrature and requires a numerical lower cut-off because
        its ideal instantaneous modulus diverges as ``tau -> 0``.
        """

        if tau_d <= 0.0:
            raise ValueError("tau_d must be positive.")
        if not (0.0 < n_e < 1.0):
            raise ValueError("n_e must lie between 0 and 1.")
        if not (0.0 < n_g < 1.0):
            raise ValueError("n_g must lie between 0 and 1.")
        if tau_c_over_tau_m <= 0.0:
            raise ValueError("tau_c_over_tau_m must be positive.")
        if n_entanglement_modes < 4:
            raise ValueError("Use at least four entanglement modes.")
        if include_glassy and n_glassy_modes < 4:
            raise ValueError("Use at least four glassy modes.")
        if not (0.0 < glassy_lower_ratio < 1.0):
            raise ValueError("glassy_lower_ratio must lie strictly between 0 and 1.")

        tau_m = tau_d * (1.0 + n_e) / n_e

        # Exact equal-mass quadrature for H_e dln(tau).  Since
        # H_e/G0 dln(tau)=n_e x^(n_e-1) dx and y=x^n_e is uniform,
        # each mode has weight 1/N.
        y = (np.arange(n_entanglement_modes, dtype=float) + 0.5) / n_entanglement_modes
        x = y ** (1.0 / n_e)
        tau_e = tau_m * x
        weight_e = np.full(n_entanglement_modes, 1.0 / n_entanglement_modes)

        weights = [weight_e]
        taus = [tau_e]
        tau_c = tau_c_over_tau_m * tau_m

        if include_glassy:
            log_edges = np.linspace(
                np.log(tau_c * glassy_lower_ratio),
                np.log(tau_c),
                n_glassy_modes + 1,
            )
            log_centres = 0.5 * (log_edges[:-1] + log_edges[1:])
            tau_g = np.exp(log_centres)
            dlog = np.diff(log_edges)
            weight_g = n_e * (tau_g / tau_c) ** (-n_g) * dlog
            weights.append(weight_g)
            taus.append(tau_g)

        all_weights = np.concatenate(weights)
        all_taus = np.concatenate(taus)
        return cls(
            weights=all_weights,
            taus=all_taus,
            name="continuous BSW quadrature",
            metadata={
                "tau_d": float(tau_d),
                "tau_m": float(tau_m),
                "n_e": float(n_e),
                "n_g": float(n_g),
                "tau_c": float(tau_c),
                "tau_c_over_tau_m": float(tau_c_over_tau_m),
                "include_glassy": bool(include_glassy),
                "glassy_lower_ratio": float(glassy_lower_ratio),
                "source_status": (
                    "Equations exact; n_e/n_g externally recovered; "
                    "tau_c/tau_m reconstructed from Figure 8"
                ),
            },
        )
