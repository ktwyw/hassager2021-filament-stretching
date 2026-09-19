import numpy as np

from hassager2021 import (
    PronySpectrum,
    de_mode_steady_orientation,
    de_steady_state,
    rolie_high_rate_limit,
    rolie_steady_z,
)


def test_rolie_positive_root_and_limit() -> None:
    z = rolie_steady_z(2.0, 0.4)
    residual = 2.0 * (2.0 + 2.0 * z - 0.4 * z**2) - z
    assert abs(residual) < 1.0e-12
    assert np.isclose(rolie_high_rate_limit(0.4), 5.854101966249685)


def test_oldroyd_toy_singularity_side() -> None:
    assert np.isclose(rolie_steady_z(0.25, 0.0), 1.0)
    assert rolie_steady_z(0.49, 0.0) > 40.0


def test_de_single_mode_steady_level() -> None:
    orientation = de_mode_steady_orientation(2.0)
    assert np.isclose(orientation, 0.77541194, rtol=2.0e-7)
    state = de_steady_state(
        rate=2.0,
        spectrum=PronySpectrum.single_mode(1.0),
        eta_r=0.06,
    )
    assert np.isclose(state.total_stress, 2.7092145, rtol=2.0e-6)


def test_bsw_scaling() -> None:
    spectrum = PronySpectrum.bsw(
        tau_d=1.0,
        tau_c_over_tau_m=1.0e-3,
        include_glassy=True,
        n_entanglement_modes=160,
        n_glassy_modes=160,
    )
    assert np.isclose(spectrum.metadata["tau_m"], (1.0 + 0.23) / 0.23)
    assert np.isclose(spectrum.zero_shear_time, 1.004, rtol=3.0e-3)
    state = de_steady_state(rate=2.0, spectrum=spectrum, eta_r=0.03)
    assert 1.52 < state.total_stress < 1.58
