import numpy as np
import pytest

from hassager2021 import LagrangianConfig, PronySpectrum, simulate_lagrangian
from hassager2021.analytical import de_steady_state

ANALYTICAL = 2.7092145477785126  # DE-toy, Wi=2, 6% solvent, Wi_R << 1


def _homogeneous(scheme: str, dt: float):
    return simulate_lagrangian(
        LagrangianConfig(
            model="de_toy",
            startup_rate=2.0,
            startup_time=2.0,
            dt=dt,
            n_nodes=41,
            eta_r=0.06,
            include_molecular_stretch=True,
            stretch_relaxation_time=1.0e-6,
            control="exact_midplane",
            stokes_coefficient=0.0,
            time_discretization=scheme,
        ),
        PronySpectrum.single_mode(1.0),
    )


def test_second_order_scheme_hits_analytical_steady_stress() -> None:
    coarse = _homogeneous("second_order", 0.05)
    fine = _homogeneous("second_order", 0.025)
    assert abs(coarse.true_stress[-1] - ANALYTICAL) < 2.0e-3
    # error ratio close to 4 for a second-order scheme
    ratio = abs(coarse.true_stress[-1] - ANALYTICAL) / abs(fine.true_stress[-1] - ANALYTICAL)
    assert 3.0 < ratio < 5.0


def test_original_scheme_has_documented_first_order_bias() -> None:
    coarse = _homogeneous("matlab_first_order", 0.05)
    fine = _homogeneous("matlab_first_order", 0.025)
    assert 0.09 < coarse.true_stress[-1] - ANALYTICAL < 0.13
    ratio = (coarse.true_stress[-1] - ANALYTICAL) / (fine.true_stress[-1] - ANALYTICAL)
    assert 1.7 < ratio < 2.3


def test_oldroyd_b_lagrangian_matches_homogeneous_solution() -> None:
    # Analytical Oldroyd-B startup at Wi=1 with eta_s=0.25 (20% solvent).
    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        lambda t, y: [2.0 * y[0] - (y[0] - 1.0), -y[1] - (y[1] - 1.0)],
        (0.0, 2.0),
        [1.0, 1.0],
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    expected = sol.y[0, -1] - sol.y[1, -1] + 3.0 * 0.25 * 1.0
    result = simulate_lagrangian(
        LagrangianConfig(
            model="oldroyd_b",
            startup_rate=1.0,
            startup_time=2.0,
            dt=0.02,
            n_nodes=41,
            eta_r=0.2,
            include_molecular_stretch=False,
            control="exact_midplane",
            stokes_coefficient=0.0,
        ),
        PronySpectrum.single_mode(1.0),
    )
    assert np.isclose(result.true_stress[-1], expected, rtol=2.0e-4)


def test_bsw_sub_step_modes_are_lumped_into_solvent() -> None:
    bsw = PronySpectrum.bsw(tau_d=1.0, tau_c_over_tau_m=1.0e-3)
    steady = de_steady_state(rate=2.0, spectrum=bsw, eta_r=0.03).total_stress
    result = simulate_lagrangian(
        LagrangianConfig(
            model="de_toy",
            startup_rate=2.0,
            startup_time=2.0,
            dt=0.05,
            n_nodes=41,
            eta_r=0.03,
            include_molecular_stretch=True,
            stretch_relaxation_time=1.0e-6,
            control="exact_midplane",
            stokes_coefficient=0.0,
        ),
        bsw,
    )
    assert result.lumped_fast_mode_viscosity > 0.0
    assert abs(result.true_stress[-1] - steady) < 5.0e-3


@pytest.mark.parametrize("scheme", ["matlab_first_order", "second_order"])
def test_stretch_relaxation_protocol_runs_for_both_schemes(scheme: str) -> None:
    result = simulate_lagrangian(
        LagrangianConfig(
            model="de_toy",
            startup_rate=0.03,
            startup_time=100.0,
            relaxation_time=100.0,
            dt=2.0,
            n_nodes=41,
            eta_r=0.01,
            include_molecular_stretch=True,
            stretch_relaxation_time=100.0,
            control="exact_midplane",
            time_discretization=scheme,
        ),
        PronySpectrum.single_mode(1000.0),
    )
    mid = result.molecular_stretch.shape[0] // 2
    assert 2.3 < result.molecular_stretch[mid].max() < 2.6
    assert result.nominal_strain[-1] < result.nominal_strain[50]
