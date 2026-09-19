import numpy as np

from hassager2021 import (
    EulerianConfig,
    LagrangianConfig,
    PronySpectrum,
    simulate_eulerian,
    simulate_lagrangian,
)


def test_eulerian_rolie_smoke() -> None:
    result = simulate_eulerian(
        EulerianConfig(
            model="rolie_poly_toy",
            target_rate=2.0,
            end_time=0.5,
            beta=0.4,
            eta_r=0.2,
            n_points=101,
            cfl=0.5,
            stokes_strength=0.3,
            snapshot_times=(0.0, 0.5),
        )
    )
    assert np.max(np.abs(result.tracking_error)) < 8.0e-3
    assert np.all(result.final_area > 0.0)
    assert abs(result.volume[-1] / result.volume[0] - 1.0) < 0.03


def test_lagrangian_exact_midplane_smoke() -> None:
    result = simulate_lagrangian(
        LagrangianConfig(
            model="de_toy",
            startup_rate=2.0,
            startup_time=0.5,
            dt=0.05,
            n_nodes=51,
            eta_r=0.06,
            include_molecular_stretch=True,
            stretch_relaxation_time=1.0e-6,
            control="exact_midplane",
        ),
        PronySpectrum.single_mode(1.0),
    )
    assert np.max(np.abs(result.tracking_error)) < 1.0e-10
    assert np.all(result.local_stretch_ratio > 0.0)
    assert np.all(result.molecular_stretch > 0.0)
    assert result.newton_iterations.max() <= 6


def test_lagrangian_long_protocol_smoke() -> None:
    result = simulate_lagrangian(
        LagrangianConfig(
            model="de_toy",
            startup_rate=0.03,
            startup_time=100.0,
            relaxation_time=100.0,
            dt=1.0,
            n_nodes=51,
            eta_r=0.01,
            include_molecular_stretch=True,
            stretch_relaxation_time=100.0,
            control="exact_midplane",
        ),
        PronySpectrum.single_mode(1000.0),
    )
    assert np.isclose(result.true_strain[-1], 3.0, atol=1.0e-10)
    assert result.nominal_strain[-1] < result.nominal_strain[100]


def test_eulerian_conservative_scheme_is_exact_in_volume_and_tracking() -> None:
    result = simulate_eulerian(
        EulerianConfig(
            model="rolie_poly_toy",
            target_rate=2.0,
            end_time=1.0,
            beta=0.4,
            eta_r=0.2,
            n_points=201,
            cfl=0.5,
            stokes_strength=0.3,
            area_scheme="conservative",
            snapshot_times=(0.0, 1.0),
        )
    )
    assert abs(result.volume[-1] / result.volume[0] - 1.0) < 1.0e-12
    assert np.max(np.abs(result.tracking_error)) < 1.0e-12
    pointwise = simulate_eulerian(
        EulerianConfig(
            model="rolie_poly_toy",
            target_rate=2.0,
            end_time=1.0,
            beta=0.4,
            eta_r=0.2,
            n_points=201,
            cfl=0.5,
            stokes_strength=0.3,
            area_scheme="pointwise",
            snapshot_times=(0.0, 1.0),
        )
    )
    # While the neck is resolved the two schemes agree closely.
    assert abs(result.nominal_strain[-1] - pointwise.nominal_strain[-1]) < 5.0e-3
    assert abs(result.mid_stress[-1] - pointwise.mid_stress[-1]) < 5.0e-3
    assert result.snapshots[1.0].neck_cells > 10
