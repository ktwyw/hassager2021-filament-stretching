"""Generate paper-style panel figures and numerical data files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .analytical import (
    de_homogeneous_startup,
    de_mode_steady_orientation,
    de_steady_state,
    de_steady_stretch,
    rolie_startup,
    rolie_steady_z,
)
from .eulerian import EulerianConfig, EulerianResult, simulate_eulerian
from .lagrangian import LagrangianConfig, LagrangianResult, simulate_lagrangian
from .spectra import PronySpectrum


def _finish_figure(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _save_eulerian_data(path: Path, result: EulerianResult) -> None:
    payload: dict[str, Any] = {
        "time": result.time,
        "true_strain": result.true_strain,
        "nominal_strain": result.nominal_strain,
        "mid_stress": result.mid_stress,
        "engineering_tension": result.engineering_tension,
        "nominal_rate": result.nominal_rate,
        "mid_rate": result.mid_rate,
        "volume": result.volume,
        "u": result.u,
        "initial_area": result.initial_area,
        "final_area": result.final_area,
        "final_conformation": result.final_conformation,
    }
    for time, snap in result.snapshots.items():
        tag = f"t_{time:.6g}".replace(".", "p")
        payload[f"{tag}_z"] = snap.z
        payload[f"{tag}_radius"] = snap.radius
        payload[f"{tag}_area"] = snap.physical_area
        payload[f"{tag}_Z"] = snap.conformation
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _save_lagrangian_data(path: Path, result: LagrangianResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time=result.time,
        target_true_strain=result.target_true_strain,
        true_strain=result.true_strain,
        nominal_strain=result.nominal_strain,
        true_stress=result.true_stress,
        engineering_force=result.engineering_force,
        positions=result.positions,
        segment_midpoints=result.segment_midpoints,
        radius=result.radius,
        molecular_stretch=result.molecular_stretch,
        local_stretch_ratio=result.local_stretch_ratio,
        newton_errors=result.newton_errors,
        newton_iterations=result.newton_iterations,
    )


TRUE_COLOR = "tab:blue"
NOMINAL_COLOR = "tab:red"


def _plot_strains(
    time: np.ndarray,
    true_strain: np.ndarray,
    nominal_strain: np.ndarray,
    *,
    title: str,
    path: Path,
    dpi: int,
    xlabel: str = "Time",
) -> None:
    """True (blue) and nominal (red) Hencky strain, in the paper's colours."""

    plt.figure(figsize=(6.4, 4.7))
    plt.plot(time, true_strain, color=TRUE_COLOR, linewidth=2.0, label=r"true $\epsilon$")
    plt.plot(
        time, nominal_strain, color=NOMINAL_COLOR, linewidth=2.0, label=r"nominal $\epsilon_N$"
    )
    plt.xlabel(xlabel)
    plt.ylabel("Hencky strain")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    _finish_figure(path, dpi)


def _plot_eulerian_profiles(
    result: EulerianResult,
    *,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(6.4, 4.7))
    for time in sorted(result.snapshots):
        snap = result.snapshots[time]
        plt.plot(snap.z, snap.radius, linewidth=1.9, label=f"t={time:.2f}")
    plt.xlabel("Physical length, z")
    plt.ylabel("Radius, R")
    plt.ylim(0.0, 1.04)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2)
    _finish_figure(path, dpi)


def _plot_lagrangian_profiles(
    result: LagrangianResult,
    times: tuple[float, ...],
    *,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(6.4, 4.7))
    for requested in times:
        snap = result.snapshot(requested)
        plt.plot(
            snap.segment_midpoints,
            snap.radius,
            linewidth=1.9,
            label=f"t={snap.time:.2f}",
        )
    plt.xlabel("Physical length, z")
    plt.ylabel("Radius, R")
    plt.ylim(0.0, 1.04)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2)
    _finish_figure(path, dpi)


def _plot_oldroyd_comparison(
    eulerian: EulerianResult,
    lagrangian: LagrangianResult,
    times: tuple[float, ...],
    *,
    path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(6.5, 4.8))
    for requested in times:
        e_snap = eulerian.snapshots[requested]
        l_snap = lagrangian.snapshot(requested)
        plt.plot(
            e_snap.z,
            e_snap.radius,
            linewidth=1.8,
            label=f"t={requested:.2f}",
        )
        plt.plot(
            l_snap.segment_midpoints,
            l_snap.radius,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Lagrangian" if requested == times[0] else None,
        )
    plt.xlabel("Physical length, z")
    plt.ylabel("Radius, R")
    plt.ylim(0.3, 1.04)
    plt.title("Figure 3(b): Oldroyd-B, Eulerian (colour) vs Lagrangian (dashed)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2)
    _finish_figure(path, dpi)


def _plot_stress_curve(
    time: np.ndarray,
    stress: np.ndarray,
    steady: float,
    *,
    title: str,
    path: Path,
    dpi: int,
    ylabel: str = r"Stress, $\sigma/G_0$",
    xlabel: str = "Time",
    analytical: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Mid-filament stress from the filament simulation (blue), the analytical
    steady stress (red), and optionally the homogeneous analytical transient
    (thin dashed black) as an independent check of the filament solver."""

    plt.figure(figsize=(6.4, 4.7))
    plt.plot(time, stress, color=TRUE_COLOR, linewidth=2.0, label=r"simulated mid-filament $\sigma$")
    if analytical is not None:
        plt.plot(
            analytical[0],
            analytical[1],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="homogeneous analytical transient",
        )
    plt.axhline(steady, color=NOMINAL_COLOR, linewidth=1.8, label=r"analytical steady $\sigma_a$")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    _finish_figure(path, dpi)


def _figure_01_schematic(path: Path, dpi: int) -> None:
    plt.figure(figsize=(8.0, 3.5))
    ax = plt.gca()
    ax.add_patch(Rectangle((0.2, -1.15), 0.15, 2.3, alpha=0.6))
    ax.add_patch(Rectangle((1.65, -1.15), 0.15, 2.3, alpha=0.6))
    ax.plot([0.35, 1.65], [1.0, 1.0], linewidth=2.2)
    ax.plot([0.35, 1.65], [-1.0, -1.0], linewidth=2.2)
    ax.text(1.0, 1.2, "initial filament")

    ax.add_patch(Rectangle((3.0, -1.15), 0.15, 2.3, alpha=0.6))
    ax.add_patch(Rectangle((7.3, -1.15), 0.15, 2.3, alpha=0.6))
    z = np.linspace(3.15, 7.3, 300)
    radius = 0.25 + 0.75 * (
        np.exp(-((z - 3.15) / 0.35) ** 2)
        + np.exp(-((7.3 - z) / 0.35) ** 2)
    )
    radius = np.minimum(radius, 1.0)
    ax.plot(z, radius, linewidth=2.2)
    ax.plot(z, -radius, linewidth=2.2)
    ax.axvline((3.15 + 7.3) / 2.0, linestyle=":", linewidth=1.3)
    ax.text(4.65, 1.2, "stretched filament and midplane")
    ax.set_xlim(0.0, 7.7)
    ax.set_ylim(-1.45, 1.55)
    ax.set_xlabel("Axial coordinate, z")
    ax.set_ylabel("Radial coordinate, r")
    ax.set_title("Figure 1 computational schematic (photo intentionally omitted)")
    ax.set_aspect("equal", adjustable="box")
    _finish_figure(path, dpi)


def reproduce_all(
    output_directory: str | Path,
    *,
    quality: str = "standard",
    dpi: int = 200,
) -> dict[str, Any]:
    """Reproduce Figures 1--10 and return diagnostic metrics.

    ``quality='fast'`` reduces spatial resolution for quick checking.  The
    standard preset remains modest enough for a laptop while resolving the
    narrow necks in Figures 5, 7, and 8.
    """

    out = Path(output_directory)
    figures = out / "figures"
    data_dir = out / "data"
    figures.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if quality not in {"fast", "standard"}:
        raise ValueError("quality must be 'fast' or 'standard'.")
    n_e = 201 if quality == "fast" else 401
    n_l = 101 if quality == "fast" else 401
    n_long = 101 if quality == "fast" else 201

    metrics: dict[str, Any] = {
        "quality": quality,
        "reconstruction_notes": {
            "eulerian_controller": "ideal exact midplane-rate constraint",
            "eulerian_area_scheme": "pointwise (Hoyle-Fielding non-conservative upwind); 'conservative' available",
            "lagrangian_controller": "ideal exact midplane-strain constraint",
            "lagrangian_time_discretization": "second_order (trapezoidal history, BDF2 solvent rate, theta-method stretch, sub-step modes lumped)",
            "bsw_tau_c_over_tau_m": 1.0e-3,
            "eulerian_solvent_factor": 1.0,
            "lagrangian_solvent_factor": 3.0,
        },
    }

    _figure_01_schematic(figures / "figure_01_schematic.png", dpi)

    # Figure 2: Newtonian liquid.
    times_2 = (0.0, 0.39, 0.79, 1.20, 1.60, 2.00)
    newtonian = simulate_eulerian(
        EulerianConfig(
            model="newtonian",
            target_rate=1.0,
            end_time=2.0,
            n_points=n_e,
            cfl=0.5,
            stokes_strength=0.30,
            snapshot_times=times_2,
        )
    )
    _plot_strains(
        newtonian.time,
        newtonian.true_strain,
        newtonian.nominal_strain,
        title="Figure 2(a): controlled Newtonian stretching",
        path=figures / "figure_02a_newtonian_strains.png",
        dpi=dpi,
    )
    _plot_eulerian_profiles(
        newtonian,
        title="Figure 2(b): Newtonian radial profiles",
        path=figures / "figure_02b_newtonian_profiles.png",
        dpi=dpi,
    )
    _save_eulerian_data(data_dir / "figure_02_newtonian.npz", newtonian)
    metrics["figure_02"] = {
        "final_true_strain": float(newtonian.true_strain[-1]),
        "final_nominal_strain": float(newtonian.nominal_strain[-1]),
        "relative_volume_error": float(newtonian.volume[-1] / newtonian.volume[0] - 1.0),
    }

    # Figure 3: Oldroyd-B, Eulerian and Lagrangian.
    times_3 = (0.0, 0.40, 0.80, 1.20, 1.60, 2.00)
    oldroyd_e = simulate_eulerian(
        EulerianConfig(
            model="oldroyd_b_toy",
            target_rate=1.0,
            end_time=2.0,
            tau=1.0,
            eta_r=0.2,
            n_points=n_e,
            cfl=0.5,
            stokes_strength=0.25,
            snapshot_times=times_3,
        )
    )
    single_1 = PronySpectrum.single_mode(1.0)
    oldroyd_l = simulate_lagrangian(
        LagrangianConfig(
            model="oldroyd_b",
            startup_rate=1.0,
            startup_time=2.0,
            dt=0.02,
            n_nodes=n_l,
            eta_r=0.2,
            include_molecular_stretch=False,
            control="exact_midplane",
        ),
        single_1,
    )
    _plot_strains(
        oldroyd_e.time,
        oldroyd_e.true_strain,
        oldroyd_e.nominal_strain,
        title=r"Figure 3(a): Oldroyd-B, $\eta_r=0.2$, Wi=1",
        path=figures / "figure_03a_oldroyd_strains.png",
        dpi=dpi,
    )
    _plot_oldroyd_comparison(
        oldroyd_e,
        oldroyd_l,
        times_3,
        path=figures / "figure_03b_oldroyd_profiles_comparison.png",
        dpi=dpi,
    )
    _save_eulerian_data(data_dir / "figure_03_oldroyd_eulerian.npz", oldroyd_e)
    _save_lagrangian_data(data_dir / "figure_03_oldroyd_lagrangian.npz", oldroyd_l)
    metrics["figure_03"] = {
        "eulerian_final_nominal_strain": float(oldroyd_e.nominal_strain[-1]),
        "lagrangian_final_nominal_strain": float(oldroyd_l.nominal_strain[-1]),
        "lagrangian_tracking_error_max": float(np.max(np.abs(oldroyd_l.tracking_error))),
    }

    # Figure 4(a): steady scalar Rolie-Poly toy curves.
    wi = np.logspace(-2.0, 2.0, 500)
    plt.figure(figsize=(6.4, 4.7))
    for beta in (0.05, 0.10, 0.20, 0.40):
        plt.loglog(wi, rolie_steady_z(wi, beta), linewidth=1.8, label=fr"$\beta={beta:.2f}$")
    wi_ob = np.logspace(-2.0, np.log10(0.495), 250)
    plt.loglog(wi_ob, rolie_steady_z(wi_ob, 0.0), linewidth=1.8, label=r"Oldroyd-B toy, $\beta=0$")
    plt.xlabel("Non-dimensional rate, Wi")
    plt.ylabel(r"Non-dimensional stress, $\sigma/G_0$")
    plt.title("Figure 4(a): steady non-stretching Rolie-Poly toy")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2, which="both")
    _finish_figure(figures / "figure_04a_rolie_steady_stress.png", dpi)

    t_short = np.linspace(0.0, 2.0, 401)
    rolie_growth = rolie_startup(
        t_short,
        rate=2.0,
        tau=1.0,
        beta=0.4,
        eta_r=0.2,
        solvent_factor=1.0,
    )

    # Figures 4(b) and 5 are one simulation: the unstable Rolie-Poly filament.
    rolie_filament = simulate_eulerian(
        EulerianConfig(
            model="rolie_poly_toy",
            target_rate=2.0,
            end_time=2.0,
            tau=1.0,
            beta=0.4,
            eta_r=0.2,
            n_points=n_e,
            cfl=0.5,
            stokes_strength=0.30,
            pre_stretch=0.0,
            artificial_diffusion=0.0,
            snapshot_times=times_3,
        )
    )
    _plot_stress_curve(
        rolie_filament.time,
        rolie_filament.mid_stress,
        rolie_growth.steady_stress,
        title=r"Figure 4(b): Rolie-Poly toy, $\eta_r=0.2$, $\beta=0.4$, Wi=2",
        path=figures / "figure_04b_rolie_stress_growth.png",
        dpi=dpi,
        analytical=(rolie_growth.time, rolie_growth.stress),
    )
    metrics["figure_04"] = {
        "beta_0p4_high_rate_limit": float((1.0 + np.sqrt(1.8)) / 0.4),
        "wi_2_steady_stress": float(rolie_growth.steady_stress),
        "simulated_mid_stress_at_t2": float(rolie_filament.mid_stress[-1]),
    }

    _plot_strains(
        rolie_filament.time,
        rolie_filament.true_strain,
        rolie_filament.nominal_strain,
        title=r"Figure 5(a): Rolie-Poly toy, $\eta_r=0.2$, $\beta=0.4$, Wi=2",
        path=figures / "figure_05a_rolie_strains.png",
        dpi=dpi,
    )
    _plot_eulerian_profiles(
        rolie_filament,
        title="Figure 5(b): localized Rolie-Poly neck",
        path=figures / "figure_05b_rolie_profiles.png",
        dpi=dpi,
    )
    _save_eulerian_data(data_dir / "figure_05_rolie_filament.npz", rolie_filament)
    metrics["figure_05"] = {
        "peak_nominal_strain": float(np.max(rolie_filament.nominal_strain)),
        "final_nominal_strain": float(rolie_filament.nominal_strain[-1]),
        "final_minimum_radius": float(np.min(rolie_filament.snapshots[2.0].radius)),
        "neck_cells_at_t2": rolie_filament.snapshots[2.0].neck_cells,
        "neck_cells_at_t1p6": rolie_filament.snapshots[1.6].neck_cells,
        "relative_volume_error": float(rolie_filament.volume[-1] / rolie_filament.volume[0] - 1.0),
    }

    # Figure 6(a): DE-toy steady curve, with and without stretch.
    wi_de = np.logspace(-1.0, 2.0, 260)
    orientation = np.asarray(de_mode_steady_orientation(wi_de))
    stress_nonstretch = 3.0 * orientation
    stretch_values = np.array(
        [de_steady_stretch(float(d), 0.1 * float(w)) for d, w in zip(orientation, wi_de, strict=True)]
    )
    stress_stretch = 3.0 * orientation * stretch_values**2
    plt.figure(figsize=(6.4, 4.7))
    plt.loglog(wi_de, stress_nonstretch, linewidth=2.0, label="non-stretching")
    plt.loglog(wi_de, stress_stretch, linestyle="--", linewidth=2.0, label=r"stretching, $Wi_R=0.1Wi$")
    plt.xlabel("Non-dimensional rate, Wi")
    plt.ylabel(r"Non-dimensional stress, $\sigma/G_0$")
    plt.title("Figure 6(a): steady DE-toy stress")
    plt.legend()
    plt.grid(alpha=0.2, which="both")
    _finish_figure(figures / "figure_06a_de_steady_stress.png", dpi)

    de_growth = de_homogeneous_startup(
        t_short, rate=2.0, spectrum=single_1, eta_r=0.06
    )

    # Figures 6(b) and 7 are one simulation: the single-mode DE-toy filament.
    de_single_filament = simulate_lagrangian(
        LagrangianConfig(
            model="de_toy",
            startup_rate=2.0,
            startup_time=2.0,
            dt=0.05,
            n_nodes=n_l,
            eta_r=0.06,
            include_molecular_stretch=True,
            stretch_relaxation_time=1.0e-6,
            control="exact_midplane",
        ),
        single_1,
    )
    _plot_stress_curve(
        de_single_filament.time,
        de_single_filament.true_stress,
        de_growth.steady_stress,
        title=r"Figure 6(b): single-mode DE-toy, $\eta_r=0.06$, Wi=2, $Wi_R\ll1$",
        path=figures / "figure_06b_de_stress_growth.png",
        dpi=dpi,
        analytical=(de_growth.time, de_growth.stress),
    )
    metrics["figure_06"] = {
        "nonstretch_high_wi_limit": 3.0,
        "wi_2_steady_stress": float(de_growth.steady_stress),
        "simulated_mid_stress_at_t2": float(de_single_filament.true_stress[-1]),
        "wi_100_stretch_stress": float(stress_stretch[-1]),
    }

    _plot_strains(
        de_single_filament.time,
        de_single_filament.true_strain,
        de_single_filament.nominal_strain,
        title=r"Figure 7(a): single-mode DE-toy, $\eta_r=0.06$, Wi=2",
        path=figures / "figure_07a_de_strains.png",
        dpi=dpi,
    )
    _plot_lagrangian_profiles(
        de_single_filament,
        times_3,
        title="Figure 7(b): single-mode DE-toy radial profiles",
        path=figures / "figure_07b_de_profiles.png",
        dpi=dpi,
    )
    _save_lagrangian_data(data_dir / "figure_07_de_single_mode.npz", de_single_filament)
    metrics["figure_07"] = {
        "peak_nominal_strain": float(np.max(de_single_filament.nominal_strain)),
        "final_nominal_strain": float(de_single_filament.nominal_strain[-1]),
        "final_minimum_radius": float(np.min(de_single_filament.radius[:, -1])),
        "neck_segments_at_t2": de_single_filament.snapshot(2.0).neck_segments,
        "neck_segments_at_t1p6": de_single_filament.snapshot(1.6).neck_segments,
        "tracking_error_max": float(np.max(np.abs(de_single_filament.tracking_error))),
    }

    # Figure 8: continuous BSW spectrum. tau_c/tau_m is a transparent inference.
    bsw = PronySpectrum.bsw(
        tau_d=1.0,
        n_e=0.23,
        n_g=0.70,
        tau_c_over_tau_m=1.0e-3,
        include_glassy=True,
    )
    bsw_growth = de_homogeneous_startup(
        t_short, rate=2.0, spectrum=bsw, eta_r=0.03
    )
    bsw_filament = simulate_lagrangian(
        LagrangianConfig(
            model="de_toy",
            startup_rate=2.0,
            startup_time=2.0,
            dt=0.05,
            n_nodes=n_l,
            eta_r=0.03,
            include_molecular_stretch=True,
            stretch_relaxation_time=1.0e-6,
            control="exact_midplane",
        ),
        bsw,
    )
    _plot_stress_curve(
        bsw_filament.time,
        bsw_filament.true_stress,
        bsw_growth.steady_stress,
        title=r"Figure 8(a): BSW DE-toy, $\eta_r=0.03$, Wi=2, $Wi_R\ll1$",
        path=figures / "figure_08a_bsw_stress_growth.png",
        dpi=dpi,
        analytical=(bsw_growth.time, bsw_growth.stress),
    )
    _plot_lagrangian_profiles(
        bsw_filament,
        times_3,
        title="Figure 8(b): BSW DE-toy radial profiles",
        path=figures / "figure_08b_bsw_profiles.png",
        dpi=dpi,
    )
    _save_lagrangian_data(data_dir / "figure_08_de_bsw.npz", bsw_filament)
    metrics["figure_08"] = {
        "bsw_tau_m": float(bsw.metadata["tau_m"]),
        "bsw_tau_c_over_tau_m": float(bsw.metadata["tau_c_over_tau_m"]),
        "zero_shear_time": float(bsw.zero_shear_time),
        "steady_stress": float(bsw_growth.steady_stress),
        "simulated_mid_stress_at_t2": float(bsw_filament.true_stress[-1]),
        "lumped_sub_step_mode_viscosity": float(bsw_filament.lumped_fast_mode_viscosity),
        "final_nominal_strain": float(bsw_filament.nominal_strain[-1]),
        "neck_segments_at_t2": bsw_filament.snapshot(2.0).neck_segments,
    }

    # Figures 9 and 10: startup then relaxation with molecular stretch.
    long_single = PronySpectrum.single_mode(1000.0)
    stretch_relaxation = simulate_lagrangian(
        LagrangianConfig(
            model="de_toy",
            startup_rate=0.03,
            relaxation_rate=0.0,
            startup_time=100.0,
            relaxation_time=400.0,
            dt=1.0,
            n_nodes=n_long,
            G0=1.0,
            eta_r=0.01,
            include_molecular_stretch=True,
            stretch_relaxation_time=100.0,
            control="exact_midplane",
            max_newton_iterations=12,
        ),
        long_single,
    )
    _plot_strains(
        stretch_relaxation.time,
        stretch_relaxation.true_strain,
        stretch_relaxation.nominal_strain,
        title=r"Figure 9(a): DE-toy startup/relaxation, $\eta_r=0.01$, Wi=30, $Wi_R=3$",
        path=figures / "figure_09a_relaxation_strains.png",
        dpi=dpi,
        xlabel="Time (s)",
    )
    long_steady = de_steady_state(
        rate=0.03,
        spectrum=long_single,
        eta_r=0.01,
        tau_s=100.0,
    )
    _plot_stress_curve(
        stretch_relaxation.time,
        stretch_relaxation.true_stress,
        long_steady.total_stress,
        title=r"Figure 9(b): mid-filament stress, $\sigma/G_0$ ($G_0=250$ kPa)",
        path=figures / "figure_09b_relaxation_stress.png",
        dpi=dpi,
        xlabel="Time (s)",
    )

    long_times = (1.0, 100.0, 200.0, 300.0, 400.0, 500.0)
    _plot_lagrangian_profiles(
        stretch_relaxation,
        long_times,
        title="Figure 10(a): radial profiles during stress relaxation",
        path=figures / "figure_10a_relaxation_profiles.png",
        dpi=dpi,
    )
    plt.figure(figsize=(6.4, 4.7))
    for requested in long_times:
        snap = stretch_relaxation.snapshot(requested)
        plt.plot(
            snap.segment_midpoints,
            snap.molecular_stretch,
            linewidth=1.9,
            label=f"t={snap.time:.0f} s",
        )
    plt.xlabel("Physical length, z")
    plt.ylabel("Molecular stretch")
    plt.ylim(0.45, 3.0)
    plt.title("Figure 10(b): molecular stretch profiles")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2)
    _finish_figure(figures / "figure_10b_molecular_stretch.png", dpi)
    _save_lagrangian_data(
        data_dir / "figure_09_10_startup_relaxation.npz", stretch_relaxation
    )
    metrics["figure_09_10"] = {
        "analytical_steady_stress_over_G0": float(long_steady.total_stress),
        "analytical_steady_stress_kPa": float(250.0 * long_steady.total_stress),
        "simulated_peak_stress_over_G0": float(np.max(stretch_relaxation.true_stress)),
        "nominal_strain_at_100s": float(stretch_relaxation.nominal_strain[100]),
        "nominal_strain_at_500s": float(stretch_relaxation.nominal_strain[-1]),
        "maximum_midplane_molecular_stretch": float(
            np.max(stretch_relaxation.molecular_stretch[stretch_relaxation.molecular_stretch.shape[0] // 2, :])
        ),
        "tracking_error_max": float(np.max(np.abs(stretch_relaxation.tracking_error))),
    }

    with (out / "reproduction_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics
