#!/usr/bin/env python3
"""Discretization convergence of both filament solvers.

Part 1 (Lagrangian): the Figure 6(b)/7 preset (single-mode DE-toy, Wi=2, 6%
solvent, Wi_R << 1) for the original MATLAB-program scheme and the second-order scheme:
mid-filament stress at t=2 (analytical steady value 2.70921 G0) and the peak
nominal Hencky strain, the most sensitive geometric quantity because it is set
by the necking instability.

Part 2 (Eulerian): grid refinement of the Figure 4(b)/5 preset (Rolie-Poly toy,
Wi=2, 20% solvent) for the pointwise Hoyle-Fielding scheme and the
conservative finite-volume scheme: final nominal strain, volume drift,
midplane tracking error, mid stress at t=1.5 and t=2, and the number of grid
points across the neck at t=2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hassager2021 import (
    EulerianConfig,
    LagrangianConfig,
    PronySpectrum,
    simulate_eulerian,
    simulate_lagrangian,
)
from hassager2021.analytical import de_steady_state, rolie_startup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs") / "convergence")
    parser.add_argument("--n-nodes", type=int, default=201)
    parser.add_argument(
        "--dts", type=float, nargs="+", default=(0.1, 0.05, 0.025, 0.0125)
    )
    parser.add_argument("--grids", type=int, nargs="+", default=(201, 401, 801, 1601))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    spectrum = PronySpectrum.single_mode(1.0)
    steady = de_steady_state(rate=2.0, spectrum=spectrum, eta_r=0.06).total_stress
    rows: list[dict[str, float | str]] = []
    for scheme in ("matlab_first_order", "second_order"):
        for dt in args.dts:
            config = LagrangianConfig(
                model="de_toy",
                startup_rate=2.0,
                startup_time=2.0,
                dt=dt,
                n_nodes=args.n_nodes,
                eta_r=0.06,
                include_molecular_stretch=True,
                stretch_relaxation_time=1.0e-6,
                control="exact_midplane",
                time_discretization=scheme,
                strict_convergence=False,
            )
            try:
                result = simulate_lagrangian(config, spectrum)
            except RuntimeError as exc:  # pragma: no cover - reporting only
                rows.append({"scheme": scheme, "dt": dt, "error": str(exc)})
                continue
            rows.append(
                {
                    "scheme": scheme,
                    "dt": dt,
                    "stress_t2": float(result.true_stress[-1]),
                    "stress_error": float(result.true_stress[-1] - steady),
                    "peak_nominal_strain": float(result.nominal_strain.max()),
                    "final_nominal_strain": float(result.nominal_strain[-1]),
                    "max_newton_iterations": int(result.newton_iterations.max()),
                }
            )

    print(f"analytical steady stress: {steady:.5f}")
    print(f"{'scheme':<20}{'dt':>8}{'stress(t=2)':>14}{'error':>10}{'peak eps_N':>12}{'final eps_N':>13}")
    for row in rows:
        if "error" in row:
            print(f"{row['scheme']:<20}{row['dt']:>8}  {row['error']}")
        else:
            print(
                f"{row['scheme']:<20}{row['dt']:>8.4f}{row['stress_t2']:>14.5f}"
                f"{row['stress_error']:>+10.5f}{row['peak_nominal_strain']:>12.4f}"
                f"{row['final_nominal_strain']:>13.4f}"
            )
    with (args.output / "convergence.json").open("w", encoding="utf-8") as handle:
        json.dump({"analytical_steady_stress": steady, "rows": rows}, handle, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for scheme, marker in (("matlab_first_order", "s"), ("second_order", "o")):
        sel = [r for r in rows if r["scheme"] == scheme and "error" not in r]
        dts = np.array([r["dt"] for r in sel])
        axes[0].loglog(
            dts, np.abs([r["stress_error"] for r in sel]), marker=marker, label=scheme
        )
        axes[1].semilogx(
            dts, [r["peak_nominal_strain"] for r in sel], marker=marker, label=scheme
        )
    ref = np.array(args.dts)
    axes[0].loglog(ref, 0.1 * ref / ref[0] * 2.0, "k:", label=r"$\propto\Delta t$")
    axes[0].loglog(ref, 1.3e-3 * (ref / 0.05) ** 2, "k--", label=r"$\propto\Delta t^2$")
    axes[0].set_xlabel(r"$\Delta t$")
    axes[0].set_ylabel(r"$|\sigma(t{=}2)-\sigma_a|/G_0$")
    axes[0].set_title("Stress error, DE-toy Wi=2 (Figure 6b preset)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2, which="both")
    axes[1].set_xlabel(r"$\Delta t$")
    axes[1].set_ylabel(r"peak nominal strain $\epsilon_N$")
    axes[1].set_title("Neck geometry, Figure 7(a) preset")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(args.output / "convergence.png", dpi=160)

    # ---- Part 2: Eulerian grid refinement --------------------------------
    analytical = rolie_startup(np.linspace(0.0, 2.0, 401), rate=2.0, beta=0.4, eta_r=0.2)
    sigma_15 = float(np.interp(1.5, analytical.time, analytical.stress))
    sigma_20 = float(analytical.stress[-1])
    e_rows: list[dict[str, float | int | str]] = []
    for scheme in ("pointwise", "conservative"):
        for n in args.grids:
            result = simulate_eulerian(
                EulerianConfig(
                    model="rolie_poly_toy",
                    target_rate=2.0,
                    end_time=2.0,
                    beta=0.4,
                    eta_r=0.2,
                    n_points=n,
                    cfl=0.5,
                    stokes_strength=0.30,
                    area_scheme=scheme,
                    snapshot_times=(2.0,),
                )
            )
            e_rows.append(
                {
                    "scheme": scheme,
                    "n_points": n,
                    "peak_nominal_strain": float(result.nominal_strain.max()),
                    "final_nominal_strain": float(result.nominal_strain[-1]),
                    "volume_error": float(result.volume[-1] / result.volume[0] - 1.0),
                    "tracking_error": float(np.max(np.abs(result.tracking_error))),
                    "stress_t1p5": float(np.interp(1.5, result.time, result.mid_stress)),
                    "stress_t2": float(result.mid_stress[-1]),
                    "neck_cells_t2": result.snapshots[2.0].neck_cells,
                }
            )
    print(f"\nEulerian Rolie-Poly preset; analytical sigma(1.5)={sigma_15:.4f}, sigma(2)={sigma_20:.4f}")
    print(
        f"{'scheme':<14}{'n':>6}{'peak eps_N':>12}{'final eps_N':>13}{'vol err':>10}"
        f"{'track err':>11}{'sigma(1.5)':>12}{'sigma(2)':>10}{'neck pts':>10}"
    )
    for row in e_rows:
        print(
            f"{row['scheme']:<14}{row['n_points']:>6}{row['peak_nominal_strain']:>12.4f}"
            f"{row['final_nominal_strain']:>13.4f}{row['volume_error']:>10.1e}"
            f"{row['tracking_error']:>11.1e}{row['stress_t1p5']:>12.4f}{row['stress_t2']:>10.4f}"
            f"{row['neck_cells_t2']:>10d}"
        )
    with (args.output / "convergence_eulerian.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"analytical_stress_t1p5": sigma_15, "analytical_stress_t2": sigma_20, "rows": e_rows},
            handle,
            indent=2,
        )
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
