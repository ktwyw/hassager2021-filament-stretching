#!/usr/bin/env python3
"""README images: the hero animation and the gallery, computed with the package.

Run from the repository root after `python reproduce_all.py` (the gallery reads the
`outputs/data/*.npz` arrays and `outputs/convergence/convergence.json` written by the
reproduction and convergence scripts):

    python make_readme_images.py --output docs/images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from hassager2021 import LagrangianConfig, PronySpectrum, simulate_lagrangian
from hassager2021.analytical import de_homogeneous_startup, de_mode_steady_orientation, de_steady_stretch, rolie_steady_z

STYLE = {
    "figure.dpi": 110,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
}
COLORS = ["#1f4e79", "#e07a1f", "#2e8b57", "#a23b72", "#7a5230", "#444444"]


def _filament(ax, z, r, color="#1f4e79", alpha=0.9):
    """Draw a symmetric filament r(z) with its two end plates."""
    ax.fill_between(z, -r, r, color=color, alpha=alpha, lw=0)
    ax.plot(z, r, color=color, lw=1); ax.plot(z, -r, color=color, lw=1)
    for z0 in (z[0], z[-1]):
        ax.add_patch(Rectangle((z0 - 0.08, -1.25), 0.16, 2.5, color="#555555"))


def hero(out: Path) -> None:
    spectrum = PronySpectrum.single_mode(1.0)
    cfg = LagrangianConfig(model="de_toy", startup_rate=2.0, startup_time=2.0, dt=0.02, n_nodes=201, eta_r=0.06,
                           include_molecular_stretch=True, stretch_relaxation_time=1.0e-6, control="exact_midplane")
    res = simulate_lagrangian(cfg, spectrum)
    growth = de_homogeneous_startup(np.linspace(0.0, 2.0, 401), rate=2.0, spectrum=spectrum, eta_r=0.06)
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=80, gridspec_kw={"width_ratios": [1.25, 1]})
        ax1.set_xlim(-0.3, 4.3); ax1.set_ylim(-1.45, 1.45); ax1.set_aspect("equal"); ax1.grid(False)
        ax1.set_xlabel("axial position z  (initial radius = 1)"); ax1.set_yticks([-1, 0, 1])
        base_title = "single-mode DE-toy filament at Wi = 2, midplane strain rate held constant\n"
        ax2.plot(growth.time, growth.stress, "k--", lw=1, label="homogeneous analytical transient")
        ax2.axhline(growth.steady_stress, color="0.6", lw=0.8, label=r"analytical steady $\sigma_s$")
        (line,) = ax2.plot([], [], color=COLORS[1], lw=2, label="simulated mid-filament stress")
        (dot,) = ax2.plot([], [], "o", color=COLORS[1])
        ax2.set_xlim(0, 2.05); ax2.set_ylim(0, 3.2); ax2.set_xlabel("time  (units of the relaxation time)"); ax2.set_ylabel(r"$\sigma / G_0$")
        ax2.legend(fontsize=8, loc="lower right")
        ax1.set_title(base_title, fontsize=9.5)
        fig.tight_layout()
        fig.subplots_adjust(top=0.84)

        def draw(i):
            for art in list(ax1.collections) + list(ax1.lines) + list(ax1.patches):
                art.remove()
            z = res.segment_midpoints[:, i]; r = res.radius[:, i]
            _filament(ax1, z, r)
            ax1.set_title(base_title + f"t = {res.time[i]:.2f}     midplane Hencky strain {res.true_strain[i]:.2f}     plate (nominal) strain {res.nominal_strain[i]:.2f}", fontsize=9.5)
            line.set_data(res.time[: i + 1], res.true_stress[: i + 1]); dot.set_data([res.time[i]], [res.true_stress[i]])
            return line, dot

        FuncAnimation(fig, draw, frames=len(res.time), blit=False).save(out / "hero.gif", writer=PillowWriter(fps=12))
        plt.close(fig)


def _profiles_panel(ax, d, tags, title):
    for c, tag in zip(COLORS, tags):
        z, r = d[f"{tag}_z"], d[f"{tag}_radius"]
        ax.plot(z, r, color=c, lw=1.5, label=f"t = {tag[2:].replace('p', '.')}")
    ax.set_xlabel("physical length z"); ax.set_ylabel("radius R"); ax.set_ylim(0, 1.05); ax.set_title(title, fontsize=10); ax.legend(fontsize=8)


def gallery(data: Path, out: Path) -> None:
    with plt.rc_context(STYLE):
        # 1. Newtonian vs Rolie-Poly necking (Figures 2 and 5)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        _profiles_panel(axes[0], np.load(data / "figure_02_newtonian.npz"), ["t_0", "t_0p39", "t_0p79", "t_1p2", "t_1p6", "t_2"], "Newtonian: the filament thins uniformly")
        _profiles_panel(axes[1], np.load(data / "figure_05_rolie_filament.npz"), ["t_0", "t_0p4", "t_0p8", "t_1p2", "t_1p6", "t_2"], r"Rolie-Poly toy, Wi = 2: a neck localises")
        fig.tight_layout(); fig.savefig(out / "necking.png"); plt.close(fig)

        # 2. Stress growth vs the homogeneous analytical transient (Figures 6(b), 8(a))
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
        bsw = PronySpectrum.bsw(tau_d=1.0, n_e=0.23, n_g=0.70, tau_c_over_tau_m=1.0e-3, include_glassy=True)
        for ax, (name, title, spectrum, eta_r) in zip(axes, (("figure_07_de_single_mode.npz", "single-mode DE-toy, Wi = 2", PronySpectrum.single_mode(1.0), 0.06), ("figure_08_de_bsw.npz", "DE-toy with a BSW relaxation spectrum, Wi = 2", bsw, 0.03))):
            d = np.load(data / name)
            ax.plot(d["time"], d["true_stress"], color=COLORS[1], lw=2, label="simulated mid-filament stress")
            g = de_homogeneous_startup(np.linspace(0.0, 2.0, 401), rate=2.0, spectrum=spectrum, eta_r=eta_r)
            ax.plot(g.time, g.stress, "k--", lw=1, label="homogeneous analytical transient"); ax.axhline(g.steady_stress, color="0.6", lw=0.8, label="analytical steady stress")
            ax.set_xlabel("time"); ax.set_ylabel(r"$\sigma/G_0$"); ax.set_title(title, fontsize=10); ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout(); fig.savefig(out / "stress_growth.png"); plt.close(fig)

        # 3. Eulerian vs Lagrangian solvers on the Oldroyd-B filament (Figure 3(b))
        de, dl = np.load(data / "figure_03_oldroyd_eulerian.npz"), np.load(data / "figure_03_oldroyd_lagrangian.npz")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for c, tag in zip(COLORS, ["t_0", "t_0p4", "t_0p8", "t_1p2", "t_1p6", "t_2"]):
            ax.plot(de[f"{tag}_z"], de[f"{tag}_radius"], color=c, lw=1.6, label=f"Eulerian, t = {tag[2:].replace('p', '.')}")
            t = float(tag[2:].replace("p", "."))
            j = int(np.argmin(np.abs(dl["time"] - t)))
            ax.plot(dl["segment_midpoints"][:, j], dl["radius"][:, j], "k--", lw=1, label="Lagrangian" if tag == "t_0" else None)
        ax.set_xlabel("physical length z"); ax.set_ylabel("radius R"); ax.set_ylim(0.3, 1.05); ax.set_title("Oldroyd-B, Wi = 1: two independent solvers, one answer", fontsize=10); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(out / "solvers_agree.png"); plt.close(fig)

        # 4. Stress relaxation and molecular stretch (Figures 9-10)
        d = np.load(data / "figure_09_10_startup_relaxation.npz")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].plot(d["time"], d["true_stress"], color=COLORS[1], lw=2); axes[0].set_xlabel("time (s)"); axes[0].set_ylabel(r"$\sigma/G_0$"); axes[0].set_title("start-up at Wi = 30, then the plates stop: stress relaxation", fontsize=10)
        for c, t in zip(COLORS, (100, 200, 300, 400, 500)):
            j = int(np.argmin(np.abs(d["time"] - t)))
            axes[1].plot(d["segment_midpoints"][:, j], d["molecular_stretch"][:, j], color=c, lw=1.5, label=f"t = {t} s")
        axes[1].axhline(1, color="0.6", lw=0.8); axes[1].set_xlabel("physical length z"); axes[1].set_ylabel("molecular stretch"); axes[1].set_title("the stretch relaxes in the neck and inverts near the plates", fontsize=10); axes[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "relaxation.png"); plt.close(fig)

        # 5. Steady stress of the toy models (Figures 4(a), 6(a))
        wi = np.logspace(-2, 2, 200)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for c, beta in zip(COLORS, (0.05, 0.1, 0.2, 0.4)):
            axes[0].loglog(wi, rolie_steady_z(wi, beta), color=c, lw=1.6, label=rf"$\beta$ = {beta}")
        axes[0].set_xlabel("Wi"); axes[0].set_ylabel(r"$\sigma/G_0$"); axes[0].set_title("Rolie-Poly toy: steady stress saturates with stretch relaxation", fontsize=10); axes[0].legend(fontsize=8)
        wi_de = np.logspace(-1.0, 2.0, 260)
        orientation = np.asarray(de_mode_steady_orientation(wi_de))
        stretch = np.array([de_steady_stretch(float(d), 0.1 * float(w)) for d, w in zip(orientation, wi_de, strict=True)])
        axes[1].loglog(wi_de, 3.0 * orientation, color=COLORS[0], lw=1.6, label="non-stretching DE-toy (plateau at 3)")
        axes[1].loglog(wi_de, 3.0 * orientation * stretch**2, "--", color=COLORS[1], lw=1.6, label=r"with molecular stretch, $Wi_R = 0.1\,Wi$")
        axes[1].set_xlabel("Wi"); axes[1].set_ylabel(r"$\sigma/G_0$"); axes[1].set_title("DE-toy: orientation saturates, stretch does not", fontsize=10); axes[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "steady_stress.png"); plt.close(fig)


def convergence(conv: Path, out: Path) -> None:
    rows = json.loads((conv / "convergence.json").read_text())["rows"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for scheme, c, lab in (("matlab_first_order", COLORS[3], "original MATLAB scheme (first order)"), ("second_order", COLORS[0], "second-order scheme (default)")):
            r = [x for x in rows if x["scheme"] == scheme]
            ax.loglog([x["dt"] for x in r], [abs(x["stress_error"]) for x in r], "o-", color=c, lw=1.6, label=lab)
        dts = np.array([0.0125, 0.1])
        ax.loglog(dts, 0.25 * dts / 0.1, ":", color="0.5", lw=1, label=r"$\propto dt$"); ax.loglog(dts, 0.025 * (dts / 0.1) ** 2, "--", color="0.5", lw=1, label=r"$\propto dt^2$")
        ax.set_xlabel("time step dt"); ax.set_ylabel(r"|stress error at t = 2| / $G_0$"); ax.set_title("time-step convergence of the Lagrangian solver", fontsize=10); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "convergence.png"); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("docs") / "images")
    p.add_argument("--data", type=Path, default=Path("outputs") / "data")
    p.add_argument("--convergence", type=Path, default=Path("outputs") / "convergence")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    gallery(a.data, a.output)
    convergence(a.convergence, a.output)
    hero(a.output)
    for f in sorted(a.output.iterdir()):
        print(f"{f.name:<20} {f.stat().st_size / 1024:7.1f} kB")
