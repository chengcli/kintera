#!/usr/bin/env python3
"""Paper figure: kintera reproduces the analytic Chapman ozone layer.

kintera's Photolysis and Arrhenius modules compute the photolysis rates J(z)
and rate constants k(z) on a stratospheric column (the TOA solar flux is
attenuated through the overhead O2 column). The Chapman steady state is then
obtained two independent ways:
  * by integrating the box kinetics with kintera's implicit solver
    (evolve_implicit), and
  * from the closed-form analytic Chapman steady state of the same J, k.
They agree to the solver tolerance and reproduce the canonical ozone layer.

Run under the dev env:
  /opt/anaconda3/bin/python3.10 fig_chapman_match.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import kintera
from kintera import (Reaction, PhotolysisOptions, Photolysis,
                     ArrheniusOptions, Arrhenius)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
OUT = Path(__file__).resolve().parent
NA = 6.02214076e23
KB = 1.380649e-23


def build_kinetics():
    kintera.set_species_names(["N2", "O2", "O", "O3"])
    kintera.set_species_weights([28e-3, 32e-3, 16e-3, 48e-3])
    kintera.set_species_cref_R([2.5, 2.5, 1.5, 3.5])

    # O2: strong Schumann-Runge below ~190 nm (absorbed high in the mesosphere)
    # + the weak Herzberg continuum 200-242 nm (~few 1e-24 cm^2) that penetrates
    # to the stratosphere and drives the ozone layer near 25 km.
    wave_o2 = [120., 140., 160., 175., 185., 195., 200., 210., 220., 230.,
               242.]
    xs_o2 = [1.0e-17, 1.0e-17, 1.0e-17, 1.0e-19, 1.0e-21, 8.0e-23, 3.5e-23,
             3.0e-23, 2.3e-23, 1.4e-23, 5.0e-24]
    wave_o3 = [200., 210., 220., 230., 240., 250., 255., 260., 270., 280.,
               290., 300., 310., 320., 330.]
    xs_o3 = [3.0e-19, 6.0e-19, 1.0e-18, 2.5e-18, 5.0e-18, 8.0e-18, 1.1e-17,
             1.0e-17, 7.0e-18, 4.0e-18, 2.0e-18, 8.0e-19, 3.0e-19, 5.0e-20,
             5.0e-21]
    waves = sorted(set(wave_o2 + wave_o3))

    def interp(wv, xs, w):
        return float(np.interp(w, wv, xs, left=xs[0], right=xs[-1]))
    sig_o2 = np.array([interp(wave_o2, xs_o2, w) for w in waves])   # cm^2
    sig_o3 = np.array([interp(wave_o3, xs_o3, w) for w in waves])
    # O2 does not absorb beyond the Schumann-Runge/Herzberg edge (~242 nm), so
    # it must NOT attenuate the Hartley-band (O3) flux -- otherwise J_O2 and
    # J_O3 attenuate in lock-step and the ozone-layer peak vanishes.
    sig_o2[np.asarray(waves) > 242.0] = 0.0

    po = PhotolysisOptions()
    po.reactions([Reaction("O2 => 2 O"), Reaction("O3 => O2 + O")])
    po.wavelength([float(w) for w in waves])
    po.temperature([200., 300.])
    po.cross_section(list(sig_o2) + list(sig_o3))
    po.branches([[{"O2": 1.0}], [{"O3": 1.0}]])
    photo = Photolysis(po)

    ao = ArrheniusOptions()
    ao.reactions([Reaction("O + O2 => O3"), Reaction("O + O3 => 2 O2")])
    ao.A([1.7e-14, 8.0e-12]); ao.b([-2.4, 0.0]); ao.Ea_R([0.0, 2060.0])
    arr = Arrhenius(ao)
    return photo, arr, np.array(waves), sig_o2


def main():
    photo, arr, waves, sig_o2 = build_kinetics()
    nw = waves.size

    # --- stratospheric column ----------------------------------------------
    z = np.linspace(8.0, 64.0, 49)                       # km
    T = np.full_like(z, 250.0)                            # isothermal
    H = 7.0e3                                             # scale height [m]
    n_tot = (1.013e5 / (KB * 250.0)) * np.exp(-z * 1e3 / H)   # m^-3
    n_cgs = n_tot * 1e-6                                   # molecule cm^-3
    # kintera's Arrhenius A here is cgs (cm^3 molecule^-1 s^-1), so the box runs
    # in molecule/cm^3 throughout; J is s^-1.
    # overhead O2 column [cm^-2] = n_O2[cm^-3] * H[cm] (exponential atmosphere)
    col_o2 = (0.21 * n_cgs) * (H * 1e2)

    # TOA solar actinic flux [photons cm^-2 s^-1 per bin], example shape
    f_toa = np.where(waves < 200, 1e10 * np.exp(-(200 - waves) / 30),
            np.where(waves < 320, 1e13 * np.exp(-(waves - 250) ** 2 / 5000),
                     1e14))
    # attenuate per altitude through the overhead O2 column (Beer-Lambert)
    flux = f_toa[None, :] * np.exp(-sig_o2[None, :] * col_o2[:, None])  # (nz,nw)

    tT = torch.tensor(T)
    # isothermal column: set the dissociation xs once at the single temperature,
    # then evaluate J per altitude (the flux is what varies with height)
    t1 = torch.tensor([float(T[0])])
    photo.update_xs_diss_stacked(t1)
    Jrows = [photo.forward(t1, torch.tensor(flux[i]))[0] for i in range(z.size)]
    J = torch.stack(Jrows)                                # (nz, 2) [1/s]
    J_O2, J_O3 = J[:, 0].numpy(), J[:, 1].numpy()

    pres = torch.tensor(n_tot * KB * T)                   # Pa
    nt = torch.tensor(n_cgs)                              # molecule cm^-3
    c = torch.zeros(z.size, 4)
    c[:, 0] = 0.78 * nt                                   # N2
    c[:, 1] = 0.21 * nt                                   # O2
    c[:, 2] = 1e-12 * nt                                  # O seed
    c[:, 3] = 1e-9 * nt                                   # O3 seed
    k = arr.forward(tT, pres, c, {})                      # (nz, 2) cm^3/s
    k2, k4 = k[:, 0].numpy(), k[:, 1].numpy()

    # --- analytic Chapman steady state (exact, from kintera J, k) -----------
    o2 = (0.21 * nt).numpy()
    disc = (J_O2 * k4) ** 2 + 4 * k2 * k4 * J_O2 * J_O3
    O_an = (J_O2 * k4 + np.sqrt(disc)) / (2 * k2 * k4)     # [O] mol/m^3
    O3_an = J_O2 * o2 / (k4 * O_an)                        # [O3] mol/m^3

    # --- kintera implicit integration of the box to steady state ------------
    S = torch.tensor([[0., 0., 0., 0.],          # N2
                      [-1., -1., 1., 2.],        # O2
                      [2., -1., 1., -1.],        # O
                      [0., 1., -1., -1.]])       # O3
    jO2 = torch.tensor(J_O2); jO3 = torch.tensor(J_O3)
    k2t = torch.tensor(k2); k4t = torch.tensor(k4)

    def rate_jac(cc):
        c1, c2, c3 = cc[:, 1], cc[:, 2], cc[:, 3]
        r = torch.stack([jO2 * c1, k2t * c2 * c1, jO3 * c3, k4t * c2 * c3], -1)
        jac = torch.zeros(z.size, 4, 4)
        jac[:, 0, 1] = jO2
        jac[:, 1, 1] = k2t * c2; jac[:, 1, 2] = k2t * c1
        jac[:, 2, 3] = jO3
        jac[:, 3, 2] = k4t * c3; jac[:, 3, 3] = k4t * c2
        return r, jac

    cc = c.clone()
    dt = 1.0e-2
    for _ in range(120):                                  # ramp dt to SS
        r, jac = rate_jac(cc)
        dc = kintera.evolve_implicit(r, S, jac, dt)
        # hold the N2/O2 bath fixed (the analytic SS assumes constant O2);
        # only the odd-oxygen species O, O3 evolve
        cc[:, 2] = (cc[:, 2] + dc[:, 2]).clamp_min(0.0)
        cc[:, 3] = (cc[:, 3] + dc[:, 3]).clamp_min(0.0)
        dt = min(dt * 1.4, 1.0e9)
    O_kt, O3_kt = cc[:, 2].numpy(), cc[:, 3].numpy()

    rel_O3 = np.abs(O3_kt - O3_an) / np.maximum(O3_an, 1e-300)
    rel_O = np.abs(O_kt - O_an) / np.maximum(O_an, 1e-300)
    # assess only cells where the species is non-negligible; deep in the column
    # J_O2 underflows and both kt and analytic are ~0 (meaningless ratio noise)
    alive = O3_an > 1e-5 * O3_an.max()
    print(f"O3 peak at z={z[np.argmax(O3_an)]:.0f} km; max rel diff over alive "
          f"cells: O3={rel_O3[alive].max():.2e}  O={rel_O[alive].max():.2e}")

    f = 1.0                                              # already molecule cm^-3
    # --- figure ------------------------------------------------------------
    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans",
                         "axes.linewidth": 0.9, "mathtext.default": "regular",
                         "xtick.direction": "in", "ytick.direction": "in"})
    fig, axL = plt.subplots(1, 1, figsize=(6.2, 5.4))

    axL.plot(O3_an * f, z, "-", color="#1f77b4", lw=2.2, label="O$_3$ analytic")
    axL.plot(O3_kt[::2] * f, z[::2], "o", color="#1f77b4", ms=6, mfc="white",
             mew=1.4, label="O$_3$ kintera")
    axL.plot(O_an * f, z, "-", color="#d62728", lw=2.2, label="O analytic")
    axL.plot(O_kt[::2] * f, z[::2], "s", color="#d62728", ms=5.5, mfc="white",
             mew=1.4, label="O kintera")
    axL.set_xscale("log")
    axL.set_xlabel("number density  [cm$^{-3}$]")
    axL.set_ylabel("altitude  [km]")
    axL.legend(loc="upper right", fontsize=10, frameon=False)
    axL.set_ylim(z.min(), z.max())
    axL.grid(alpha=0.25)
    axL.text(0.04, 0.04, "kintera vs analytic\nmax rel. diff = "
             f"{max(rel_O3[alive].max(), rel_O[alive].max()):.0e}",
             transform=axL.transAxes, ha="left", va="bottom", fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7"))

    fig.suptitle("kintera reproduces the analytic\nChapman ozone steady state",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_chapman_match.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"wrote {OUT}/fig_chapman_match.png / .pdf")


if __name__ == "__main__":
    main()
