"""Titan C2 photochemistry module for the snapy GCM (device-agnostic).

Runs under snapy's pyenv (python3.12, pip kintera) AND the dev environment:
it depends only on torch, numpy, and the *old* kintera API
(``KineticsOptions.from_yaml``, ``Kinetics.forward_nogil``, ``jacobian``,
``evolve_implicit``).

Three rate families are combined into one implicit step:

1. kintera-native thermal reactions (42 arrhenius + 4 Troe falloff) from the
   generated ``titan_c2_chem.yaml`` (validated against the dev KINETICS-base
   stack by ``validate_c2_network.py``).
2. The 11 KB ``UPDATE_CHEMB`` custom-rate reactions, vendored verbatim from
   ``kintera/python/kinetics_base/titan/chemb_overrides.py`` (the dev file is
   not installed in snapy's environment). Units: cm-based (KB-native);
   converted to SI inside :meth:`TitanC2Chemistry.rates`.
3. Photolysis branches (13): ``rate_r = J_r * [parent]`` with diagonal
   Jacobian; J comes from :meth:`photolysis_rates` (per-bin sigma*F sum, the
   KB convention) given an actinic-flux field from the RT driver.

State convention: ``scalar_s`` carries species partial densities [kg/m^3]
(snapy passive scalars, species-FIRST axis); concentrations are mol/m^3.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import kintera

N_AVOGADRO = 6.02214076e23
CM3_TO_SI = 1.0e6 / N_AVOGADRO    # molecule/cm^3/s -> mol/m^3/s
SI_TO_CM3 = N_AVOGADRO / 1.0e6    # mol/m^3 -> molecule/cm^3


# ---------------------------------------------------------------------------
# KB UPDATE_CHEMB custom rates, vendored verbatim from
# python/kinetics_base/titan/chemb_overrides.py (validated Stage-1 layer).
# k units: cm^6/s for M-reactions (full rate = k*[A][B][M]), cm^3/s otherwise.
# ---------------------------------------------------------------------------
def _troe_falloff(k_low, k_inf, density, fc=0.6):
    ratio = k_low * density / torch.clamp(k_inf, min=1e-300)
    safe_ratio = torch.clamp(ratio, min=1e-30)
    log_ratio = torch.log10(safe_ratio)
    fc_factor = torch.pow(torch.full_like(ratio, fc),
                          1.0 / (1.0 + log_ratio * log_ratio))
    return k_low / (1.0 + ratio) * fc_factor


def _k_2h_m_h2(t, d):
    rk3 = torch.clamp(2.7e-31 * torch.pow(t, -0.6), max=1.0e-32)
    rk2 = torch.full_like(t, 1.0e-11)
    return _troe_falloff(rk3, rk2, d)


def _k_h_ch3_m_ch4(t, d):
    rk3 = torch.where(t <= 316.0, torch.full_like(t, 3.46e-29),
                      7.81e-18 * torch.pow(t, -3.87) * torch.exp(-1222.0 / t))
    rk2 = torch.where(t <= 105.0, torch.full_like(t, 4.8e-11),
                      4.6e-7 * torch.pow(t, -1.0) * torch.exp(-474.0 / t))
    fc = torch.clamp(0.31 + torch.exp(-t / 425.0), max=1.0)
    return _troe_falloff(rk3, rk2, d) * (fc / 0.6)


def _k_h_c2h2_m_c2h3(t, d):
    rk3 = 3.34e-26 * torch.pow(t, -1.46) * torch.exp(-1144.0 / t)
    rk2 = 2.3e-11 * torch.exp(-1350.0 / t)
    return _troe_falloff(rk3, rk2, d)


def _k_h_c2h3_m_c2h4(t, d):
    rk3 = 1.75e-27 * torch.pow(t, -0.3)
    rk2 = 7.0e-11 * torch.pow(t, 0.18)
    return _troe_falloff(rk3, rk2, d)


def _k_h_c2h3_c2h2_h2(t, d):
    rk2 = 7.0e-11 * torch.pow(t, 0.18)
    return torch.clamp(rk2 - _k_h_c2h3_m_c2h4(t, d) * d, min=0.0)


def _k_h_c2h4_m_c2h5(t, d):
    rk3 = 5.4e-25 * torch.pow(t, -1.46) * torch.exp(-1300.0 / t)
    rk2 = 1.8e-13 * torch.pow(t, 0.70) * torch.exp(-600.0 / t)
    return _troe_falloff(rk3, rk2, d)


def _k_h_c2h5_m_c2h6(t, d):
    rk3 = torch.where(t <= 200.0, torch.full_like(t, 2.489e-27),
                      4.0e-19 * torch.pow(t, -3.0) * torch.exp(-600.0 / t))
    rk2 = torch.full_like(t, 2.0e-10)
    return _troe_falloff(rk3, rk2, d)


def _k_ch_ch4_c2h4_h(t, d):
    return torch.where(t <= 295.0,
                       3.96e-8 * torch.pow(t, -1.04) * torch.exp(-36.1 / t),
                       1.58e-8 * torch.pow(t, -0.9))


def _k_2ch3_m_c2h6(t, d):
    rk3 = torch.where(t <= 300.0, 6.15e-18 * torch.pow(t, -3.5),
                      3.51e-7 * torch.pow(t, -7.03) * torch.exp(-1390.0 / t))
    rk2 = 1.12e-9 * torch.pow(t, -0.5) * torch.exp(-25.0 / t)
    return _troe_falloff(rk3, rk2, d)


def _k_c2h3_h2_c2h4_h(t, d):
    return 5.23e-15 * torch.pow(t, 0.7) * torch.exp(-2574.0 / t)


def _k_2c2h3_m_c4h6(t, d):
    # needed by the derived 2C2H3 -> C2H4 + C2H2 branch (rk2 - k*d); the
    # C4H6 channel itself is OUTSIDE the C2 species set and is not applied.
    rk3 = 5.0e-18 * torch.pow(t, -3.75) * torch.exp(-300.0 / t)
    rk2 = torch.full_like(t, 1.4e-10)
    return _troe_falloff(rk3, rk2, d)


def _k_2c2h3_c2h4_c2h2(t, d):
    rk2 = torch.full_like(t, 1.4e-10)
    return torch.clamp(rk2 - _k_2c2h3_m_c4h6(t, d) * d, min=0.0)


# signature key: (sorted reactants incl M, sorted products incl M)
CUSTOM_RATES = {
    (("H", "H", "M"), ("H2", "M")): (_k_2h_m_h2, True),
    (("CH3", "H", "M"), ("CH4", "M")): (_k_h_ch3_m_ch4, True),
    (("C2H2", "H", "M"), ("C2H3", "M")): (_k_h_c2h2_m_c2h3, True),
    (("C2H3", "H"), ("C2H2", "H2")): (_k_h_c2h3_c2h2_h2, False),
    (("C2H3", "H", "M"), ("C2H4", "M")): (_k_h_c2h3_m_c2h4, True),
    (("C2H4", "H", "M"), ("C2H5", "M")): (_k_h_c2h4_m_c2h5, True),
    (("C2H5", "H", "M"), ("C2H6", "M")): (_k_h_c2h5_m_c2h6, True),
    (("CH", "CH4"), ("C2H4", "H")): (_k_ch_ch4_c2h4_h, False),
    (("CH3", "CH3", "M"), ("C2H6", "M")): (_k_2ch3_m_c2h6, True),
    (("C2H3", "H2"), ("C2H4", "H")): (_k_c2h3_h2_c2h4_h, False),
    (("C2H3", "C2H3"), ("C2H2", "C2H4")): (_k_2c2h3_c2h4_c2h2, False),
}


GAMMA_ROS2 = 1.0 + 1.0 / 2.0 ** 0.5    # 1 + 1/sqrt(2): the L-stable ROS2 root


def _batched_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve A x = b (A:(...,n,n), b:(...,n)) with a Tikhonov fallback for the
    singular stiff cells (an algebraic photochemical balance can make A exactly
    singular at large dt: CUDA ``linalg.solve`` RAISES, CPU may return
    inf/nan)."""
    n = A.shape[-1]
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    try:
        x = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:                                       # _LinAlgError
        lam = 1e-10 * A.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-300)
        x = torch.linalg.solve(A + lam * eye, b.unsqueeze(-1)).squeeze(-1)
    bad = ~torch.isfinite(x).all(dim=-1)
    if bad.any():
        lam = 1e-10 * A[bad].abs().amax(dim=(-2, -1),
                                        keepdim=True).clamp_min(1e-300)
        x[bad] = torch.linalg.solve(A[bad] + lam * eye,
                                    b[bad].unsqueeze(-1)).squeeze(-1)
    return x


def evolve_implicit_torch(rate: torch.Tensor, stoich: torch.Tensor,
                          jac: torch.Tensor, dt: float) -> torch.Tensor:
    """Linearized backward-Euler step, mirroring ``kintera.evolve_implicit``:
    solve (I/dt - S J) delta = S rate  for delta (..., nsp).

    Pure batched torch so it runs on any device; kintera's compiled CUDA
    kernel requests more shared memory than available for this network size
    (73 reactions x 16 species). Verified equivalent to
    ``kintera.evolve_implicit`` on CPU in
    ``test_titan_c2.py::test_b0_evolve_matches_kintera``. Retained for that
    gate and as the first-order reference; the runtime integrator is ROS2
    (:meth:`TitanC2Chemistry._ros2_substep`).

    Args:
        rate: (..., nrxn); stoich: (nsp, nrxn); jac: (..., nrxn, nsp)
    """
    nsp = stoich.shape[0]
    b = torch.einsum("sr,...r->...s", stoich, rate)            # S rate
    sj = torch.einsum("sr,...rn->...sn", stoich, jac)          # S J
    eye = torch.eye(nsp, dtype=rate.dtype, device=rate.device)
    return _batched_solve(eye / dt - sj, b)


def _parse_side(side: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for term in side.split("+"):
        term = term.strip()
        if not term or term == "M":
            continue
        parts = term.split()
        if len(parts) == 2:
            out[parts[1]] = out.get(parts[1], 0) + int(parts[0])
        else:
            out[term] = out.get(term, 0) + 1
    return out


def normalize_equation(eq: str) -> tuple:
    """Signature of an irreversible equation, robust to formatting."""
    lhs, rhs = eq.replace("<=>", "=>").split("=>")
    l, r = _parse_side(lhs), _parse_side(rhs)
    key = lambda dd: tuple(sorted((k, v) for k, v in dd.items()))
    return (key(l), key(r))


class TitanC2Chemistry:
    """C2 photochemistry source: kintera thermal + KB custom + photolysis."""

    def __init__(self, yaml_path: str | Path, data_npz: str | Path,
                 device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float64):
        self.device = torch.device(device)
        self.dtype = dtype
        d = np.load(data_npz, allow_pickle=False)
        self.species: list[str] = [str(s) for s in d["storage_species"]]
        self.nsp = len(self.species)
        self.sp = {s: i for i, s in enumerate(self.species)}
        self.mw = torch.tensor(d["mw"], dtype=dtype, device=self.device)  # kg/mol

        # --- kintera thermal block -------------------------------------
        opts = kintera.KineticsOptions.from_yaml(str(yaml_path))
        self.kinet = kintera.Kinetics(opts)
        self.kinet.to(self.device)
        self.kin_species: list[str] = list(opts.species())
        assert set(self.kin_species) <= set(self.species), (
            f"kinetics species not in storage set: "
            f"{set(self.kin_species) - set(self.species)}")
        self._kin_idx = torch.tensor(
            [self.sp[s] for s in self.kin_species], device=self.device)
        kin_stoich = self.kinet.buffer("stoich").to(dtype=dtype)  # (nsp_kin, nrxn)
        self.nrxn_kin = kin_stoich.shape[1]
        self.kin_equations = [r.equation() for r in opts.reactions()]

        # --- custom (KB UPDATE_CHEMB) block ------------------------------
        self.custom = []
        for reac, prod in zip(d["custom_reactants"], d["custom_products"]):
            rl, pl = str(reac).split("|"), str(prod).split("|")
            fn, has_m = self._lookup_custom(rl, pl)
            self.custom.append(dict(reactants=rl, products=pl, fn=fn, has_m=has_m))
        self.nrxn_custom = len(self.custom)

        # --- photolysis block --------------------------------------------
        self.wavelengths = torch.tensor(d["wavelengths"], dtype=dtype,
                                        device=self.device)
        self.nwave = self.wavelengths.numel()
        self.photo_sigma = torch.tensor(d["photo_sigma"], dtype=dtype,
                                        device=self.device)  # (nphoto, nwave) cm^2
        self.toa_flux = torch.tensor(d["toa_flux"], dtype=dtype,
                                     device=self.device)     # photons/cm^2/s
        self.photo_parents = [str(s) for s in d["photo_parents"]]
        self.photo_products = [str(s).split("|") for s in d["photo_products"]]
        self.photo_parent_idx = torch.tensor(
            [self.sp[p] for p in self.photo_parents], device=self.device)
        self.nphoto = len(self.photo_parents)

        # absorbers for the RT optical depth
        self.abs_species = [str(s) for s in d["abs_species"]]
        self.abs_sigma = torch.tensor(d["abs_sigma"], dtype=dtype,
                                      device=self.device)    # (nabs, nwave) cm^2
        self.abs_idx = torch.tensor([self.sp[s] for s in self.abs_species],
                                    device=self.device)

        # --- combined stoichiometry over storage species ------------------
        ntot = self.nrxn_kin + self.nrxn_custom + self.nphoto
        stoich = torch.zeros(self.nsp, ntot, dtype=dtype, device=self.device)
        stoich[self._kin_idx, :self.nrxn_kin] = kin_stoich.to(self.device)
        for j, c in enumerate(self.custom):
            col = self.nrxn_kin + j
            for x in c["reactants"]:
                stoich[self.sp[x], col] -= 1.0
            for x in c["products"]:
                stoich[self.sp[x], col] += 1.0
        for j in range(self.nphoto):
            col = self.nrxn_kin + self.nrxn_custom + j
            stoich[self.photo_parent_idx[j], col] -= 1.0
            for p in self.photo_products[j]:
                stoich[self.sp[p], col] += 1.0
        self.stoich = stoich
        self.nrxn_total = ntot

    @staticmethod
    def _lookup_custom(rl: list[str], pl: list[str]):
        for has_m in (True, False):
            key = (tuple(sorted(rl + (["M"] if has_m else []))),
                   tuple(sorted(pl + (["M"] if has_m else []))))
            if key in CUSTOM_RATES:
                fn, m_flag = CUSTOM_RATES[key]
                return fn, m_flag
        raise KeyError(f"no custom rate for {rl} -> {pl}")

    # -- public API --------------------------------------------------------

    def photolysis_rates(self, actinic_flux: torch.Tensor) -> torch.Tensor:
        """J (..., nphoto) [1/s] from actinic flux (..., nwave) [ph/cm^2/s]."""
        return torch.einsum("...w,pw->...p", actinic_flux, self.photo_sigma)

    def rates(self, temp: torch.Tensor, pres: torch.Tensor,
              conc: torch.Tensor, jrate: torch.Tensor | None = None
              ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-reaction rates + Jacobian.

        Args:
            temp, pres: (...,) [K], [Pa]
            conc: (..., nsp) [mol/m^3], storage species order
            jrate: (..., nphoto) [1/s] photolysis rates (None -> photolysis off)
        Returns:
            rate (..., nrxn_total) [mol/m^3/s],
            jac  (..., nrxn_total, nsp) = d(rate)/d(conc)
        """
        conc = conc.clamp_min(0.0)
        out_rate = conc.new_zeros(conc.shape[:-1] + (self.nrxn_total,))
        out_jac = conc.new_zeros(conc.shape[:-1] + (self.nrxn_total, self.nsp))

        # kintera block (operates in SI on its own species subset)
        conc_kin = conc[..., self._kin_idx].contiguous()
        rate_k, rc_ddC, rc_ddT = self.kinet.forward_nogil(temp, pres, conc_kin)
        cvol = torch.ones_like(temp)
        jac_k = self.kinet.jacobian(temp, conc_kin, cvol, rate_k, rc_ddC, rc_ddT)
        out_rate[..., :self.nrxn_kin] = rate_k
        # scatter jacobian columns from kin species order into storage order
        out_jac[..., :self.nrxn_kin, :].index_copy_(
            -1, self._kin_idx,
            jac_k[..., :, :len(self.kin_species)])

        # custom block (KB cgs units)
        conc_cgs = conc * SI_TO_CM3                  # molecule/cm^3
        m_cgs = conc_cgs.sum(dim=-1)                 # total density proxy
        for j, c in enumerate(self.custom):
            col = self.nrxn_kin + j
            k = c["fn"](temp, m_cgs)                 # cm^6/s (M rxn) or cm^3/s
            r = k * (m_cgs if c["has_m"] else 1.0)
            prodc = torch.ones_like(temp)
            for x in c["reactants"]:
                prodc = prodc * conc_cgs[..., self.sp[x]]
            out_rate[..., col] = r * prodc * CM3_TO_SI
            for x in set(c["reactants"]):
                n = c["reactants"].count(x)
                dpdx = n * prodc / conc_cgs[..., self.sp[x]].clamp_min(1e-300)
                # d(rate_SI)/d(conc_SI) = d(rate_cgs)/d(conc_cgs)
                out_jac[..., col, self.sp[x]] = r * dpdx

        # photolysis block
        if jrate is not None:
            base = self.nrxn_kin + self.nrxn_custom
            cp = conc[..., self.photo_parent_idx]    # (..., nphoto)
            out_rate[..., base:] = jrate * cp
            for j in range(self.nphoto):
                out_jac[..., base + j, self.photo_parent_idx[j]] = jrate[..., j]

        return out_rate, out_jac

    def _ros2_substep(self, temp: torch.Tensor, pres: torch.Tensor,
                      conc: torch.Tensor, jrate: torch.Tensor | None,
                      dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """One 2nd-order, L-stable linearly-implicit Rosenbrock (ROS2) step.

        Returns ``(delta, err)``: the 2nd-order concentration increment and an
        embedded 2nd/1st-order local-error vector (~O(dt^2) for smooth
        solutions) used for step control. Two rate evaluations, one Jacobian,
        one factorization shared across both stages (the Rosenbrock property).
        Element-conserving: the stoichiometry annihilates any conserved-atom
        vector, so each stage solve preserves it exactly (up to the >=0 clamp).
        Tableau validated against ``kintera.evolve_ros2`` to ~1e-12 rel.
        """
        g = GAMMA_ROS2
        rate1, jac = self.rates(temp, pres, conc, jrate)
        nsp = self.nsp
        F1 = torch.einsum("sr,...r->...s", self.stoich, rate1)
        J = torch.einsum("sr,...rn->...sn", self.stoich, jac)
        eye = torch.eye(nsp, dtype=conc.dtype, device=conc.device)
        W = eye / (g * dt) - J
        k1 = _batched_solve(W, F1)
        conc2 = (conc + k1 / g).clamp_min(0.0)
        rate2, _ = self.rates(temp, pres, conc2, jrate)        # stage 2: rate only
        F2 = torch.einsum("sr,...r->...s", self.stoich, rate2)
        k2 = _batched_solve(W, F2 - (2.0 / (g * dt)) * k1)
        delta = (3.0 / (2.0 * g)) * k1 + (1.0 / (2.0 * g)) * k2
        err = 0.5 * (k2 - k1)
        return delta, err

    def advance(self, temp: torch.Tensor, pres: torch.Tensor,
                scalar_s: torch.Tensor, dt: float,
                jrate: torch.Tensor | None = None,
                rtol: float = 5.0e-2, atol: float = 1.0e-15,
                dt_max: float | None = None,
                max_substeps: int = 400) -> None:
        """Advance chemistry by dt (in-place on scalar_s) with an adaptive,
        block-synchronized ROS2 integrator.

        Each sub-step is a 2nd-order Rosenbrock step (:meth:`_ros2_substep`)
        with an embedded error estimate. ONE step size is chosen for the whole
        block from the worst-cell error norm, rather than per-cell: a dense
        batched solve processes every cell regardless of whether it has
        converged, and the GCM barrier-syncs ranks each hydro step, so per-cell
        adaptivity buys no wall time here ("they wait anyway"). What it does
        buy over a hand-tuned fixed sub-step is *error control* -- the
        controller spends sub-steps only where the chemistry is stiff (parcels
        crossing the day/night terminator, where a single first-order step
        overshoots radicals 5-10x).

        Controller: ROS2 (order 2) with an embedded order-1 estimate, so the
        step exponent is 1/(1+1) = 1/2; safety factor 1.4 (Frey et al. 2025,
        the default 0.9 systematically over-shrinks). Accept when the
        block-max scaled error <= 1, else shrink and retry. ``atol`` is in
        mol/m^3: the reactive species here span 1e-12..1e-16 while the bath is
        ~1e-5, so atol=1e-15 keeps the species that matter (CH3/H/C2Hx, all
        >~atol/rtol) in the rtol-controlled regime without chasing negligible
        trace abundances to zero.

        The step is floored at ``dt / max_substeps`` and force-accepted there,
        so the integration always covers the full ``dt`` in at most
        ``max_substeps`` accepted steps (no silent under-integration) even on a
        pathologically stiff cell -- e.g. a radical building from exactly zero,
        which only occurs at IC spin-up, not in the near-equilibrium GCM
        interior where the controller takes a handful of steps.

        Args:
            temp, pres: (...) grid fields [K], [Pa]
            scalar_s: (nsp, ...) species partial densities [kg/m^3]
            dt: total chemistry time step [s]
            jrate: (..., nphoto) photolysis rates or None (held fixed over dt)
            rtol, atol: relative / absolute tolerance for the error norm
            dt_max: optional hard cap on the sub-step [s] (None -> dt)
            max_substeps: floor the sub-step at dt/max_substeps; cap the count
        """
        grid = temp.shape
        conc0 = (scalar_s.movedim(0, -1) / self.mw).reshape(-1, self.nsp)
        flat_t = temp.reshape(-1)
        flat_p = pres.reshape(-1)
        flat_j = jrate.reshape(-1, self.nphoto) if jrate is not None else None

        conc = conc0.clamp_min(0.0)
        hmax = dt if (dt_max is None or dt_max <= 0.0) else min(dt, float(dt_max))
        hmin = dt / max_substeps
        t, h, n = 0.0, hmax, 0
        while t < dt * (1.0 - 1e-9) and n < max_substeps:
            h = max(hmin, min(h, dt - t, hmax))
            delta, err = self._ros2_substep(flat_t, flat_p, conc, flat_j, h)
            scale = atol + rtol * conc.abs()
            enorm = (err.abs() / scale).amax().item()          # block-max
            if enorm <= 1.0 or h <= hmin * (1.0 + 1e-9):       # accept (forced at floor)
                conc = (conc + delta).clamp_min(0.0)
                t += h
                n += 1
                fac = 5.0 if enorm <= 1e-30 else min(5.0, max(1.0, 1.4 * enorm ** -0.5))
                h = h * fac
            else:                                              # reject, shrink
                h = max(hmin, h * min(0.9, max(0.2, 1.4 * enorm ** -0.5)))
        self._last_nsub = n

        ds = ((conc - conc0) * self.mw).reshape(*grid, self.nsp).movedim(-1, 0)
        scalar_s.add_(ds)
