"""Core-engine chemistry source for the Titan atm2d implicit step.

`CoreChemistrySource` is a single `LocalSourceTerm` that replaces the hand-rolled
per-reaction Titan source terms: it evaluates the full thermal + photolysis
chemistry through the compiled core `kintera` engine and returns the cell-local
tendency and Jacobian on the atm2d grid.

- Thermal: rate constants from core `Arrhenius` + `KBFalloff` (the moses00 `.pun`
  network via `KineticsOptions.from_kinetics_base_pun`), with KB `UPDATE_CHEMB`
  overrides applied by :class:`ChembOverrideLayer`. Mass action is assembled with
  the rate constant treated as frozen in concentration (matching the validated
  hand-rolled baseline, which evaluates `k(T, density)` with density frozen).
- Photolysis: J(z) = Σ_λ σ_r(λ) F_att(z,λ) via core `Photolysis` (unit
  quadrature weights → the KB per-bin sum), fed the Titan attenuated actinic
  flux, with the baseline's per-term min-altitude / profile-multiplier /
  suppress-reactant-loss handling.

Validated to reproduce the hand-rolled net dC/dt to machine precision (see
`diagnostics/stage5_core_full_check.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

import kintera as kt

from ..titan.models import KBTitanState
from .core_chemb import build_chemb_override_layer
from .radiation import _kinetics_base_pyharp_actinic_flux
from .source_integration import (
    _photo_rate_profile,
    _rate_profile_multiplier_on_state_grid,
)

try:  # the atm2d package provides the source-term dataclass
    from ...atm2d.sources.protocol import LocalSourceLinearization
except Exception:  # pragma: no cover - fallback when imported standalone
    @dataclass
    class LocalSourceLinearization:  # type: ignore
        tendency: torch.Tensor
        jacobian: torch.Tensor


class CoreChemistrySource:
    """Evaluate Titan thermal + photo chemistry through the core engine."""

    def __init__(self, titan_state: KBTitanState, pun_path: str, source_terms: list[Any]):
        self.titan_state = titan_state
        self.species = list(titan_state.species)
        self.sp_idx = {n: i for i, n in enumerate(self.species)}
        self.nsp = len(self.species)

        # --- thermal: core Kinetics options + rate-constant modules ---
        self.options = kt.KineticsOptions.from_kinetics_base_pun(pun_path)
        self.arrhenius = kt.Arrhenius(self.options.arrhenius())
        self.kbfalloff = kt.KBFalloff(self.options.kb_falloff())
        reactions = self.options.reactions()  # arrhenius(...) then kb_falloff(...)
        self.nrxn = len(reactions)
        self.chemb = build_chemb_override_layer(reactions)

        # reactant/net stoichiometry in ts.species order (skip species not tracked)
        react = torch.zeros((self.nsp, self.nrxn), dtype=torch.float64)
        net = torch.zeros((self.nsp, self.nrxn), dtype=torch.float64)
        self._thermal_ok = torch.ones(self.nrxn, dtype=torch.bool)
        for j, r in enumerate(reactions):
            for name, c in r.reactants().items():
                i = self.sp_idx.get(name)
                if i is None:
                    self._thermal_ok[j] = False
                    continue
                react[i, j] += c
                net[i, j] -= c
            for name, c in r.products().items():
                i = self.sp_idx.get(name)
                if i is None:
                    self._thermal_ok[j] = False
                    continue
                net[i, j] += c
        self.react_st = react
        self.net_st = net

        # --- photolysis: active single-reactant photo terms ---
        self.photo_terms = [
            t for t in source_terms
            if getattr(t, "kind", "") == "pun_photo_rate_reaction"
            and len(t.reactants) == 1
            and t.reactants[0] in self.sp_idx
            and all(p in self.sp_idx for p in t.products)
            and not isinstance(t.parameters.get("secondary_impact"), dict)
            and isinstance(t.parameters.get("wavelengths"), list)
            and isinstance(t.parameters.get("cross_section"), list)
            and isinstance(t.parameters.get("flux"), list)
            and t.parameters["wavelengths"]
        ]
        self._photo = self._build_photo_module()

    def _build_photo_module(self):
        if not self.photo_terms:
            return None
        wl = self.photo_terms[0].parameters["wavelengths"]
        nwave = len(wl)
        reactions, branches, names, xs, nslabs = [], [], [], [], []
        self._photo_meta = []
        for t in self.photo_terms:
            sig = t.parameters["cross_section"]
            if len(sig) != nwave:
                continue
            reactions.append(kt.Reaction(f"{t.reactants[0]} => " + " + ".join(t.products)))
            diss: dict[str, float] = {}
            for p in t.products:
                diss[p] = diss.get(p, 0.0) + 1.0
            branches.append([{t.reactants[0]: 1.0}, diss])
            names.append(["a", "b"])
            for w in range(nwave):
                xs += [sig[w], sig[w]]
            nslabs.append(1)
            self._photo_meta.append(t)
        opt = kt.PhotolysisOptions()
        opt.reactions(reactions)
        opt.wavelength(list(wl))
        opt.cross_section(xs)
        opt.cross_section_nslabs(nslabs)
        opt.branches(branches)
        opt.branch_names(names)
        opt.quadrature_weights([1.0] * nwave)
        return kt.Photolysis(opt)

    def _state(self, state):
        return KBTitanState(
            species=self.titan_state.species, fixed_species=self.titan_state.fixed_species,
            varying_species=self.titan_state.varying_species,
            conversion=self.titan_state.conversion, concentration=self.titan_state.concentration,
            density=self.titan_state.density, kzz=self.titan_state.kzz, state=state)

    def linearize(self, state) -> "LocalSourceLinearization":
        conc = state.concentration
        T = state.temperature
        P = state.pressure
        density = self.titan_state.density
        tendency = torch.zeros_like(conc)
        jacobian = torch.zeros((state.ncol, state.nlyr, self.nsp, self.nsp),
                               dtype=conc.dtype, device=conc.device)

        # ---- thermal (frozen-k mass action) ----
        k = torch.cat([
            self.arrhenius.forward(T, P, conc, {}),
            self.kbfalloff.forward(T, P, conc, {"number_density": density}),
        ], dim=-1)
        k = self.chemb.apply(k, T, density)
        k = k * self._thermal_ok.to(k.dtype)  # drop reactions touching untracked species

        csafe = conc.clamp_min(1e-300)
        logc = torch.log(csafe)
        # clamped mass-action product (uses csafe for any zero reactant)
        prod_clamped = torch.exp(torch.einsum("ij,blj->bli", self.react_st.T, logc))
        # genuine product: zero if any reactant is exactly zero
        has_zero = torch.einsum("ij,blj->bli", self.react_st.T, (conc <= 0).double()) > 0
        rate_f = k * torch.where(has_zero, torch.zeros_like(prod_clamped), prod_clamped)
        tendency = tendency + torch.einsum("ij,blj->bli", self.net_st, rate_f)
        # Jacobian: d(rate_f_j)/dC_m = react_st[m,j] * k_j * prod_excl(m) where the
        # m-exponent is reduced by one. Using the clamped product (NOT the
        # zero-masked rate) with 1/csafe makes this correct even when C_m = 0 and
        # react_st[m,j] = 1 (the csafe `tiny` cancels), matching the hand-rolled
        # baseline's exponent-reduction convention.
        rate_clamped = k * prod_clamped
        inv_csafe = 1.0 / csafe
        drate_dc = (torch.einsum("blj,mj->blmj", rate_clamped, self.react_st)
                    * inv_csafe.unsqueeze(-1))
        jacobian = jacobian + torch.einsum("sj,blmj->blsm", self.net_st, drate_dc)

        # ---- photolysis (first order in parent) ----
        if self._photo is not None:
            sp_idx = self.sp_idx
            tstate = self._state(state)
            flux0 = torch.tensor(self._photo_meta[0].parameters["flux"], dtype=conc.dtype)
            nwave = len(self._photo_meta[0].parameters["wavelengths"])
            Fatt = _kinetics_base_pyharp_actinic_flux(
                self._photo_meta[0], tstate, conc, sp_idx, flux0, nwave, conc.dtype, conc.device)
            if Fatt is not None:
                self._photo.update_xs_diss_stacked(T)
                J = self._photo.forward(T, Fatt.movedim(-1, 0))  # (ncol, nlyr, nphoto)
                alt_km = state.x1v.to(dtype=conc.dtype, device=conc.device).view(1, -1) / 1.0e5
                for c, t in enumerate(self._photo_meta):
                    Jr = J[..., c]
                    ma = t.parameters.get("min_altitude_km")
                    if ma is not None:
                        Jr = Jr * (alt_km >= float(ma))
                    mult = _rate_profile_multiplier_on_state_grid(t, tstate, conc.dtype, conc.device)
                    if mult is not None:
                        Jr = Jr * mult
                    reactant = sp_idx[t.reactants[0]]
                    parent = torch.clamp(conc[:, :, reactant], min=0.0)
                    rate = Jr * parent
                    if not bool(t.parameters.get("suppress_reactant_loss", False)):
                        tendency[:, :, reactant] = tendency[:, :, reactant] - rate
                        jacobian[:, :, reactant, reactant] = jacobian[:, :, reactant, reactant] - Jr
                    pc: dict[int, int] = {}
                    for p in t.products:
                        pc[sp_idx[p]] = pc.get(sp_idx[p], 0) + 1
                    for prod_i, coeff in pc.items():
                        tendency[:, :, prod_i] = tendency[:, :, prod_i] + coeff * rate
                        jacobian[:, :, prod_i, reactant] = jacobian[:, :, prod_i, reactant] + coeff * Jr

        return LocalSourceLinearization(tendency=tendency, jacobian=jacobian)
