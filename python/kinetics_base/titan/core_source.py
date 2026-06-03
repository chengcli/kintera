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

        # --- thermal: full core Kinetics (mass action + Jacobian in core) ---
        import kintera as _kt
        net = _kt.parse_kinetics_base_pun(pun_path)
        core_species = [s.name for s in sorted(net.species, key=lambda s: s.id)]
        self.options = kt.KineticsOptions.from_kinetics_base_pun(pun_path)
        self.kin = kt.Kinetics(self.options)
        reactions = self.options.reactions()  # arrhenius(...) then kb_falloff(...)
        self.nrxn = len(reactions)
        self.chemb = build_chemb_override_layer(reactions)

        # permutation ts.species <-> core (.pun id-sorted) species order
        core_idx = {n: i for i, n in enumerate(core_species)}
        self._nsp_core = len(core_species)
        self._ts_to_core = [core_idx[n] for n in self.species]
        # core stoichiometry (nsp_core, nrxn); irreversible => nrxn columns
        self._stoich = self.kin.stoich

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

        # ---- thermal: mass action + Jacobian assembled by core Kinetics ----
        # permute concentration into core (.pun id-sorted) species order
        conc_core = torch.zeros(
            (state.ncol, state.nlyr, self._nsp_core), dtype=conc.dtype, device=conc.device
        )
        conc_core[..., self._ts_to_core] = conc

        # external rate-constant override: KB UPDATE_CHEMB on the matched columns,
        # NaN elsewhere (core leaves those reactions' computed k untouched).
        kf_override = torch.full(
            (state.ncol, state.nlyr, self.nrxn), float("nan"),
            dtype=conc.dtype, device=conc.device,
        )
        if len(self.chemb):
            ov = self.chemb.override_rate_constants(T, density)  # (ncol, nlyr, n_override)
            for i, col in enumerate(self.chemb.columns):
                kf_override[..., col] = ov[..., i]

        extra = {"number_density": density, "kf_override": kf_override}
        rate, rc_ddC, rc_ddT = self.kin.forward(T, P, conc_core, extra)
        cvol = torch.ones_like(T)
        jac_rxn = self.kin.jacobian(T, conc_core, cvol, rate, rc_ddC, rc_ddT)
        stoich = self._stoich.to(dtype=rate.dtype, device=rate.device)
        # tendency_core[s] = Σ_r stoich[s,r] * rate_f[r];  jac_core[s,n] = Σ_r stoich[s,r] * jac_rxn[r,n]
        tend_core = torch.einsum("sr,clr->cls", stoich, rate)
        jac_core = torch.einsum("sr,clrn->clsn", stoich, jac_rxn)
        # permute core -> ts.species order
        tendency = tendency + tend_core[..., self._ts_to_core]
        jacobian = jacobian + jac_core[..., self._ts_to_core, :][..., self._ts_to_core]

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
