"""Core-engine chemistry source for the Titan atm2d implicit step.

`CoreChemistrySource` is a single `LocalSourceTerm` that replaces the hand-rolled
per-reaction Titan source terms: it evaluates the full thermal + photolysis
chemistry through the compiled core `kintera` engine and returns the cell-local
tendency and Jacobian on the atm2d grid.

- Thermal: mass action + Jacobian assembled by core `Kinetics.forward`/`jacobian`
  (the moses00 `.pun` network via `KineticsOptions.from_kinetics_base_pun`,
  Arrhenius + `KBFalloff`). KB `UPDATE_CHEMB` overrides are injected as external
  rate constants via `extra["kf_override"]` (the core then does their mass
  action / Jacobian too). KBFalloff stays frozen-in-C because `number_density`
  is supplied via `extra`, matching the hand-rolled baseline.
- Photolysis: J·parent source + Jacobian assembled by core `PhotoChem.forward`/
  `jacobian` (its `Photolysis` does the per-bin Σσ·F with the Titan attenuated
  actinic flux). The only Titan-side residue is the per-reaction multiplier /
  min-altitude correction applied to the photo rate.

So the entire chemistry tendency + Jacobian is assembled by the core engine;
Titan supplies only the chemb-override VALUES, the photo multiplier, the
attenuated flux, and the species permutation. Validated to reproduce the
hand-rolled net dC/dt + Jacobian to machine precision
(`diagnostics/stage5_core_*.py`).
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
        self._photochem = self._build_photochem()

    def _build_photochem(self):
        """Build a core PhotoChem over the active photo reactions (J·parent +
        Jacobian done by core); Titan per-reaction multipliers are applied as a
        thin correction in `linearize`."""
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
            if bool(t.parameters.get("suppress_reactant_loss", False)):
                # PhotoChem assembles a fixed net stoich (parent consumed); the
                # suppress-loss variant can't be expressed here. moses00 has none.
                raise NotImplementedError(
                    "suppress_reactant_loss photo term unsupported via core PhotoChem")
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
        popt = kt.PhotolysisOptions()
        popt.reactions(reactions)
        popt.wavelength(list(wl))
        popt.cross_section(xs)
        popt.cross_section_nslabs(nslabs)
        popt.branches(branches)
        popt.branch_names(names)
        popt.quadrature_weights([1.0] * nwave)  # KB per-bin Σσ·F
        pco = kt.PhotoChemOptions()
        pco.photolysis(popt)
        pco.vapor_ids(list(range(self._nsp_core)))  # species() -> core .pun order
        return kt.PhotoChem(pco)

    def _state(self, state):
        return KBTitanState(
            species=self.titan_state.species, fixed_species=self.titan_state.fixed_species,
            varying_species=self.titan_state.varying_species,
            conversion=self.titan_state.conversion, concentration=self.titan_state.concentration,
            density=self.titan_state.density, kzz=self.titan_state.kzz, state=state)

    def _photo_multiplier(self, state, tstate, ncol, nlyr, dtype, device):
        """Per-reaction (ncol,nlyr,nphoto) Titan multiplier × min-altitude mask."""
        nphoto = len(self._photo_meta)
        mult = torch.ones((ncol, nlyr, nphoto), dtype=dtype, device=device)
        alt_km = state.x1v.to(dtype=dtype, device=device).view(1, -1) / 1.0e5
        for c, t in enumerate(self._photo_meta):
            ma = t.parameters.get("min_altitude_km")
            if ma is not None:
                mult[..., c] = mult[..., c] * (alt_km >= float(ma))
            m = _rate_profile_multiplier_on_state_grid(t, tstate, dtype, device)
            if m is not None:
                mult[..., c] = mult[..., c] * m
        return mult

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

        # ---- photolysis: J·parent source + Jacobian assembled by core PhotoChem ----
        if self._photochem is not None:
            tstate = self._state(state)
            flux0 = torch.tensor(self._photo_meta[0].parameters["flux"], dtype=conc.dtype)
            nwave = len(self._photo_meta[0].parameters["wavelengths"])
            # attenuated actinic flux (shared opacity); Titan supplies it
            Fatt = _kinetics_base_pyharp_actinic_flux(
                self._photo_meta[0], tstate, conc, self.sp_idx, flux0, nwave,
                conc.dtype, conc.device)
            if Fatt is not None:
                self._photochem.module("photolysis").update_xs_diss_stacked(T)
                # core PhotoChem: rate_r = J_r · [parent_r]
                prate = self._photochem.forward(T, conc_core, Fatt.movedim(-1, 0))
                # thin Titan correction: per-reaction multiplier / min-altitude
                prate = prate * self._photo_multiplier(
                    state, tstate, state.ncol, state.nlyr, conc.dtype, conc.device)
                pjac = self._photochem.jacobian(conc_core, prate)
                pstoich = self._photochem.stoich.to(dtype=prate.dtype, device=prate.device)
                tend_p = torch.einsum("sr,clr->cls", pstoich, prate)
                jac_p = torch.einsum("sr,clrn->clsn", pstoich, pjac)
                tendency = tendency + tend_p[..., self._ts_to_core]
                jacobian = jacobian + jac_p[..., self._ts_to_core, :][..., self._ts_to_core]

        return LocalSourceLinearization(tendency=tendency, jacobian=jacobian)
