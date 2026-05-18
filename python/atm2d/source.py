from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from .atm_state2d import AtmState2D
from .matrix import SparseSystemMatrix, add_sparse_system_matrices


@dataclass
class LocalSourceLinearization:
    """Cell-local source tendency and Jacobian on an ``AtmState2D`` grid."""

    tendency: torch.Tensor
    jacobian: torch.Tensor


class LocalSourceTerm(Protocol):
    """Protocol for source terms that can be linearized cell-by-cell."""

    def linearize(self, state: AtmState2D) -> LocalSourceLinearization:
        """Return source tendency and d(source)/d(concentration)."""


RateProvider = Callable[[AtmState2D], torch.Tensor]


def build_source_linearization(
    state: AtmState2D,
    source_terms: list[LocalSourceTerm],
    charge_balance_indices: "tuple[list[int], int] | None" = None,
) -> LocalSourceLinearization:
    """Combine local source-term tendencies and Jacobian blocks.

    When ``charge_balance_indices=(cation_indices, e_index)`` is provided,
    fold the implicit constraint ``E = Σ(cations)`` into the Jacobian by
    propagating ``dF_i/dc_E`` into the cation columns via
    ``dc_E/dc_X+ = 1``::

        For every species row i:
            for every cation column j ∈ cation_indices:
                J[..., i, j] += J[..., i, e_index]

    Without this fold the BE Newton sees ``c_E`` as an independent variable
    and lags it via post-Newton charge-balance reset (Picard iteration),
    which causes the cation cascade to keep growing until E catches up.
    """

    tendency = torch.zeros_like(state.concentration)
    jacobian = torch.zeros(
        (state.ncol, state.nlyr, state.nspecies, state.nspecies),
        dtype=state.dtype,
        device=state.device,
    )
    for term in source_terms:
        linearization = term.linearize(state)
        term_tendency = torch.as_tensor(
            linearization.tendency, dtype=state.dtype, device=state.device
        )
        term_jacobian = torch.as_tensor(
            linearization.jacobian, dtype=state.dtype, device=state.device
        )
        if term_tendency.shape != state.concentration.shape:
            raise ValueError("source tendency must match state concentration shape")
        if term_jacobian.shape != jacobian.shape:
            raise ValueError("source jacobian must have shape (ncol, nlyr, nspecies, nspecies)")
        tendency = tendency + term_tendency
        jacobian = jacobian + term_jacobian

    if charge_balance_indices is not None:
        cation_idx_list, e_index = charge_balance_indices
        if cation_idx_list:
            cation_idx = torch.tensor(
                cation_idx_list, dtype=torch.long, device=jacobian.device
            )
            delta = jacobian[..., e_index].clone()  # (ncol, nlyr, nspecies)
            jacobian = jacobian.clone()
            jacobian[..., cation_idx] = (
                jacobian[..., cation_idx] + delta.unsqueeze(-1)
            )
    return LocalSourceLinearization(tendency=tendency, jacobian=jacobian)


def build_source_global_operator(
    state: AtmState2D,
    source_terms: list[LocalSourceTerm],
) -> SparseSystemMatrix | None:
    """Combine optional non-local source Jacobians into a global operator."""

    matrices: list[SparseSystemMatrix] = []
    for term in source_terms:
        global_operator = getattr(term, "global_operator", None)
        if global_operator is None:
            continue
        matrix = global_operator(state)
        if matrix is not None:
            matrices.append(matrix)
    if not matrices:
        return None
    return add_sparse_system_matrices(*matrices)


@dataclass
class IndexedFirstOrderSource:
    """A local first-order reaction with frozen rate profile."""

    reactant: int
    products: list[int]
    rate: float | torch.Tensor | RateProvider
    suppress_reactant_loss: bool = False

    def linearize(self, state: AtmState2D) -> LocalSourceLinearization:
        tendency = torch.zeros_like(state.concentration)
        jacobian = _empty_jacobian(state)
        rate = _rate_on_grid(self.rate, state)
        parent = torch.clamp(state.concentration[:, :, self.reactant], min=0.0)
        flux = rate * parent
        if not self.suppress_reactant_loss:
            tendency[:, :, self.reactant] = tendency[:, :, self.reactant] - flux
            jacobian[:, :, self.reactant, self.reactant] = (
                jacobian[:, :, self.reactant, self.reactant] - rate
            )
        for product in self.products:
            tendency[:, :, product] = tendency[:, :, product] + flux
            jacobian[:, :, product, self.reactant] = (
                jacobian[:, :, product, self.reactant] + rate
            )
        return LocalSourceLinearization(tendency=tendency, jacobian=jacobian)


@dataclass
class IndexedMassActionSource:
    """A cell-local mass-action reaction with analytic frozen Jacobian."""

    reactants: list[int]
    products: list[int]
    reactant_coefficients: list[int]
    product_coefficients: list[int]
    rate_constant: float | torch.Tensor | RateProvider

    def linearize(self, state: AtmState2D) -> LocalSourceLinearization:
        tendency = torch.zeros_like(state.concentration)
        jacobian = _empty_jacobian(state)
        rate_constant = _rate_on_grid(self.rate_constant, state)
        rate = rate_constant
        for reactant, coeff in zip(self.reactants, self.reactant_coefficients):
            rate = rate * torch.clamp(state.concentration[:, :, reactant], min=0.0) ** coeff
        for product, coeff in zip(self.products, self.product_coefficients):
            tendency[:, :, product] = tendency[:, :, product] + coeff * rate
        for reactant, coeff in zip(self.reactants, self.reactant_coefficients):
            tendency[:, :, reactant] = tendency[:, :, reactant] - coeff * rate

        reactant_values = [
            torch.clamp(state.concentration[:, :, reactant], min=0.0)
            for reactant in self.reactants
        ]
        for wrt_index, (wrt, wrt_coeff) in enumerate(
            zip(self.reactants, self.reactant_coefficients)
        ):
            derivative = rate_constant * wrt_coeff
            for index, (value, coeff) in enumerate(
                zip(reactant_values, self.reactant_coefficients)
            ):
                exponent = coeff - (1 if index == wrt_index else 0)
                if exponent:
                    derivative = derivative * value**exponent
            for product, coeff in zip(self.products, self.product_coefficients):
                jacobian[:, :, product, wrt] = jacobian[:, :, product, wrt] + coeff * derivative
            for reactant, coeff in zip(self.reactants, self.reactant_coefficients):
                jacobian[:, :, reactant, wrt] = jacobian[:, :, reactant, wrt] - coeff * derivative
        return LocalSourceLinearization(tendency=tendency, jacobian=jacobian)


@dataclass
class IndexedReversibleFirstOrderSource:
    """A two-species first-order exchange with frozen forward/reverse rates."""

    left: int
    right: int
    forward_rate: float | torch.Tensor | RateProvider
    reverse_rate: float | torch.Tensor | RateProvider

    def linearize(self, state: AtmState2D) -> LocalSourceLinearization:
        tendency = torch.zeros_like(state.concentration)
        jacobian = _empty_jacobian(state)
        forward = _rate_on_grid(self.forward_rate, state)
        reverse = _rate_on_grid(self.reverse_rate, state)
        left_value = torch.clamp(state.concentration[:, :, self.left], min=0.0)
        right_value = torch.clamp(state.concentration[:, :, self.right], min=0.0)
        flux = forward * left_value - reverse * right_value
        tendency[:, :, self.left] = tendency[:, :, self.left] - flux
        tendency[:, :, self.right] = tendency[:, :, self.right] + flux
        jacobian[:, :, self.left, self.left] = jacobian[:, :, self.left, self.left] - forward
        jacobian[:, :, self.left, self.right] = jacobian[:, :, self.left, self.right] + reverse
        jacobian[:, :, self.right, self.left] = jacobian[:, :, self.right, self.left] + forward
        jacobian[:, :, self.right, self.right] = jacobian[:, :, self.right, self.right] - reverse
        return LocalSourceLinearization(tendency=tendency, jacobian=jacobian)


@dataclass
class IndexedBoundaryFluxSource:
    """A constant flux applied to one vertical boundary cell."""

    species: int
    value: float
    boundary: str

    def linearize(self, state: AtmState2D) -> LocalSourceLinearization:
        tendency = torch.zeros_like(state.concentration)
        jacobian = _empty_jacobian(state)
        dz = _vertical_cell_widths(state)
        if self.boundary == "lower":
            tendency[:, 0, self.species] = tendency[:, 0, self.species] + self.value / dz[0]
        elif self.boundary == "upper":
            tendency[:, -1, self.species] = tendency[:, -1, self.species] - self.value / dz[-1]
        else:
            raise ValueError("boundary must be 'lower' or 'upper'")
        return LocalSourceLinearization(tendency=tendency, jacobian=jacobian)


@dataclass
class IndexedBoundaryVelocitySource:
    """A velocity flux applied to one vertical boundary cell."""

    species: int
    value: float
    boundary: str

    def linearize(self, state: AtmState2D) -> LocalSourceLinearization:
        tendency = torch.zeros_like(state.concentration)
        jacobian = _empty_jacobian(state)
        dz = _vertical_cell_widths(state)
        if self.boundary == "lower":
            rate = self.value / dz[0]
            tendency[:, 0, self.species] = tendency[:, 0, self.species] + rate * state.concentration[:, 0, self.species]
            jacobian[:, 0, self.species, self.species] = jacobian[:, 0, self.species, self.species] + rate
        elif self.boundary == "upper":
            rate = self.value / dz[-1]
            tendency[:, -1, self.species] = tendency[:, -1, self.species] - rate * state.concentration[:, -1, self.species]
            jacobian[:, -1, self.species, self.species] = jacobian[:, -1, self.species, self.species] - rate
        else:
            raise ValueError("boundary must be 'lower' or 'upper'")
        return LocalSourceLinearization(tendency=tendency, jacobian=jacobian)


def _empty_jacobian(state: AtmState2D) -> torch.Tensor:
    return torch.zeros(
        (state.ncol, state.nlyr, state.nspecies, state.nspecies),
        dtype=state.dtype,
        device=state.device,
    )


def _rate_on_grid(rate: float | torch.Tensor | RateProvider, state: AtmState2D) -> torch.Tensor:
    value = rate(state) if callable(rate) else rate
    tensor = torch.as_tensor(value, dtype=state.dtype, device=state.device)
    if tensor.ndim == 0:
        return torch.full((state.ncol, state.nlyr), float(tensor.item()), dtype=state.dtype, device=state.device)
    if tensor.shape != (state.ncol, state.nlyr):
        raise ValueError("rate profile must have shape (ncol, nlyr)")
    return tensor


def _vertical_cell_widths(state: AtmState2D) -> torch.Tensor:
    faces = state.x1f.to(dtype=state.dtype, device=state.device)
    dz = faces[1:] - faces[:-1]
    if torch.any(dz <= 0):
        raise ValueError("vertical cell widths must be positive")
    return dz
