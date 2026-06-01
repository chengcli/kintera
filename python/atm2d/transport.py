from __future__ import annotations

import os
import torch

from .matrix import SparseSystemMatrix, add_sparse_system_matrices, flatten_state_index
from .atm_state2d import AtmState2D, SpeciesBoundaryCondition, SpeciesBoundaryConditions2D

GAS_CONSTANT_CGS = 8.31446261815324e7

# Transport form selection. ``c_diffusion`` discretises ∂_z(K · ∂_z c)
# (the original kintera form). ``mr_diffusion`` discretises
# ∂_z(K · n_tot · ∂_z(c / n_tot)) (the KB form). The env-var default
# stays ``c_diffusion`` until the MR-form is validated on the Titan
# regression set; flipping the default is a follow-up change.
_TRANSPORT_FORMS = ("c_diffusion", "mr_diffusion")


def _resolve_transport_form(form: str | None) -> str:
    """Resolve the transport form via explicit kwarg → env var → default."""
    if form is None:
        form = os.environ.get("KINTERA_TRANSPORT_FORM", "c_diffusion")
    if form not in _TRANSPORT_FORMS:
        raise ValueError(
            f"unknown transport form {form!r}; expected one of {_TRANSPORT_FORMS}"
        )
    return form


def _face_density_z(
    density: torch.Tensor | None,
    state: AtmState2D,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(n_face, n_center)`` for the vertical FV faces.

    ``n_face`` has shape ``(ncol, nlyr-1)`` and is the arithmetic mean of
    the cell-centered density at adjacent layers.

    ``n_center`` has shape ``(ncol, nlyr)`` and is the cell-centered
    density (broadcast/expanded from ``density`` to match the shape
    expected by per-column scalings).

    Cells with zero density (the grid's "extended" above-atmosphere
    slots) propagate as ``n_face = 0`` on any face that touches them;
    callers must guard the divide ``n_face / n_center`` with the
    matching mask before using the result.
    """
    if density is None:
        raise ValueError(
            "density tensor is required for mr_diffusion transport form"
        )
    d = density
    if d.dim() == 1:
        d = d.view(1, -1)
    if d.shape != (state.ncol, state.nlyr):
        raise ValueError(
            f"density must have shape ({state.ncol}, {state.nlyr}), got {tuple(d.shape)}"
        )
    n_face = 0.5 * (d[:, :-1] + d[:, 1:])
    return n_face, d


def build_eddy_diffusion_matrix(
    state: AtmState2D,
    kzz: torch.Tensor,
    *,
    kyy: torch.Tensor | None = None,
    kzy: torch.Tensor | None = None,
    kyz: torch.Tensor | None = None,
    species_diffusion_scale: torch.Tensor | None = None,
    boundary_conditions: SpeciesBoundaryConditions2D | None = None,
    density: torch.Tensor | None = None,
    form: str | None = None,
) -> SparseSystemMatrix:
    """Assemble the turbulent diffusion operator on the 2D species state.

    Parameters
    ----------
    kzz, kyy:
        Cell-centered scalar eddy diffusivities for the vertical and horizontal
        directions.
    kzy, kyz:
        Optional cell-centered cross-diffusion coefficients. If both are
        provided they are averaged.

    Notes
    -----
    Center-defined coefficients are interpolated to faces internally using
    arithmetic averaging in the interior and constant extrapolation at the
    domain boundaries.
    """
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    vals: list[torch.Tensor] = []
    resolved_form = _resolve_transport_form(form)
    species_scale = _species_diffusion_scale(species_diffusion_scale, state)
    kzz_x1f = _center_to_x1_faces_scalar(kzz, state)
    _assemble_vertical_scalar_diffusion(
        rows, cols, vals, state, kzz_x1f[:, 1:-1], species_scale,
        form=resolved_form, density=density,
    )
    if kyy is not None:
        kyy_x2f = _center_to_x2_faces_scalar(kyy, state)
        _assemble_horizontal_scalar_diffusion(
            rows, cols, vals, state, kyy_x2f[1:-1, :], species_scale
        )
    cross_center = None
    if kzy is not None:
        cross_center = _as_center_scalar(kzy, state)
    if kyz is not None:
        cross_center = _as_center_scalar(kyz, state) if cross_center is None else 0.5 * (
            cross_center + _as_center_scalar(kyz, state)
        )
    if cross_center is not None and state.ncol > 1 and state.nlyr > 1:
        _assemble_cross_diffusion(rows, cols, vals, state, cross_center, species_scale)
    matrix = _matrix_from_triplets(rows, cols, vals, state)
    return _apply_boundary_conditions(state, matrix, boundary_conditions)


def build_binary_diffusion_matrix(
    state: AtmState2D,
    binary_diffusion: torch.Tensor,
    molecular_weights: torch.Tensor,
    *,
    include_gravity: bool = True,
    gas_constant: float = GAS_CONSTANT_CGS,
    species_diffusion_scale: torch.Tensor | None = None,
    boundary_conditions: SpeciesBoundaryConditions2D | None = None,
    density: torch.Tensor | None = None,
    form: str | None = None,
) -> SparseSystemMatrix:
    """Assemble the vertical multicomponent binary-diffusion operator.

    ``binary_diffusion`` is expected at cell centers with shape
    ``(ncol, nlyr, nspecies, nspecies)`` and is interpolated to vertical faces
    internally. When ``include_gravity`` is enabled, the molecular-weight
    separation term is added using the cell-centered thermodynamic state.
    """
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    vals: list[torch.Tensor] = []
    binary_x1f = _center_to_x1_faces_matrix(binary_diffusion, state)
    binary_4d = binary_x1f[:, 1:-1]
    molecular_weights = torch.as_tensor(molecular_weights, dtype=state.dtype, device=state.device)
    if molecular_weights.shape != (state.nspecies,):
        raise ValueError("molecular_weights must have shape (nspecies,)")

    gravity_term = None
    if include_gravity:
        xmid, tmid, dwtm = _interface_thermo(state, molecular_weights)
        gravity_term = 0.5 * torch.einsum("czij,czj->czi", binary_4d, dwtm)
        # tmid is (ncol, nlyr-1); gravity_term is (ncol, nlyr-1, nspecies).
        # Broadcast tmid over the species axis explicitly — pytorch's
        # right-align rule would otherwise misalign (1, nlyr-1) against
        # (1, nlyr-1, nspecies) along the wrong dim.
        # Guard against zero/missing T at extended (above-atmosphere)
        # slots: where tmid <= 0 we force gravity_term to zero rather
        # than divide and produce NaN.
        tmid_safe = torch.where(
            tmid > 0, tmid, torch.ones_like(tmid)
        ).unsqueeze(-1)
        gravity_term = gravity_term * (state.gravity / (tmid_safe * gas_constant))
        gravity_term = torch.where(
            (tmid > 0).unsqueeze(-1).expand_as(gravity_term),
            gravity_term,
            torch.zeros_like(gravity_term),
        )

    resolved_form = _resolve_transport_form(form)
    _assemble_vertical_matrix_diffusion(
        rows, cols, vals, state, binary_4d, gravity_term,
        form=resolved_form, density=density,
    )
    matrix = _matrix_from_triplets(rows, cols, vals, state)
    return _apply_boundary_conditions(state, matrix, boundary_conditions)


def build_transport_matrix(
    state: AtmState2D,
    kzz: torch.Tensor,
    *,
    kyy: torch.Tensor | None = None,
    kzy: torch.Tensor | None = None,
    kyz: torch.Tensor | None = None,
    binary_diffusion: torch.Tensor | None = None,
    molecular_weights: torch.Tensor | None = None,
    include_gravity: bool = True,
    gas_constant: float = GAS_CONSTANT_CGS,
    species_diffusion_scale: torch.Tensor | None = None,
    boundary_conditions: SpeciesBoundaryConditions2D | None = None,
    density: torch.Tensor | None = None,
    form: str | None = None,
) -> SparseSystemMatrix:
    """Assemble the full transport operator from eddy and binary diffusion.

    This is the main entry point for transport-only solves. It combines the
    scalar eddy-diffusion operator with the optional vertical binary-diffusion
    operator, then applies any species boundary conditions to the resulting
    sparse matrix.
    """
    resolved_form = _resolve_transport_form(form)
    matrices = [
        build_eddy_diffusion_matrix(
            state,
            kzz,
            kyy=kyy,
            kzy=kzy,
            kyz=kyz,
            species_diffusion_scale=species_diffusion_scale,
            boundary_conditions=None,
            density=density,
            form=resolved_form,
        )
    ]
    if binary_diffusion is not None:
        if molecular_weights is None:
            raise ValueError("molecular_weights are required with binary_diffusion")
        matrices.append(
            build_binary_diffusion_matrix(
                state,
                binary_diffusion,
                molecular_weights,
                include_gravity=include_gravity,
                gas_constant=gas_constant,
                boundary_conditions=None,
                density=density,
                form=resolved_form,
            )
        )
    transport = add_sparse_system_matrices(*matrices)
    return _apply_boundary_conditions(state, transport, boundary_conditions)


def _assemble_vertical_scalar_diffusion(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    state: AtmState2D,
    kzz: torch.Tensor,
    species_scale: torch.Tensor,
    *,
    form: str = "c_diffusion",
    density: torch.Tensor | None = None,
) -> None:
    diffusion_block = torch.diag(species_scale)
    if form == "c_diffusion":
        for icol in range(state.ncol):
            for ilev in range(state.nlyr - 1):
                block = kzz[icol, ilev] * diffusion_block
                _add_two_cell_block(rows, cols, vals, state, icol, ilev, icol, ilev + 1, block, axis="z")
        return

    # mr_diffusion: per-face flux ∝ K · n_face · (c_R/n_R - c_L/n_L)
    # The block scaling for the left column is K · n_face/n_L, for the
    # right column is K · n_face/n_R. Zero-density cells produce zero
    # flux on any touching face (skip the face entirely).
    n_face, n_center = _face_density_z(density, state)
    for icol in range(state.ncol):
        for ilev in range(state.nlyr - 1):
            nf = n_face[icol, ilev]
            nL = n_center[icol, ilev]
            nR = n_center[icol, ilev + 1]
            if not (nf > 0 and nL > 0 and nR > 0):
                continue
            base = kzz[icol, ilev] * diffusion_block
            left_block = (nf / nL) * base
            right_block = (nf / nR) * base
            _add_two_cell_block(
                rows, cols, vals, state,
                icol, ilev, icol, ilev + 1,
                base,  # ignored when left/right_col_block supplied
                axis="z",
                left_col_block=left_block,
                right_col_block=right_block,
            )


def _assemble_horizontal_scalar_diffusion(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    state: AtmState2D,
    kyy: torch.Tensor,
    species_scale: torch.Tensor,
) -> None:
    diffusion_block = torch.diag(species_scale)
    for icol in range(state.ncol - 1):
        for ilev in range(state.nlyr):
            block = kyy[icol, ilev] * diffusion_block
            _add_two_cell_block(rows, cols, vals, state, icol, ilev, icol + 1, ilev, block, axis="y")


def _assemble_vertical_matrix_diffusion(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    state: AtmState2D,
    binary_diffusion: torch.Tensor,
    gravity_term: torch.Tensor | None,
    *,
    form: str = "c_diffusion",
    density: torch.Tensor | None = None,
) -> None:
    if form == "mr_diffusion":
        n_face, n_center = _face_density_z(density, state)
    else:
        n_face = None
        n_center = None
    for icol in range(state.ncol):
        for ilev in range(state.nlyr - 1):
            block = binary_diffusion[icol, ilev]
            if form == "mr_diffusion":
                assert n_face is not None and n_center is not None
                nf = n_face[icol, ilev]
                nL = n_center[icol, ilev]
                nR = n_center[icol, ilev + 1]
                if not (nf > 0 and nL > 0 and nR > 0):
                    continue
                left_block = (nf / nL) * block
                right_block = (nf / nR) * block
                _add_two_cell_block(
                    rows, cols, vals, state,
                    icol, ilev, icol, ilev + 1,
                    block,
                    axis="z",
                    left_col_block=left_block,
                    right_col_block=right_block,
                )
            else:
                _add_two_cell_block(rows, cols, vals, state, icol, ilev, icol, ilev + 1, block, axis="z")
            if gravity_term is not None:
                # Centered FV for the gravitational-separation flux in
                # mr_diffusion form. The mol_grad term already includes the
                # bath density-gradient correction, so the gravity term
                # adds only the species-specific contribution
                #     J_grav = -K * c * (1/H_i - 1/H_atm)
                #            = -K * c * g * (m_i - m_atm)/(RT)
                # At the face between cells L and L+1, centered FV writes
                #     J_face = -gravity_term * (c[L] + c[L+1])
                # where ``gravity_term = 0.5 * K * (m_i - m_atm) * g/(RT)``
                # (the 0.5 is folded in by build_binary_diffusion_matrix).
                # Divergence -∂J/∂z then gives the four matrix entries
                # below. For HEAVY species (gravity_term > 0) the diagonal
                # at row L is +gravity_term/dx[L], so c[L] (lower cell)
                # grows — heavy species sink into L. For LIGHT species
                # (gravity_term < 0) the signs flip and c[L] drains
                # upward. The previous 2-entry assembly used the opposite
                # sign convention (effectively heavy rising, light sinking)
                # and is not equivalent to either an upwind or a centered
                # scheme; it accidentally gave reasonable L60 ratios but
                # diverged sharply at L70+.
                gravity_diag = torch.diag(gravity_term[icol, ilev])
                _add_block(
                    rows, cols, vals, state,
                    icol, ilev, icol, ilev,
                    gravity_diag / state.dx1f[ilev],
                )
                _add_block(
                    rows, cols, vals, state,
                    icol, ilev, icol, ilev + 1,
                    gravity_diag / state.dx1f[ilev],
                )
                _add_block(
                    rows, cols, vals, state,
                    icol, ilev + 1, icol, ilev,
                    -gravity_diag / state.dx1f[ilev + 1],
                )
                _add_block(
                    rows, cols, vals, state,
                    icol, ilev + 1, icol, ilev + 1,
                    -gravity_diag / state.dx1f[ilev + 1],
                )


def _assemble_cross_diffusion(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    state: AtmState2D,
    kzy_center: torch.Tensor,
    species_scale: torch.Tensor,
) -> None:
    diffusion_block = torch.diag(species_scale)
    for icol in range(state.ncol):
        for ilev in range(state.nlyr - 1):
            coeff = 0.5 * (kzy_center[icol, ilev] + kzy_center[icol, ilev + 1])
            grad_weights = _average_gradient_weights_y(state, icol, ilev, ilev + 1)
            for src_col, src_lyr, weight in grad_weights:
                block = coeff * weight * diffusion_block
                _add_scaled_flux_pair(rows, cols, vals, state, icol, ilev, icol, ilev + 1, src_col, src_lyr, block, axis="z")

    for icol in range(state.ncol - 1):
        for ilev in range(state.nlyr):
            coeff = 0.5 * (kzy_center[icol, ilev] + kzy_center[icol + 1, ilev])
            grad_weights = _average_gradient_weights_z(state, icol, icol + 1, ilev)
            for src_col, src_lyr, weight in grad_weights:
                block = coeff * weight * diffusion_block
                _add_scaled_flux_pair(rows, cols, vals, state, icol, ilev, icol + 1, ilev, src_col, src_lyr, block, axis="y")


def _add_two_cell_block(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    state: AtmState2D,
    left_col: int,
    left_lyr: int,
    right_col: int,
    right_lyr: int,
    block: torch.Tensor,
    *,
    axis: str,
    left_col_block: torch.Tensor | None = None,
    right_col_block: torch.Tensor | None = None,
) -> None:
    """Assemble the four matrix entries for one finite-volume face.

    Standard usage (concentration-form diffusion) passes a single
    ``block`` tensor — it is used for both the left-column and
    right-column contributions, producing the symmetric Laplacian

        M[L,L] -= block / (Δz_L · Δv_face)
        M[L,R] += block / (Δz_L · Δv_face)
        M[R,L] += block / (Δz_R · Δv_face)
        M[R,R] -= block / (Δz_R · Δv_face)

    For mixing-ratio-form diffusion the LEFT-COLUMN entries (``M[*,L]``)
    need a different scaling from the RIGHT-COLUMN entries (``M[*,R]``)
    because the flux operates on ``c / n_tot`` and the per-column
    factors ``n_face / n_L`` and ``n_face / n_R`` are different. To
    enable that, callers can pass ``left_col_block`` and
    ``right_col_block`` instead of (or in addition to) ``block``.
    """
    if axis == "z":
        left_scale = 1.0 / (state.dx1f[left_lyr] * state.dx1v[left_lyr])
        right_scale = 1.0 / (state.dx1f[right_lyr] * state.dx1v[left_lyr])
    elif axis == "y":
        left_scale = 1.0 / (state.dx2f[left_col] * state.dx2v[left_col])
        right_scale = 1.0 / (state.dx2f[right_col] * state.dx2v[left_col])
    else:
        raise ValueError("axis must be 'y' or 'z'")

    if left_col_block is None:
        left_col_block = block
    if right_col_block is None:
        right_col_block = block

    _add_block(rows, cols, vals, state, left_col, left_lyr, left_col, left_lyr, -left_scale * left_col_block)
    _add_block(rows, cols, vals, state, left_col, left_lyr, right_col, right_lyr, left_scale * right_col_block)
    _add_block(rows, cols, vals, state, right_col, right_lyr, left_col, left_lyr, right_scale * left_col_block)
    _add_block(rows, cols, vals, state, right_col, right_lyr, right_col, right_lyr, -right_scale * right_col_block)


def _add_scaled_flux_pair(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    state: AtmState2D,
    left_col: int,
    left_lyr: int,
    right_col: int,
    right_lyr: int,
    src_col: int,
    src_lyr: int,
    block: torch.Tensor,
    *,
    axis: str,
) -> None:
    if axis == "z":
        left_scale = 1.0 / state.dx1f[left_lyr]
        right_scale = -1.0 / state.dx1f[right_lyr]
    elif axis == "y":
        left_scale = 1.0 / state.dx2f[left_col]
        right_scale = -1.0 / state.dx2f[right_col]
    else:
        raise ValueError("axis must be 'y' or 'z'")
    _add_block(rows, cols, vals, state, left_col, left_lyr, src_col, src_lyr, left_scale * block)
    _add_block(rows, cols, vals, state, right_col, right_lyr, src_col, src_lyr, right_scale * block)


def _add_block(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    state: AtmState2D,
    row_col: int,
    row_lyr: int,
    col_col: int,
    col_lyr: int,
    block: torch.Tensor,
) -> None:
    nz = block.nonzero(as_tuple=False)
    if nz.numel() == 0:
        return
    row_base = flatten_state_index(
        torch.full((nz.size(0),), row_col, dtype=torch.int64, device=state.device),
        torch.full((nz.size(0),), row_lyr, dtype=torch.int64, device=state.device),
        nz[:, 0].to(device=state.device, dtype=torch.int64),
        state.nlyr,
        state.nspecies,
    )
    col_base = flatten_state_index(
        torch.full((nz.size(0),), col_col, dtype=torch.int64, device=state.device),
        torch.full((nz.size(0),), col_lyr, dtype=torch.int64, device=state.device),
        nz[:, 1].to(device=state.device, dtype=torch.int64),
        state.nlyr,
        state.nspecies,
    )
    rows.append(row_base)
    cols.append(col_base)
    vals.append(block[nz[:, 0], nz[:, 1]])


def _matrix_from_triplets(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    state: AtmState2D,
) -> SparseSystemMatrix:
    if rows:
        indices = torch.stack([torch.cat(rows), torch.cat(cols)])
        values = torch.cat(vals)
    else:
        indices = torch.empty((2, 0), dtype=torch.int64, device=state.device)
        values = torch.empty((0,), dtype=state.dtype, device=state.device)
    return SparseSystemMatrix.from_coo(
        indices,
        values,
        ncol=state.ncol,
        nlyr=state.nlyr,
        nspecies=state.nspecies,
        device=state.device,
        dtype=state.dtype,
    )


def _apply_boundary_conditions(
    state: AtmState2D,
    matrix: SparseSystemMatrix,
    boundary_conditions: SpeciesBoundaryConditions2D | None,
) -> SparseSystemMatrix:
    if boundary_conditions is None:
        return matrix

    override_mask = torch.zeros(
        (state.ncol, state.nlyr, state.nspecies), dtype=torch.bool, device=state.device
    )
    override_values = torch.zeros(
        (state.ncol, state.nlyr, state.nspecies), dtype=state.dtype, device=state.device
    )

    row_entries: dict[int, tuple[list[int], list[float]]] = {}

    if boundary_conditions.left is not None:
        _apply_single_boundary(
            row_entries, override_mask, override_values, state, boundary_conditions.left, side="left"
        )
    if boundary_conditions.right is not None:
        _apply_single_boundary(
            row_entries, override_mask, override_values, state, boundary_conditions.right, side="right"
        )
    if boundary_conditions.bottom is not None:
        _apply_single_boundary(
            row_entries, override_mask, override_values, state, boundary_conditions.bottom, side="bottom"
        )
    if boundary_conditions.top is not None:
        _apply_single_boundary(
            row_entries, override_mask, override_values, state, boundary_conditions.top, side="top"
        )

    if not row_entries:
        return matrix
    row_ids: list[int] = []
    col_ids: list[int] = []
    values: list[float] = []
    for row_index in sorted(row_entries):
        cols, row_values = row_entries[row_index]
        row_ids.extend([row_index] * len(cols))
        col_ids.extend(cols)
        values.extend(row_values)
    return matrix.replace_rows(
        torch.tensor(row_ids, dtype=torch.int64, device=state.device),
        torch.tensor(col_ids, dtype=torch.int64, device=state.device),
        torch.tensor(values, dtype=state.dtype, device=state.device),
        rhs_override_mask=override_mask,
        rhs_override_values=override_values,
    )


def _apply_single_boundary(
    row_entries: dict[int, tuple[list[int], list[float]]],
    override_mask: torch.Tensor,
    override_values: torch.Tensor,
    state: AtmState2D,
    condition: SpeciesBoundaryCondition,
    *,
    side: str,
) -> None:
    kinds = condition.kinds(state.nspecies)
    if side in {"left", "right"}:
        nedge = state.nlyr
    else:
        nedge = state.ncol
    bc_values = condition.values(nedge, state.nspecies, dtype=state.dtype, device=state.device)

    for edge_idx in range(nedge):
        if side == "left":
            row_col, row_lyr = 0, edge_idx
            nei_col, nei_lyr = 1, edge_idx
            spacing = state.dx2v[0]
        elif side == "right":
            row_col, row_lyr = state.ncol - 1, edge_idx
            nei_col, nei_lyr = state.ncol - 2, edge_idx
            spacing = state.dx2v[-1]
        elif side == "bottom":
            row_col, row_lyr = edge_idx, 0
            nei_col, nei_lyr = edge_idx, 1
            spacing = state.dx1v[0]
        elif side == "top":
            row_col, row_lyr = edge_idx, state.nlyr - 1
            nei_col, nei_lyr = edge_idx, state.nlyr - 2
            spacing = state.dx1v[-1]
        else:
            raise ValueError("unknown boundary side")

        for ispecies, kind in enumerate(kinds):
            row_index = int(flatten_state_index(row_col, row_lyr, ispecies, state.nlyr, state.nspecies).item())
            kind = kind.lower()
            if kind == "none":
                continue
            override_mask[row_col, row_lyr, ispecies] = True
            override_values[row_col, row_lyr, ispecies] = bc_values[edge_idx, ispecies]
            if kind == "dirichlet":
                row_entries[row_index] = ([row_index], [1.0])
            elif kind == "neumann":
                nei_index = int(
                    flatten_state_index(nei_col, nei_lyr, ispecies, state.nlyr, state.nspecies).item()
                )
                if side in {"left", "bottom"}:
                    row_entries[row_index] = (
                        [row_index, nei_index],
                        [-1.0 / float(spacing), 1.0 / float(spacing)],
                    )
                else:
                    row_entries[row_index] = (
                        [nei_index, row_index],
                        [-1.0 / float(spacing), 1.0 / float(spacing)],
                    )
            else:
                raise ValueError("boundary condition kind must be 'none', 'dirichlet', or 'neumann'")


def _average_gradient_weights_y(
    state: AtmState2D,
    col_idx: int,
    lower_lyr: int,
    upper_lyr: int,
) -> list[tuple[int, int, torch.Tensor]]:
    weights = _cell_gradient_weights_y(state, col_idx, lower_lyr)
    other = _cell_gradient_weights_y(state, col_idx, upper_lyr)
    merged: dict[tuple[int, int], torch.Tensor] = {}
    for src_col, src_lyr, weight in weights + other:
        key = (src_col, src_lyr)
        merged[key] = merged.get(key, torch.zeros((), dtype=state.dtype, device=state.device)) + 0.5 * weight
    return [(src_col, src_lyr, weight) for (src_col, src_lyr), weight in merged.items()]


def _average_gradient_weights_z(
    state: AtmState2D,
    left_col: int,
    right_col: int,
    lyr_idx: int,
) -> list[tuple[int, int, torch.Tensor]]:
    weights = _cell_gradient_weights_z(state, left_col, lyr_idx)
    other = _cell_gradient_weights_z(state, right_col, lyr_idx)
    merged: dict[tuple[int, int], torch.Tensor] = {}
    for src_col, src_lyr, weight in weights + other:
        key = (src_col, src_lyr)
        merged[key] = merged.get(key, torch.zeros((), dtype=state.dtype, device=state.device)) + 0.5 * weight
    return [(src_col, src_lyr, weight) for (src_col, src_lyr), weight in merged.items()]


def _cell_gradient_weights_y(
    state: AtmState2D,
    col_idx: int,
    lyr_idx: int,
) -> list[tuple[int, int, torch.Tensor]]:
    if state.ncol < 2:
        return []
    if col_idx == 0:
        denom = state.dx2v[0]
        return [(0, lyr_idx, -1.0 / denom), (1, lyr_idx, 1.0 / denom)]
    if col_idx == state.ncol - 1:
        denom = state.dx2v[-1]
        return [(state.ncol - 2, lyr_idx, -1.0 / denom), (state.ncol - 1, lyr_idx, 1.0 / denom)]
    denom = state.x2v[col_idx + 1] - state.x2v[col_idx - 1]
    return [(col_idx - 1, lyr_idx, -1.0 / denom), (col_idx + 1, lyr_idx, 1.0 / denom)]


def _cell_gradient_weights_z(
    state: AtmState2D,
    col_idx: int,
    lyr_idx: int,
) -> list[tuple[int, int, torch.Tensor]]:
    if state.nlyr < 2:
        return []
    if lyr_idx == 0:
        denom = state.dx1v[0]
        return [(col_idx, 0, -1.0 / denom), (col_idx, 1, 1.0 / denom)]
    if lyr_idx == state.nlyr - 1:
        denom = state.dx1v[-1]
        return [(col_idx, state.nlyr - 2, -1.0 / denom), (col_idx, state.nlyr - 1, 1.0 / denom)]
    denom = state.x1v[lyr_idx + 1] - state.x1v[lyr_idx - 1]
    return [(col_idx, lyr_idx - 1, -1.0 / denom), (col_idx, lyr_idx + 1, 1.0 / denom)]


def _as_center_scalar(value: torch.Tensor, state: AtmState2D) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=state.dtype, device=state.device)
    if tensor.dim() == 0:
        return tensor.expand(state.ncol, state.nlyr)
    if tensor.shape != (state.ncol, state.nlyr):
        raise ValueError("centered tensor must have shape (ncol, nlyr)")
    return tensor


def _species_diffusion_scale(
    value: torch.Tensor | None,
    state: AtmState2D,
) -> torch.Tensor:
    if value is None:
        return torch.ones(state.nspecies, dtype=state.dtype, device=state.device)
    tensor = torch.as_tensor(value, dtype=state.dtype, device=state.device)
    if tensor.shape != (state.nspecies,):
        raise ValueError("species_diffusion_scale must have shape (nspecies,)")
    return tensor


def _center_to_x1_faces_scalar(value: torch.Tensor, state: AtmState2D) -> torch.Tensor:
    center = _as_center_scalar(value, state)
    faces = torch.empty((state.ncol, state.nlyr + 1), dtype=state.dtype, device=state.device)
    faces[:, 0] = center[:, 0]
    faces[:, -1] = center[:, -1]
    if state.nlyr > 1:
        faces[:, 1:-1] = 0.5 * (center[:, :-1] + center[:, 1:])
    return faces


def _center_to_x2_faces_scalar(value: torch.Tensor, state: AtmState2D) -> torch.Tensor:
    center = _as_center_scalar(value, state)
    faces = torch.empty((state.ncol + 1, state.nlyr), dtype=state.dtype, device=state.device)
    faces[0, :] = center[0, :]
    faces[-1, :] = center[-1, :]
    if state.ncol > 1:
        faces[1:-1, :] = 0.5 * (center[:-1, :] + center[1:, :])
    return faces


def _as_center_matrix(value: torch.Tensor, state: AtmState2D) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=state.dtype, device=state.device)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0).expand(state.ncol, state.nlyr, -1, -1)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0).expand(state.ncol, -1, -1, -1)
    if tensor.shape != (state.ncol, state.nlyr, state.nspecies, state.nspecies):
        raise ValueError(
            "centered matrix tensor must have shape (ncol, nlyr, nspecies, nspecies)"
        )
    return tensor


def _center_to_x1_faces_matrix(value: torch.Tensor, state: AtmState2D) -> torch.Tensor:
    center = _as_center_matrix(value, state)
    faces = torch.empty(
        (state.ncol, state.nlyr + 1, state.nspecies, state.nspecies),
        dtype=state.dtype,
        device=state.device,
    )
    faces[:, 0] = center[:, 0]
    faces[:, -1] = center[:, -1]
    if state.nlyr > 1:
        faces[:, 1:-1] = 0.5 * (center[:, :-1] + center[:, 1:])
    return faces


def _interface_thermo(
    state: AtmState2D,
    molecular_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = _mole_fraction(state.concentration)
    xmid = 0.5 * (x[:, :-1] + x[:, 1:])
    tmid = 0.5 * (state.temperature[:, :-1] + state.temperature[:, 1:])
    wtm_mid = torch.einsum("czi,i->cz", xmid, molecular_weights)
    dwtm = molecular_weights.view(1, 1, -1) - wtm_mid.unsqueeze(-1)
    return xmid, tmid, dwtm


def _mole_fraction(concentration: torch.Tensor) -> torch.Tensor:
    # clamp_min must stay above float32's smallest positive normal (~1.18e-38).
    # The original 1e-300 underflows to 0 in float32, causing 0/0 NaNs at
    # extended (above-atmosphere) altitude slots where the species sum is
    # zero. 1e-30 is well above the float32 underflow but small enough to
    # behave like "infinity in the denominator" → mole fraction = 0.
    return concentration / concentration.sum(dim=-1, keepdim=True).clamp_min(1.0e-30)
