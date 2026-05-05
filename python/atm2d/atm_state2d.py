from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class SpeciesBoundaryCondition:
    """Boundary condition for one side of the 2D domain.

    Parameters
    ----------
    kind:
        Either a single string applied to every species or a length-`nspecies`
        sequence. Supported kinds are ``"dirichlet"`` and ``"neumann"``.
    value:
        Boundary value(s). Accepted shapes are scalar, ``(nspecies,)``, or
        ``(nedge, nspecies)``.

    Notes
    -----
    Dirichlet values represent the boundary state itself. Neumann values
    represent the outward normal derivative imposed on the boundary row.
    """
    kind: str | Sequence[str]
    value: torch.Tensor | Sequence[float] | float = 0.0

    def kinds(self, nspecies: int) -> tuple[str, ...]:
        """Return one lowercase boundary kind per species."""
        if isinstance(self.kind, str):
            return tuple(self.kind.lower() for _ in range(nspecies))
        if len(self.kind) != nspecies:
            raise ValueError("boundary kind sequence must have length nspecies")
        return tuple(str(item).lower() for item in self.kind)

    def values(
        self,
        nedge: int,
        nspecies: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Broadcast boundary values to ``(nedge, nspecies)``."""
        values = torch.as_tensor(self.value, dtype=dtype, device=device)
        if values.dim() == 0:
            return values.expand(nedge, nspecies)
        if values.dim() == 1:
            if values.numel() == 1:
                return values.expand(nedge, nspecies)
            if values.numel() != nspecies:
                raise ValueError("boundary values must have shape (nspecies,) or (nedge, nspecies)")
            return values.unsqueeze(0).expand(nedge, nspecies)
        if values.shape != (nedge, nspecies):
            raise ValueError("boundary values must have shape (nedge, nspecies)")
        return values


@dataclass
class SpeciesBoundaryConditions2D:
    """Species boundary conditions for the four sides of the domain.

    Boundary precedence at corners is:
    ``left/right`` applied first, then overwritten by ``bottom/top``.
    """
    left: SpeciesBoundaryCondition | None = None
    right: SpeciesBoundaryCondition | None = None
    bottom: SpeciesBoundaryCondition | None = None
    top: SpeciesBoundaryCondition | None = None


@dataclass
class AtmState2D:
    """Cell-centered atmospheric state on a 2D ``(ncol, nlyr)`` grid.

    Parameters
    ----------
    x1f:
        Vertical face coordinates with shape ``(nlyr + 1,)``.
    x2f:
        Horizontal face coordinates with shape ``(ncol + 1,)``.
    temperature:
        Cell-centered temperature with shape ``(ncol, nlyr)``.
    pressure:
        Cell-centered pressure with shape ``(ncol, nlyr)``.
    concentration:
        Cell-centered species concentration with shape
        ``(ncol, nlyr, nspecies)``.
    gravity:
        Gravitational acceleration in CGS units, used by binary diffusion.

    Notes
    -----
    The geometry follows the ``snapy`` convention:
    ``x1`` is vertical, ``x2`` is horizontal, ``f`` denotes face locations,
    and ``v`` denotes cell centers.
    """
    x1f: torch.Tensor
    x2f: torch.Tensor
    temperature: torch.Tensor
    pressure: torch.Tensor
    concentration: torch.Tensor
    gravity: float = 980.665

    def __post_init__(self) -> None:
        self.x1f = _as_1d_tensor(self.x1f, "x1f")
        self.x2f = _as_1d_tensor(self.x2f, "x2f", device=self.x1f.device, dtype=self.x1f.dtype)
        if self.x1f.numel() < 2 or self.x2f.numel() < 2:
            raise ValueError("x1f and x2f must contain at least two face coordinates")

        self.temperature = _as_2d_tensor(
            self.temperature, "temperature", device=self.x1f.device, dtype=self.x1f.dtype
        )
        self.pressure = _as_2d_tensor(
            self.pressure, "pressure", device=self.x1f.device, dtype=self.x1f.dtype
        )
        self.concentration = _as_3d_tensor(
            self.concentration,
            "concentration",
            device=self.x1f.device,
            dtype=self.x1f.dtype,
        )

        self.dx1f = self.x1f[1:] - self.x1f[:-1]
        self.dx2f = self.x2f[1:] - self.x2f[:-1]
        self.x1v = 0.5 * (self.x1f[:-1] + self.x1f[1:])
        self.x2v = 0.5 * (self.x2f[:-1] + self.x2f[1:])
        self._dx1v = self.x1v[1:] - self.x1v[:-1]
        self._dx2v = self.x2v[1:] - self.x2v[:-1]

        if self.temperature.shape != (self.ncol, self.nlyr):
            raise ValueError("temperature must have shape (ncol, nlyr)")
        if self.pressure.shape != (self.ncol, self.nlyr):
            raise ValueError("pressure must have shape (ncol, nlyr)")
        if self.concentration.shape[:2] != (self.ncol, self.nlyr):
            raise ValueError("concentration must have shape (ncol, nlyr, nspecies)")

    @property
    def nlyr(self) -> int:
        """Number of vertical cells."""
        return int(self.x1v.numel())

    @property
    def ncol(self) -> int:
        """Number of horizontal columns."""
        return int(self.x2v.numel())

    @property
    def nspecies(self) -> int:
        """Number of prognostic species."""
        return int(self.concentration.size(-1))

    @property
    def device(self) -> torch.device:
        """Torch device shared by the state tensors."""
        return self.x1f.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype shared by the state tensors."""
        return self.x1f.dtype

    @property
    def dx1v(self) -> torch.Tensor:
        """Spacing between adjacent vertical cell centers."""
        return self._dx1v

    @property
    def dx2v(self) -> torch.Tensor:
        """Spacing between adjacent horizontal cell centers."""
        return self._dx2v

    @property
    def temperature_x1f(self) -> torch.Tensor:
        """Temperature interpolated to interior vertical faces."""
        return 0.5 * (self.temperature[:, 1:] + self.temperature[:, :-1])


def _as_1d_tensor(
    value: torch.Tensor | Sequence[float],
    name: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be 1D")
    return tensor


def _as_2d_tensor(
    value: torch.Tensor | Sequence[Sequence[float]],
    name: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be 2D")
    return tensor


def _as_3d_tensor(
    value: torch.Tensor | Sequence[Sequence[Sequence[float]]],
    name: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.dim() != 3:
        raise ValueError(f"{name} must be 3D")
    return tensor
