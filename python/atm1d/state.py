from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ColumnState1D:
    z: torch.Tensor
    temperature: torch.Tensor
    pressure: torch.Tensor
    concentration: torch.Tensor
    gravity: float = 980.665

    def __post_init__(self) -> None:
        self.z = _as_1d_tensor(self.z, "z")
        self.temperature = _as_1d_tensor(
            self.temperature, "temperature", device=self.z.device, dtype=self.z.dtype
        )
        self.pressure = _as_1d_tensor(
            self.pressure, "pressure", device=self.z.device, dtype=self.z.dtype
        )
        self.concentration = _as_2d_tensor(
            self.concentration,
            "concentration",
            device=self.z.device,
            dtype=self.z.dtype,
        )
        nz = self.z.numel()
        if self.temperature.numel() != nz:
            raise ValueError("temperature must have shape (nz,)")
        if self.pressure.numel() != nz:
            raise ValueError("pressure must have shape (nz,)")
        if self.concentration.size(0) != nz:
            raise ValueError("concentration must have shape (nz, nspecies)")

    @property
    def nz(self) -> int:
        return int(self.z.numel())

    @property
    def nspecies(self) -> int:
        return int(self.concentration.size(1))

    @property
    def device(self) -> torch.device:
        return self.z.device

    @property
    def dtype(self) -> torch.dtype:
        return self.z.dtype


def _as_1d_tensor(
    value: torch.Tensor | list[float] | tuple[float, ...],
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
    value: torch.Tensor | list[list[float]],
    name: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be 2D")
    return tensor
