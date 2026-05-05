from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import kintera as kt


torch.set_default_dtype(torch.float64)


def build_implicit_system(
    nlyr: int,
    diffusivity: float,
    velocity: float,
    dt: float,
    c_left: float,
    c_right: float,
) -> tuple[kt.SparseSystemMatrix, torch.Tensor]:
    x = torch.linspace(0.0, 1.0, nlyr)
    dx = float(x[1] - x[0])

    operator = torch.zeros((nlyr, nlyr), dtype=torch.float64)
    lower = diffusivity / (dx * dx) + velocity / (2.0 * dx)
    diag = -2.0 * diffusivity / (dx * dx)
    upper = diffusivity / (dx * dx) - velocity / (2.0 * dx)
    for i in range(1, nlyr - 1):
        operator[i, i - 1] = lower
        operator[i, i] = diag
        operator[i, i + 1] = upper

    system = torch.eye(nlyr, dtype=torch.float64) - dt * operator
    system[0] = 0.0
    system[-1] = 0.0
    system[0, 0] = 1.0
    system[-1, -1] = 1.0

    rhs_override_mask = torch.zeros((1, nlyr, 1), dtype=torch.bool)
    rhs_override_values = torch.zeros((1, nlyr, 1), dtype=torch.float64)
    rhs_override_mask[0, 0, 0] = True
    rhs_override_mask[0, -1, 0] = True
    rhs_override_values[0, 0, 0] = c_left
    rhs_override_values[0, -1, 0] = c_right

    matrix = kt.SparseSystemMatrix.from_dense(
        system,
        ncol=1,
        nlyr=nlyr,
        nspecies=1,
        rhs_override_mask=rhs_override_mask,
        rhs_override_values=rhs_override_values,
    )
    return matrix, x


def build_implicit_system_on_device(
    nlyr: int,
    diffusivity: float,
    velocity: float,
    dt: float,
    c_left: float,
    c_right: float,
    *,
    device: torch.device,
) -> tuple[kt.SparseSystemMatrix, torch.Tensor]:
    matrix, x = build_implicit_system(nlyr, diffusivity, velocity, dt, c_left, c_right)
    dense = matrix.global_csr.to_dense().to(device=device)
    moved = kt.SparseSystemMatrix.from_dense(
        dense,
        ncol=matrix.ncol,
        nlyr=matrix.nlyr,
        nspecies=matrix.nspecies,
        rhs_override_mask=matrix.rhs_override_mask.to(device=device),
        rhs_override_values=matrix.rhs_override_values.to(device=device),
    )
    return moved, x.to(device=device)


def analytic_solution(
    x: torch.Tensor,
    diffusivity: float,
    velocity: float,
    c_left: float,
    c_right: float,
) -> torch.Tensor:
    peclet = velocity / diffusivity
    numerator = torch.exp(peclet * x) - 1.0
    denominator = torch.exp(torch.tensor(peclet, dtype=x.dtype, device=x.device)) - 1.0
    return c_left + (c_right - c_left) * numerator / denominator


def march_to_steady_state(
    matrix: kt.SparseSystemMatrix,
    c_left: float,
    c_right: float,
    *,
    max_steps: int = 600,
    tol: float = 1.0e-11,
) -> tuple[torch.Tensor, list[float]]:
    state = torch.zeros((1, matrix.nlyr, 1), dtype=matrix.dtype, device=matrix.device)
    state[0, 0, 0] = c_left
    state[0, -1, 0] = c_right
    residual_history: list[float] = []

    for _ in range(max_steps):
        next_state = kt.solve_sparse_system(matrix, state)
        residual = torch.max(torch.abs(next_state - state)).item()
        residual_history.append(residual)
        state = next_state
        if residual < tol:
            break
    return state[0, :, 0], residual_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nlyr = 161
    diffusivity = 2.0e-2
    velocity = 3.0e-1
    c_left = 1.0
    c_right = 0.2
    dt = 0.5

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    matrix, x = build_implicit_system_on_device(
        nlyr, diffusivity, velocity, dt, c_left, c_right, device=device
    )
    numerical, residual_history = march_to_steady_state(matrix, c_left, c_right)
    analytic = analytic_solution(x, diffusivity, velocity, c_left, c_right)
    abs_error = torch.abs(numerical - analytic)

    outdir = Path("/tmp/kintera_atm2d_plots")
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x_cpu = x.cpu()
    analytic_cpu = analytic.cpu()
    numerical_cpu = numerical.cpu()
    abs_error_cpu = abs_error.cpu()
    ax.plot(x_cpu.numpy(), analytic_cpu.numpy(), label="Analytic", linewidth=2.0)
    ax.plot(x_cpu.numpy(), numerical_cpu.numpy(), "--", label="Implicit steady state", linewidth=2.0)
    ax.set_xlabel("x1")
    ax.set_ylabel("Tracer concentration")
    ax.set_title(f"1D Advection-Diffusion With Dirichlet Boundaries ({device.type})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "atm2d_advection_diffusion_profile.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].plot(abs_error_cpu.numpy(), color="tab:red")
    axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel("Absolute error")
    axes[0].set_title(f"Max error = {abs_error_cpu.max().item():.3e}")
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(residual_history, color="tab:green")
    axes[1].set_xlabel("Backward-Euler step")
    axes[1].set_ylabel("Max update")
    axes[1].set_title("Convergence To Steady State")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "atm2d_advection_diffusion_convergence.png", dpi=180)
    plt.close(fig)

    print(f"Wrote {outdir / 'atm2d_advection_diffusion_profile.png'}")
    print(f"Wrote {outdir / 'atm2d_advection_diffusion_convergence.png'}")
    print(f"Device: {device.type}")
    print(f"Max abs error: {abs_error_cpu.max().item():.6e}")
    print(f"Final residual: {residual_history[-1]:.6e}")


if __name__ == "__main__":
    main()
