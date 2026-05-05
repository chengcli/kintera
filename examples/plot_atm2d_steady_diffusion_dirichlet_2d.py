from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import kintera as kt


torch.set_default_dtype(torch.float64)


def build_case(
    ncol: int,
    nlyr: int,
    *,
    device: torch.device,
) -> tuple[kt.SparseSystemMatrix, torch.Tensor, torch.Tensor]:
    x2f = torch.linspace(0.0, 2.0, ncol + 1, dtype=torch.float64, device=device)
    x1f = torch.linspace(0.0, 1.5, nlyr + 1, dtype=torch.float64, device=device)
    x2v = 0.5 * (x2f[:-1] + x2f[1:])
    x1v = 0.5 * (x1f[:-1] + x1f[1:])
    temp = torch.full((ncol, nlyr), 250.0, dtype=torch.float64, device=device)
    pres = torch.full((ncol, nlyr), 1.0e4, dtype=torch.float64, device=device)
    conc = torch.zeros((ncol, nlyr, 1), dtype=torch.float64, device=device)
    state = kt.AtmState2D(x1f=x1f, x2f=x2f, temperature=temp, pressure=pres, concentration=conc)

    x1_grid = x1v.unsqueeze(0).expand(ncol, nlyr)
    x2_grid = x2v.unsqueeze(1).expand(ncol, nlyr)
    analytic = 0.3 + 0.7 * (x1_grid / x1f[-1]) - 0.4 * (x2_grid / x2f[-1])

    kzz = torch.full((ncol, nlyr), 4.0e-2, dtype=torch.float64, device=device)
    kyy = torch.full((ncol, nlyr), 9.0e-2, dtype=torch.float64, device=device)
    transport = kt.build_transport_matrix(state, kzz, kyy=kyy)
    dt = 0.2
    system = torch.eye(transport.nstate, dtype=torch.float64, device=device) - dt * transport.global_csr.to_dense()
    matrix = kt.SparseSystemMatrix.from_dense(system, ncol=ncol, nlyr=nlyr, nspecies=1)

    rhs_override_mask = torch.zeros((ncol, nlyr, 1), dtype=torch.bool, device=device)
    rhs_override_values = torch.zeros((ncol, nlyr, 1), dtype=torch.float64, device=device)
    row_values: dict[int, float] = {}

    def add_dirichlet_row(icol: int, ilev: int, value: torch.Tensor) -> None:
        row = icol * nlyr + ilev
        row_values[row] = 1.0
        rhs_override_mask[icol, ilev, 0] = True
        rhs_override_values[icol, ilev, 0] = value

    for ilev in range(nlyr):
        add_dirichlet_row(0, ilev, analytic[0, ilev])
        add_dirichlet_row(ncol - 1, ilev, analytic[ncol - 1, ilev])
    for icol in range(ncol):
        add_dirichlet_row(icol, 0, analytic[icol, 0])
        add_dirichlet_row(icol, nlyr - 1, analytic[icol, nlyr - 1])

    rows = torch.tensor(sorted(row_values), dtype=torch.int64, device=device)
    matrix = matrix.replace_rows(
        rows,
        rows.clone(),
        torch.ones(rows.numel(), dtype=torch.float64, device=device),
        rhs_override_mask=rhs_override_mask,
        rhs_override_values=rhs_override_values,
    )
    return matrix, analytic, x1_grid


def march_to_steady_state(
    matrix: kt.SparseSystemMatrix,
    analytic: torch.Tensor,
    *,
    max_steps: int = 800,
    tol: float = 1.0e-11,
) -> tuple[torch.Tensor, list[float]]:
    solution = torch.zeros((matrix.ncol, matrix.nlyr, 1), dtype=matrix.dtype, device=matrix.device)
    solution[:, :, 0] = analytic
    solution[1:-1, 1:-1, 0] = 0.0
    residual_history: list[float] = []
    for _ in range(max_steps):
        next_solution = kt.solve_sparse_system(matrix, solution)
        residual = torch.max(torch.abs(next_solution - solution)).item()
        residual_history.append(residual)
        solution = next_solution
        if residual < tol:
            break
    return solution[:, :, 0], residual_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--ncol", type=int, default=61)
    parser.add_argument("--nlyr", type=int, default=45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    matrix, analytic, x1_grid = build_case(args.ncol, args.nlyr, device=device)
    numerical, residual_history = march_to_steady_state(matrix, analytic)
    abs_error = torch.abs(numerical - analytic)

    outdir = Path("/tmp/kintera_atm2d_plots")
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for ax, field, title in zip(
        axes,
        [analytic.cpu(), numerical.cpu(), abs_error.cpu()],
        ["Analytic", "Implicit steady state", "Absolute error"],
        strict=True,
    ):
        image = ax.imshow(field.T.numpy(), origin="lower", aspect="auto")
        ax.set_xlabel("Column index")
        ax.set_ylabel("Layer index")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.85)
    fig.savefig(outdir / "atm2d_diffusion_2d_dirichlet_fields.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.semilogy(residual_history, color="tab:green")
    ax.set_xlabel("Backward-Euler step")
    ax.set_ylabel("Max update")
    ax.set_title(f"2D Diffusion Convergence ({device.type})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "atm2d_diffusion_2d_dirichlet_convergence.png", dpi=180)
    plt.close(fig)

    print(f"Wrote {outdir / 'atm2d_diffusion_2d_dirichlet_fields.png'}")
    print(f"Wrote {outdir / 'atm2d_diffusion_2d_dirichlet_convergence.png'}")
    print(f"Device: {device.type}")
    print(f"Grid: ncol={args.ncol}, nlyr={args.nlyr}")
    print(f"Max abs error: {abs_error.max().item():.6e}")
    print(f"Final residual: {residual_history[-1]:.6e}")


if __name__ == "__main__":
    main()
