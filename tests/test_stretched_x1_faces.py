"""Stretched x1: eddy diffusion and the KINETICS-base face helper take face positions."""

import math

import pytest
import torch

import kintera as kt
from kintera.kinetics_base.titan import altitude_faces_from_kinetics_base_centers_km

L_CM, N, Q = 1.0e7, 64, 1.058

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def geometric_faces(n=N, q=Q, length=L_CM):
    k = torch.arange(n + 1, dtype=torch.float64)
    faces = length * torch.expm1(k * math.log(q)) / math.expm1(n * math.log(q))
    faces[-1] = length
    return faces


def column(faces):
    """y1 = 0.5 + 0.3 cos(pi z / L) as exact cell averages, y2 = 1 - y1"""
    lo, hi = faces[:-1], faces[1:]
    k = math.pi / L_CM
    y1 = 0.5 + 0.3 * (torch.sin(k * hi) - torch.sin(k * lo)) / (k * (hi - lo))
    return torch.stack([y1, 1.0 - y1], -1)


def tendency(y, kzz, dzi, x1f):
    try:
        return kt.diffusion_tendency(y, kzz, dzi, x1f=x1f)
    except TypeError:  # no x1f argument yet: the centre-spacing widths
        return kt.diffusion_tendency(y, kzz, dzi)


def residual(faces, device):
    x1f = faces.to(device)
    x1v = 0.5 * (x1f[1:] + x1f[:-1])
    dzi = x1v[1:] - x1v[:-1]
    kzz = torch.full_like(dzi, 1.0e5)
    t = tendency(column(faces).to(device), kzz, dzi, x1f)
    w = (x1f[1:] - x1f[:-1]).unsqueeze(-1)
    return float(((w * t).sum(0).abs() / (w * t.abs()).sum(0)).max())


@pytest.mark.parametrize("device", DEVICES)
def test_stretched_column_conserves_with_true_widths(device):
    r = residual(geometric_faces(), device)
    print(f"stretched q={Q} N={N} {device}: column residual {r:.3e}")
    assert r < 1.0e-12


@pytest.mark.parametrize("device", DEVICES)
def test_uniform_column_conserves(device):
    r = residual(geometric_faces(q=1.0 + 1.0e-12), device)
    print(f"uniform N={N} {device}: column residual {r:.3e}")
    assert r < 1.0e-12


def test_uniform_faces_match_the_centre_spacing_widths():
    x1f = torch.linspace(0.0, L_CM, N + 1, dtype=torch.float64)
    x1v = 0.5 * (x1f[1:] + x1f[:-1])
    dzi = x1v[1:] - x1v[:-1]
    kzz = torch.full_like(dzi, 1.0e5)
    y = column(x1f)
    t_old = kt.diffusion_tendency(y, kzz, dzi)
    t_new = tendency(y, kzz, dzi, x1f)
    assert torch.allclose(t_new, t_old, rtol=1.0e-12, atol=0.0)


def faces_from_helper(centres_km, faces_km):
    try:
        return altitude_faces_from_kinetics_base_centers_km(centres_km, faces_km=faces_km)
    except TypeError:  # no faces argument yet: faces rebuilt from centres
        return altitude_faces_from_kinetics_base_centers_km(centres_km)


@pytest.mark.parametrize("q", [Q, 1.0])
def test_helper_returns_the_true_faces(q):
    faces_km = (geometric_faces(q=q, length=1500.0) if q != 1.0
                else torch.linspace(0.0, 1500.0, N + 1, dtype=torch.float64)).tolist()
    centres_km = [0.5 * (a + b) for a, b in zip(faces_km[:-1], faces_km[1:])]
    got = faces_from_helper(centres_km, faces_km)
    want = torch.tensor(faces_km) * 1.0e5
    err_km = float((got - want).abs().max()) / 1.0e5
    print(f"helper q={q}: max face error {err_km:.3e} km")
    assert torch.equal(got, want)
