"""Regression: h2diss EOS must handle an N-D field, not just a scalar.

speciate() used to pass T as (spatial...,1) to helpers that reduce the coeff axis via select
-> (spatial...), which OUTER-PRODUCTED T against the field. Invisible at n=1 (every prior test),
fatal for any block >1 cell (snapy MoistMixture cons2prim over a MeshBlock). See kintera commit
'thermo(h2diss): field-shape fix' / AIworkshop ISSUES S5."""
import numpy as np
import pytest
import torch
from kintera import ThermoY, ThermoOptions

torch.set_default_dtype(torch.float64)

YAML = """
reference-state: {Tref: 300.0, Pref: 1.0e5, use-nasa9-cp: true, use-h2-dissociation: true}
species:
- {name: dry, composition: {H: 1.6667, He: 0.16667}, cv_R: 2.5}
"""


@pytest.fixture
def th(tmp_path):
    p = tmp_path / "dry_h2diss.yaml"
    p.write_text(YAML)
    return ThermoY(ThermoOptions.from_yaml(str(p)))


@pytest.mark.parametrize("shape", [(81,), (1, 10, 134), (1, 134, 134), (2, 3, 4)])
def test_vt_to_p_preserves_field_shape(th, shape):
    rho = torch.full(shape, 5.0)
    T = torch.full(shape, 3000.0)
    V = th.compute("DY->V", (rho, torch.zeros((0,) + shape)))
    P = th.compute("VT->P", (V, T))
    assert tuple(P.shape) == shape, "VT->P outer-produced: %s vs %s" % (tuple(P.shape), shape)


def test_block_equals_cellwise(th):
    # a batch must give exactly what the same cells give one at a time (n=1 is the trusted path)
    Tv = torch.tensor([1000.0, 2400.0, 3900.0])
    rho = torch.tensor([0.5, 1.3, 7.1])
    Vb = th.compute("DY->V", (rho, torch.zeros(0, 3)))
    Pb = th.compute("VT->P", (Vb, Tv))
    for i in range(3):
        Vi = th.compute("DY->V", (rho[i:i + 1], torch.zeros(0, 1)))
        Pi = th.compute("VT->P", (Vi, Tv[i:i + 1]))
        assert abs(float(Pb[i]) - float(Pi[0])) <= 1e-9 * float(Pi[0])


def test_cons2prim_roundtrip_on_block(th):
    rho = torch.rand(1, 10, 134) * 5 + 0.1
    T = torch.rand(1, 10, 134) * 3000 + 300
    V = th.compute("DY->V", (rho, torch.zeros(0, 1, 10, 134)))
    U = th.compute("VT->U", (V, T))
    T_vu = th.compute("VU->T", (V, U))           # the inverse snapy uses in cons2prim
    assert float((T_vu - T).abs().max()) < 1e-4
