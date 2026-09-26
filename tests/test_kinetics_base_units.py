"""The KINETICS-base reader must fill the species table in kg/mol, as YAML cards do.

Only the KINETICS-base master input is loaded (no YAML card). Its element table
is rounded to 0.01 g/mol, hence the rtol of 1e-3.
"""

import os

import pytest
import torch
import kintera
from kintera import KineticsOptions, NucleationOptions, ThermoOptions, ThermoY

torch.set_default_dtype(torch.float64)

KB_MASTER = os.path.join(os.path.dirname(__file__), "kinetics_base", "data",
                         "test_master.inp")
KG_PER_MOL = {"O": 0.0159994, "O2": 0.0319988, "O3": 0.0479982, "N2": 0.0280134}


def _kinetics_base_molar_masses(device):
    op = KineticsOptions.from_kinetics_base(KB_MASTER)
    names = list(kintera.species_names())
    weights = torch.tensor(kintera.species_weights())

    ids = [names.index(sp) for sp in KG_PER_MOL]
    expected = torch.tensor(list(KG_PER_MOL.values()))
    print("species_weights", dict(zip(KG_PER_MOL, weights[ids].tolist())))
    torch.testing.assert_close(weights[ids], expected, rtol=1e-3, atol=0.)

    # a ThermoY built on the KINETICS-base table takes the same molar masses
    th = ThermoY(ThermoOptions().vapor_ids(ids).cref_R([2.5] * len(ids))
                 .uref_R([0.] * len(ids)).sref_R([0.] * len(ids))
                 .nucleation(NucleationOptions()))
    th.to(torch.device(device))
    mu = 1. / th.inv_mu
    print("device", mu.device, "| ThermoY mu", mu.tolist())
    assert op.species()[ids[1]] == "O2"
    torch.testing.assert_close(mu.cpu(), expected, rtol=1e-3, atol=0.)


def test_kinetics_base_molar_masses_in_kg_per_mol():
    _kinetics_base_molar_masses("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_kinetics_base_molar_masses_in_kg_per_mol_cuda():
    _kinetics_base_molar_masses("cuda:0")
