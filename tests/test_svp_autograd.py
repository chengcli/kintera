"""SVP backward must use the configuration captured by its forward pass."""

import math

import pytest
import torch

import kintera as kt


RGAS = 8.31446
TEMP = 300.0
PRES = torch.tensor([1.0e5], dtype=torch.float64)
CONC = torch.tensor([[0.0, 1.0e-3]], dtype=torch.float64)
STOICH = {"stoich": torch.tensor([[1.0], [-1.0]], dtype=torch.float64)}
DIFF_C = 2.0e-5
VM = 18.0e-6
DIAMETER = 1.0e-4
KAPPA = 12.0 * DIFF_C * VM / DIAMETER**2

CARD = """reference-state: {Tref: 300., Pref: 1.e5}
species:
- {name: H2O, composition: {H: 2, O: 1}, cv_R: 2.5}
- {name: "H2O(l,p)", composition: {H: 2, O: 1}, cv_R: 7.5}
reactions:
- equation: H2O(l,p) => H2O
  type: evaporation
  rate-constant: {FORMULA, diff_c: 2.e-5, diff_T: 0., diff_P: 0.,
                  vm: 18.e-6, diameter: 1.e-4}
"""


@pytest.fixture(autouse=True)
def _register_species():
    kt.set_species_names(["H2O", "H2O(l,p)"])
    kt.set_species_weights([18.0e-3, 18.0e-3])


def _module(tmp_path, name, formula):
    card = tmp_path / f"{name}.yaml"
    card.write_text(CARD.replace("FORMULA", formula))
    owner = kt.KineticsOptions.from_yaml(str(card))
    return kt.Evaporation(owner.evaporation())


def _rate(module, temp):
    return module.forward(temp, PRES, CONC, STOICH)


def _interleaved_gradient(first, second):
    temp = torch.tensor([TEMP], dtype=torch.float64, requires_grad=True)
    rate = _rate(first, temp)
    assert rate.requires_grad
    _rate(second, torch.tensor([TEMP], dtype=torch.float64))
    rate.sum().backward()
    return rate.item(), temp.grad.item()


def test_inline_backward_uses_its_forward_time_parameters(tmp_path):
    first = _module(
        tmp_path,
        "antoine-100",
        "formula: antoine, A: 1., B: 100., C: 0.",
    )
    second = _module(
        tmp_path,
        "antoine-200",
        "formula: antoine, A: 1., B: 200., C: 0.",
    )

    rate, gradient = _interleaved_gradient(first, second)
    expected_rate = KAPPA * (1.0e5 * 10.0 ** (1.0 - 100.0 / TEMP)) / (
        RGAS * TEMP
    )
    expected_gradient = expected_rate * (
        math.log(10.0) * 100.0 / TEMP**2 - 1.0 / TEMP
    )

    assert rate == pytest.approx(expected_rate, rel=1.0e-12)
    assert gradient == pytest.approx(expected_gradient, rel=1.0e-12)


def test_named_backward_uses_its_forward_time_formula(tmp_path):
    first = _module(tmp_path, "water", "formula: h2o_ideal")
    second = _module(tmp_path, "ammonia", "formula: nh3_ideal")
    step = 1.0e-3
    plus = _rate(first, torch.tensor([TEMP + step], dtype=torch.float64)).item()
    minus = _rate(first, torch.tensor([TEMP - step], dtype=torch.float64)).item()
    expected_gradient = (plus - minus) / (2.0 * step)

    _, gradient = _interleaved_gradient(first, second)

    assert gradient == pytest.approx(expected_gradient, rel=2.0e-8)
