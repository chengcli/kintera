"""Each documented reaction independently exercises signed TP and UV equilibrium."""
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'docs'))
from reaction_validation import REACTIONS, CASES, solve, check, card
from kintera import ThermoOptions, ThermoY

@pytest.mark.parametrize('reaction', REACTIONS)
@pytest.mark.parametrize('mode', ['TP','UV'])
@pytest.mark.parametrize('dtype', ['float64','float32'])
@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_isolated_reaction(reaction,mode,dtype,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    for case in CASES:
        check(solve(reaction,case,mode,dtype,device))


def test_gas_product_rejects_partition(tmp_path):
    path=tmp_path/'mns.yaml';path.write_text(card('mns').replace('uv-solver: kkt','uv-solver: partition'))
    with pytest.raises(RuntimeError,match='partition'):
        th = ThermoY(ThermoOptions.from_yaml(str(path)))
        rho=torch.ones(1,dtype=torch.float64)
        y=torch.zeros(4,1,dtype=torch.float64)
        ivol=th.compute('DY->V',[rho,y])
        th.forward(rho,th.compute('VT->U',[ivol,torch.full_like(rho,500.)]),y)


def test_invalid_phase(tmp_path):
    path=tmp_path/'bad.yaml';path.write_text(card('mns').replace('"phase": "gas"','"phase": "plasma"',1))
    with pytest.raises(RuntimeError,match='phase'):
        ThermoOptions.from_yaml(str(path))

from reaction_validation import coupled, native_curves

@pytest.mark.parametrize('mode',['TP','UV'])
@pytest.mark.parametrize('dtype',['float64','float32'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_h2_and_h2s(mode,dtype,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    coupled(mode,dtype,device)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_documented_native_curves(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    native_curves(device)


def test_rh_requires_explicit_gas_count(tmp_path):
    from kintera import ThermoX, relative_humidity
    path=tmp_path/'mns.yaml';path.write_text(card('mns'))
    th=ThermoX(ThermoOptions.from_yaml(str(path)))
    temp=torch.tensor([500.],dtype=torch.float64);conc=torch.ones(1,5,dtype=torch.float64)
    with pytest.raises(RuntimeError,match='requires ngas'):
        relative_humidity(temp,conc,th.buffer('stoich'),th.options.nucleation())
    rh=relative_humidity(temp,conc,th.buffer('stoich'),th.options.nucleation(),4)
    conc[:,3]*=2
    other=relative_humidity(temp,conc,th.buffer('stoich'),th.options.nucleation(),4)
    torch.testing.assert_close(other,rh/2)


def test_declared_inert_gas_and_phase_reset(tmp_path):
    from kintera import ThermoX
    path=tmp_path/'phases.yaml'
    original=card('mns')
    extra='  - {name: Ar, phase: gas, composition: {Ar: 1}, cv_R: 1.5}\n'
    path.write_text(original.replace('reactions:',extra+'reactions:'))
    th=ThermoX(ThermoOptions.from_yaml(str(path)))
    assert th.options.species()==['He','Mn','H2S','H2','Ar','MnS(s)']
    assert len(th.options.vapor_ids())==5
    # Loading another card must clear the earlier explicit phase metadata.
    path.write_text(card('h2o').replace(', "phase": "gas"','').replace(', "phase": "solid"',''))
    legacy=ThermoX(ThermoOptions.from_yaml(str(path)))
    assert legacy.options.species()==['He','H2O','H2O(s)']
    assert len(legacy.options.vapor_ids())==2


@pytest.mark.parametrize('mode',['TP','UV'])
def test_missing_gas_on_both_sides_is_blocked(tmp_path,mode):
    from kintera import ThermoX
    path=tmp_path/'blocked.yaml';path.write_text(card('mns'))
    op=ThermoOptions.from_yaml(str(path));th=(ThermoX if mode=='TP' else ThermoY)(op)
    n=torch.tensor([[10.,0.,2.,0.,.2]],dtype=torch.float64)
    temp=torch.tensor([500.],dtype=torch.float64);diag=torch.zeros(1,1,dtype=torch.float64)
    if mode=='TP':
        x=n/n.sum();before=x.clone()
        th.forward(temp,torch.full_like(temp,1e5),x,False,diag)
        torch.testing.assert_close(x,before,rtol=0,atol=0)
    else:
        mass=n/th.buffer('inv_mu');rho=mass.sum(-1);y=(mass[:,1:]/rho[:,None]).t().contiguous();before=y.clone()
        ivol=th.compute('DY->V',[rho,y]);energy=th.compute('VT->U',[ivol,temp])
        th.forward(rho,energy,y,False,diag)
        torch.testing.assert_close(y,before,rtol=1e-14,atol=0)
    assert diag[0,0]>=0
