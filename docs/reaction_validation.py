"""Native TP/UV validation against an independent, bracketed scalar extent solve.

Manufactured fits isolate numerical correctness from uncertain physical data.
Every case uses real balanced stoichiometry, explicit phases and an inert gas.
"""
import argparse
import json
import platform
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
import kintera as kin
from reaction_catalogue import COMPOSITION, REACTIONS, source_fingerprint

R = kin.constants.Rgas
T0 = 500.
CASES = {
    'condense': (1., .2, .3),
    'nucleate': (1., 0., .3),
    'evaporate': (.03, .2, 2.),
    'clear': (.03, 0., 2.),
    'zero_h2': (1., 0., 0.),
    'trace_h2': (1., .2, 1.e-12),
    'absent_reactants': (0., .2, 2.),
    'rich_h2': (1., .2, 10.),
}


def specification(key):
    r = REACTIONS[key]
    names = ['He', *r['gas'], r['cloud']+'(s)']
    nu = np.array([0., *[-a for a in r['gas'].values()], float(r['cloud_nu'])])
    ngas = len(names)-1
    # Positive heat capacities and exothermic formation. This is a synthetic
    # caloric model, not measured material thermochemistry.
    cv = np.full(len(names), 2.5)
    u0 = np.zeros(len(names)); u0[-1] = -3000/nu[-1]
    a = -nu[:ngas]; asum = a.sum()
    gamma = float(nu@cv - asum)
    base = np.array([10., *[2.*v if v>0 else .3 for v in r['gas'].values()], .2])
    pres = base[:ngas].sum()*R*T0
    target = base + .3*nu
    logq0 = float(a @ np.log(pres*target[:ngas]/target[:ngas].sum()))
    return names, nu, ngas, cv, u0, gamma, base, pres, logq0


def card(key, max_iter=80, ftol=1e-9):
    names, nu, ngas, cv, u0, gamma, base, pres, logq0 = specification(key)
    lines = [f'reference-state: {{Tref: 0., Pref: 1.e5}}',
             f'dynamics: {{equation-of-state: {{max-iter: {max_iter}, ftol: {ftol}, uv-solver: kkt}}}}', 'species:']
    for i,name in enumerate(names):
        comp = COMPOSITION[name.removesuffix('(s)')]
        lines.append('  - '+json.dumps(dict(name=name, phase='gas' if i<ngas else 'solid', composition=comp, cv_R=float(cv[i]), u0_R=float(u0[i]))))
    def term(i):
        return (f'{abs(nu[i]):g} ' if abs(nu[i]) != 1 else '')+names[i]
    equation = ' + '.join(term(i) for i in range(len(names)) if nu[i]<0)+' <=> '+' + '.join(term(i) for i in range(len(names)) if nu[i]>0)
    lines += ['reactions:', '  - '+json.dumps(dict(equation=equation,type='nucleation', **{'rate-constant': dict(formula='ideal',T3=T0,P3=float(np.exp(logq0)),beta=6.,gamma=gamma)}))]
    return '\n'.join(lines)+'\n'


def initial(key, case):
    *_, base, pres, logq0 = specification(key)
    base = base.copy()
    scale, cloud, h2 = CASES[case]
    for i,(name,a) in enumerate(REACTIONS[key]['gas'].items(),1):
        base[i] = h2 if name=='H2' else base[i]*scale
    base[-1] = cloud
    return base


def residual(key, n, temp, pres=None):
    _,nu,ngas,cv,u0,gamma,base,pressure,l0 = specification(key)
    p = n[:ngas]*R*temp if pres is None else pres*n[:ngas]/n[:ngas].sum()
    mask = nu[:ngas] != 0
    # Evaluation floor only: never modifies inventories.
    logq = -nu[:ngas][mask] @ np.log(np.maximum(p[mask],np.finfo(float).tiny))
    return float(logq-(l0+6*(1-T0/temp)-gamma*np.log(temp/T0)))


def reference(key, n0, mode):
    _,nu,ngas,cv,u0,gamma,base,pres,l0 = specification(key)
    energy = float(n0@(u0+cv*T0))
    def state(x):
        n = n0 + nu*x
        temp = T0 if mode=='TP' else (energy-n@u0)/(n@cv)
        return n,temp
    def f(x):
        n,t = state(x)
        return residual(key,n,t,pres if mode=='TP' else None)
    lo = max(-n0[nu>0]/nu[nu>0]); hi = min(-n0[nu<0]/nu[nu<0])
    # At the evaporation limit the condensed phase can disappear.
    if f(lo) <= 0:
        return state(lo)
    assert f(hi) < 0, (key,mode,lo,hi,f(lo),f(hi))
    for _ in range(100):
        mid = .5*(lo+hi)
        if f(mid)>0: lo=mid
        else: hi=mid
    return state(.5*(lo+hi))


def solve(key, case='condense', mode='TP', dtype='float64', device='cpu', max_iter=80, ftol=None):
    ftol = ftol or (1e-9 if dtype=='float64' else 2e-5)
    names,nu,ngas,cv,u0,gamma,base,pres,l0 = specification(key)
    n0 = initial(key,case)
    ref_n, ref_t = reference(key,n0,mode)
    dt = getattr(torch,dtype)
    def tensor(a): return torch.as_tensor(np.asarray(a),dtype=dt,device=device)
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'reaction.yaml';path.write_text(card(key,max_iter,ftol))
        op=kin.ThermoOptions.from_yaml(str(path))
        th=(kin.ThermoX if mode=='TP' else kin.ThermoY)(op)
        th.to(device=torch.device(device),dtype=dt)
        assert th.options.species()==names, (th.options.species(),names)
        diag=tensor([[0.]])
        if mode=='TP':
            x=tensor([n0/n0.sum()]); t=tensor([T0]); p=tensor([pres])
            th.forward(t,p,x,False,diag)
            # Restore absolute inventory using conserved helium.
            n=x[0].cpu().numpy(); n=n*(n0[0]/n[0]); temp=T0
        else:
            mu=1/th.buffer('inv_mu').cpu().numpy()
            mass=n0*mu; rho=tensor([mass.sum()]); y=tensor((mass[1:]/mass.sum())[:,None])
            ivol=th.compute('DY->V',[rho,y]); energy=th.compute('VT->U',[ivol,tensor([T0])])
            th.forward(rho,energy,y,False,diag)
            ivol=th.compute('DY->V',[rho,y]); temp=float(th.compute('VU->T',[ivol,energy])[0])
            n=(ivol*th.buffer('inv_mu'))[0].cpu().numpy()
        conc = n * (pres/(n[:ngas].sum()*R*temp)) if mode=='TP' else n
        rh=float(kin.relative_humidity(tensor([temp]),tensor([conc]),th.buffer('stoich'),th.options.nucleation(),ngas)[0,0])
    res=residual(key,n,temp,pres if mode=='TP' else None)
    comp=max(0.,res) if n[-1]<1e-6 else abs(res)
    extent=(n[-1]-n0[-1])/nu[-1]
    conservation=float(np.max(np.abs(n-n0-nu*extent))/max(n0.max(),1.))
    energy_error=abs(float(n@(u0+cv*temp)-n0@(u0+cv*T0)))/max(abs(float(n0@(u0+cv*T0))),1.) if mode=='UV' else 0.
    return dict(reaction=key,case=case,mode=mode,dtype=dtype,device=device,budget=max_iter,ftol=ftol,
        diag=float(diag[0,0]),temperature=temp,reference_temperature=ref_t,
        cloud=float(n[-1]),reference_cloud=float(ref_n[-1]),residual=res,complementarity=comp,
        state_error=float(np.max(np.abs(n-ref_n)))/max(n0.max(),1.),
        conservation=conservation,energy_error=energy_error,minimum=float(n.min()),rh=rh)


def check(row):
    eps=2e-6 if row['dtype']=='float64' else 2e-3
    assert row['diag']>=0, row
    for name in ('state_error','conservation','energy_error','complementarity'):
        assert row[name]<eps, (name,row)
    assert row['minimum']>=-eps, row


def coupled(mode='TP', dtype='float64', device='cpu'):
    """Two balanced clouds share H2S/H2; equilibrium constructed independently."""
    names=['He','Mn','Zn','H2S','H2','MnS(s)','ZnS(s)']
    nu=np.array([[0,0],[-1,0],[0,-1],[-1,-1],[1,1],[1,0],[0,1]],dtype=float)
    n0=np.array([10.,2.,2.,4.,.3,.2,.2]); ngas=5
    cv=np.full(7,2.5);u0=np.array([0.,0.,0.,0.,0.,-3000.,-3000.])
    expected=n0+nu@np.array([.3,.4]); energy=n0@(u0+cv*T0)
    target_t=T0 if mode=='TP' else (energy-expected@u0)/(expected@cv)
    pres=n0[:ngas].sum()*R*T0
    p=expected[:ngas]*R*target_t if mode=='UV' else pres*expected[:ngas]/expected[:ngas].sum()
    gamma=nu.T@cv+nu[:ngas].sum(0)
    l0=-nu[:ngas].T@np.log(p)-6*(1-T0/target_t)+gamma*np.log(target_t/T0)
    species=[dict(name=n,phase='gas' if i<ngas else 'solid',composition=COMPOSITION[n.removesuffix('(s)')],cv_R=2.5,u0_R=float(u0[i])) for i,n in enumerate(names)]
    reactions=[dict(equation=f'{m} + H2S <=> {m}S(s) + H2',type='nucleation',**{'rate-constant':dict(formula='ideal',T3=T0,P3=float(np.exp(l0[j])),beta=6.,gamma=float(gamma[j]))}) for j,m in enumerate(('Mn','Zn'))]
    config={'reference-state':dict(Tref=0.,Pref=1e5),'dynamics':{'equation-of-state':{'max-iter':80,'ftol':1e-9 if dtype=='float64' else 2e-5,'uv-solver':'auto'}},'species':species,'reactions':reactions}
    dt=getattr(torch,dtype)
    tensor=lambda a: torch.as_tensor(np.asarray(a),dtype=dt,device=device)
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'coupled.yaml';path.write_text(json.dumps(config))
        th=(kin.ThermoX if mode=='TP' else kin.ThermoY)(kin.ThermoOptions.from_yaml(str(path)))
        th.to(device=torch.device(device),dtype=dt);diag=tensor([[0.]])
        if mode=='TP':
            x=tensor([n0/n0.sum()]);th.forward(tensor([T0]),tensor([pres]),x,False,diag)
            n=x[0].cpu().numpy();n=n*n0[0]/n[0];t=T0
        else:
            mass=n0/th.buffer('inv_mu').cpu().numpy();rho=tensor([mass.sum()]);y=tensor((mass[1:]/mass.sum())[:,None])
            ivol=th.compute('DY->V',[rho,y]);u=th.compute('VT->U',[ivol,tensor([T0])])
            th.forward(rho,u,y,False,diag);ivol=th.compute('DY->V',[rho,y])
            n=(ivol*th.buffer('inv_mu'))[0].cpu().numpy();t=float(th.compute('VU->T',[ivol,u])[0])
    p=n[:ngas]*R*t if mode=='UV' else pres*n[:ngas]/n[:ngas].sum()
    res=-nu[:ngas].T@np.log(p)-(l0+6*(1-T0/t)-gamma*np.log(t/T0))
    row=dict(mode=mode,dtype=dtype,device=device,diag=float(diag[0,0]),state_error=float(np.max(np.abs(n-expected))/10),
        conservation=float(np.max(np.abs(n-n0-nu@(n[-2:]-n0[-2:])))/10),
        energy_error=float(abs(n@(u0+cv*t)-energy)/abs(energy)) if mode=='UV' else 0.,
        complementarity=float(max(abs(res))),minimum=float(n.min()),temperature=t,reference_temperature=float(target_t))
    check(row)
    return row


def native_curves(device='cpu'):
    """Check scalar catalogue against native dispatch, not physical accuracy."""
    from reaction_catalogue import evaluate
    records=[]
    for key,r in REACTIONS.items():
        for c in r['curves']:
            name=c['name'].split(' low T')[0].split(' high T')[0].split(' (legacy')[0]
            if name not in ('h2o_ideal','h2o_bryan','nh3_ideal','h2s_ideal','h2s_antoine','ch4_ideal','so2_antoine','co2_antoine','kcl_lodders','nh3_h2s_lewis','na_h2s_visscher'):continue
            op=kin.NucleationOptions();op.logsvp([name])
            ts=np.linspace(*c['interval'],31)
            # Branch crossover itself belongs to the upper Antoine fit.
            if name=='h2s_antoine' and c['interval'][1]==212.8:ts=ts[:-1]
            t=torch.tensor(ts,dtype=torch.float64,device=device)
            conc=torch.ones((len(ts),2),dtype=t.dtype,device=device)
            stoich=torch.tensor([[-1.],[1.]],dtype=t.dtype,device=device)
            rh=kin.relative_humidity(t,conc,stoich,op,1).squeeze(-1)
            got=(torch.log(R*t)-torch.log(rh)).cpu().numpy()
            error=float(np.max(abs(got-evaluate(c,ts))))
            assert error<1e-11,(name,error)
            records.append(dict(reaction=key,formula=c['name'],device=device,max_log_error=error))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'source/_static/reactions/validation.json')
    parser.add_argument('--reaction',choices=list(REACTIONS))
    parser.add_argument('--cuda',action='store_true')
    args=parser.parse_args()
    devices=['cpu']+(['cuda'] if args.cuda and torch.cuda.is_available() else [])
    rows=[]
    for key in ([args.reaction] if args.reaction else REACTIONS):
        for mode in ('TP','UV'):
            for case in CASES:
                for dtype in ('float64','float32'):
                    for device in devices:
                        row=solve(key,case,mode,dtype,device); check(row);rows.append(row)
        print(key, 'passed', flush=True)
    convergence=[]
    for mode in ('TP','UV'):
        for dtype in ('float64','float32'):
            for budget in (1,2,3,4,6,8,12,20,40,80):
                convergence.append(solve('mns','zero_h2',mode,dtype,max_iter=budget))
    tolerance=[]
    for mode in ('TP','UV'):
        for dtype in ('float64','float32'):
            for ftol in (1e-3,1e-5,1e-7,1e-9):
                tolerance.append(solve('mns','condense',mode,dtype,ftol=ftol))
    root=Path(__file__).resolve().parents[1]
    digest=source_fingerprint()
    coupled_rows=[coupled(mode,dtype,device) for mode in ('TP','UV') for dtype in ('float64','float32') for device in devices]
    fit_rows=[row for device in devices for row in native_curves(device)]
    import matplotlib
    data=dict(coupled=coupled_rows,native_curves=fit_rows,environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,matplotlib=matplotlib.__version__,cuda=torch.version.cuda,devices=devices,gpu=torch.cuda.get_device_name(0) if 'cuda' in devices else None,
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),source_sha256=digest),
        cases=rows,convergence=convergence,tolerance=tolerance)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()
