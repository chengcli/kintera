#!/usr/bin/env python3
"""Generate offline reaction RST, CSV tables and deterministic comparison SVGs.

Native results must first be recorded with reaction_validation.py. --check
regenerates in a temporary directory and fails if checked-in artifacts differ.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reaction_catalogue import REACTIONS, evaluate, source_fingerprint

HERE=Path(__file__).resolve().parent
DEFAULT=HERE/'source'
plt.rcParams.update({'svg.hashsalt':'kintera-reactions','font.size':9})


def save(fig,path):
    fig.savefig(path,metadata={'Date':None},bbox_inches='tight')
    # Matplotlib emits trailing blanks in SVG path attributes. Newlines retain
    # the required separators while keeping generated diffs whitespace-clean.
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def table(headers,rows):
    out='.. list-table::\n   :header-rows: 1\n\n'
    for row in [headers,*rows]:
        out+='   * - '+'\n     - '.join(str(x) for x in row)+'\n'
    return out+'\n'


def equation(r):
    term=lambda name,n: (str(n)+' ' if n!=1 else '')+name
    lhs=' + '.join(term(k,v)+'(g)' for k,v in r['gas'].items() if v>0)
    rhs=term(r['cloud'],r['cloud_nu'])+'(condensed)'
    rhs+=''.join(' + '+term(k,-v)+'(g)' for k,v in r['gas'].items() if v<0)
    return lhs+' <=> '+rhs


def reaction(key,output,data):
    r=REACTIONS[key];assets=output/'_static/reactions';pages=output/'reactions'
    assets.mkdir(parents=True,exist_ok=True);pages.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(2,1,figsize=(7.5,6),sharex=True)
    primary=next((c for c in r['curves'] if 'NIST' in c['status']),r['curves'][0])
    rows=[]
    for c in r['curves']:
        ts=np.linspace(*c['interval'],180)
        ys=evaluate(c,ts)
        ax[0].plot(ts,ys,label=c['name'])
        mask=(ts>=primary['interval'][0])&(ts<=primary['interval'][1])
        if mask.any(): ax[1].plot(ts[mask],(ys[mask]-evaluate(primary,ts[mask]))/np.log(10),label=c['name'])
        for t in np.linspace(*c['interval'],3):
            rows.append([c['name'],f'{t:.6g}',f'{float(evaluate(c,t)):.10g}'])
    ax[0].set_ylabel('ln Q (numerical Pa pressures)');ax[0].legend(fontsize=7)
    ax[1].set_ylabel('log10(Q / reference Q)');ax[1].set_xlabel('Temperature [K]')
    ax[1].axhline(0,color='0.5',linewidth=.5)
    for a in ax:a.grid(alpha=.2)
    fig.suptitle(r['title']+' — coefficient comparison')
    save(fig,assets/(key+'-comparison.svg'))
    with (assets/(key+'-values.csv')).open('w') as file:
        writer=csv.writer(file,lineterminator="\n");writer.writerow(['formula','temperature_K','ln_Q_Pa']);writer.writerows(rows)
    title=r['title'];exponent=sum(r['gas'].values())
    s=title+'\n'+'='*max(8,len(title))+'\n\n'
    s+='Reaction::\n\n   '+equation(r)+'\n\n'
    s+=f'All plotted functions return natural log Q, with each partial pressure expressed numerically in Pa. The net pressure exponent is {exponent}; condensed-phase activity is one. See :doc:`../equilibrium_algorithms` for the signed quotient and H2 treatment.\n\n'
    s+='Data, provenance and limitations\n--------------------------------\n\n'+r['note']+'\n\n'
    s+='The coefficient audit was performed on 2026-10-07. Primary data links:\n\n'
    for label,url in r['sources']:s+=f'* `{label} <{url}>`_\n'
    s+='\nLegacy values are transcribed from ``src/vapors/vapor_functions.h``. PR-only candidates are transcribed from PR 108, revision ``17ba9fb88bf1e50b06b5721b9bb604a1a0691395``. An unresolved provenance label means the fit has not been certified by this audit.\n\n'
    s+='Coefficient conventions\n-----------------------\n\n'
    s+='``ideal`` parameters are [T3, P3, beta_liquid, gamma_liquid, beta_solid, gamma_solid]: L = ln(P3) + beta(1 - T3/T) - gamma ln(T/T3). The solid branch is selected at T <= T3. ``antoine`` parameters are [A, B, C]: L = ln(100000) + ln(10)(A - B/(T+C)). ``linear`` parameters are [A, B, base]: L = ln(base)(A - B/T). Temperatures and B/C offsets are in kelvin. ``table`` contains [temperatures, ln Q] with linear interpolation used only for display.\n\n'
    prows=[]
    for c in r['curves']:
        param=json.dumps(c['params']) if c['kind']!='table' else 'See tabulated values below and source log Kf columns'
        prows.append([c['name'],c['kind'],param,f"{c['interval'][0]}–{c['interval'][1]}",c['status']])
    s+=table(['Formula','Form','Parameters','Plot interval (K)','Status'],prows)
    s+=f'.. figure:: ../_static/reactions/{key}-comparison.svg\n   :alt: Coefficient curves and their log pressure-quotient ratios\n\n   The ratio reference is {primary["name"]}. Curves are compared only where their displayed intervals overlap; disagreement is not an uncertainty estimate.\n\n'
    s+=table(['Formula','T (K)','ln Q (Pa convention)'],rows)
    s+=f':download:`Numerical comparison CSV <../_static/reactions/{key}-values.csv>`\n\n'
    fits=[x for x in data.get('native_curves',[]) if x['reaction']==key]
    if fits:
        s+='Native formula evaluation agrees with the offline coefficient record as follows (float64; mathematical transcription checks, not physical uncertainty):\n\n'
        s+=table(['Formula','Device','Maximum absolute ln Q difference'],[[x['formula'],x['device'],f"{x['max_log_error']:.3e}"] for x in fits])
    else:
        s+='No verified production formula is registered for this family. Its curves above are comparison-only.\n\n'
    s+='Isolated equilibrium validation\n-------------------------------\n\n'
    s+='This reaction is tested independently with its balanced stoichiometry and explicit gas/solid phases. The native TP and UV solvers are compared with a 100-step bracketed scalar extent solve. Manufactured caloric data and a consistent inline equilibrium curve isolate solver correctness; these results do **not** validate the physical fit above. See the algorithm page for the exact construction and acceptance thresholds.\n\n'
    selected=[x for x in data['cases'] if x['reaction']==key]
    if not selected:raise ValueError('Missing native results for '+key)
    vrows=[]
    for mode in ('TP','UV'):
        for case in dict.fromkeys(x['case'] for x in selected):
            xs=[x for x in selected if x['mode']==mode and x['case']==case]
            vrows.append([mode,case,len(xs),*[f'{max(x[field] for x in xs):.3e}' for field in ('state_error','complementarity','conservation','energy_error')], 'pass' if all(x['diag']>=0 for x in xs) else 'FAIL'])
    s+=table(['Mode','Initial state','Runs','Max state error','Max residual violation','Max conservation error','Max energy error','Status'],vrows)
    fig,axes=plt.subplots(1,2,figsize=(8,3.4))
    for mode,a in zip(('TP','UV'),axes):
        xs=[x for x in selected if x['mode']==mode]
        for dtype,mark in [('float64','o'),('float32','x')]:
            ys=[x for x in xs if x['dtype']==dtype]
            a.scatter([x['reference_cloud'] for x in ys],[x['cloud'] for x in ys],label=dtype,marker=mark,s=25)
        lim=max(x['reference_cloud'] for x in xs)*1.1
        a.plot([0,lim],[0,lim],color='0.5',linewidth=.8);a.set_title(mode);a.set_xlabel('Reference cloud [mol, or mol/m³]');a.set_ylabel('Native cloud');a.legend();a.grid(alpha=.2)
    save(fig,assets/(key+'-validation.svg'))
    s+=f'.. figure:: ../_static/reactions/{key}-validation.svg\n   :alt: Native and independent scalar-reference cloud amounts\n\n   Each marker is a recorded native solve. TP amounts are restored using conserved helium; UV amounts are concentrations.\n\n'
    s+='Reproduce this reaction independently::\n\n   python docs/reaction_validation.py --reaction '+key+' --cuda --output /tmp/'+key+'.json\n   pytest -q tests/test_reaction_equilibrium.py -k '+key+'\n\n'
    (pages/(key+'.rst')).write_text('\n'.join(line.rstrip() for line in s.rstrip().splitlines())+'\n')


def algorithms(output,data):
    assets=output/'_static/reactions'
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for mode,a in zip(('TP','UV'),axes):
        for dtype in ('float64','float32'):
            xs=[x for x in data['convergence'] if x['mode']==mode and x['dtype']==dtype]
            a.semilogy([x['budget'] for x in xs],[max(x['complementarity'],1e-15) for x in xs],'o-',label=dtype)
        a.set_title(mode+'; initially absent H2');a.set_xlabel('Restarted solve iteration budget');a.set_ylabel('Complementarity violation');a.grid(alpha=.2);a.legend()
    save(fig,assets/'convergence.svg')
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for mode,a in zip(('TP','UV'),axes):
        for dtype in ('float64','float32'):
            xs=[x for x in data['tolerance'] if x['mode']==mode and x['dtype']==dtype]
            a.loglog([x['ftol'] for x in xs],[max(x['complementarity'],1e-15) for x in xs],'o-',label=dtype)
        a.set_title(mode);a.set_xlabel('Requested log quotient tolerance');a.set_ylabel('Measured violation');a.grid(alpha=.2);a.legend()
    save(fig,assets/'tolerance.svg')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    keys=list(REACTIONS)
    for field,a in zip(('conservation','energy_error'),axes):
        for dtype in ('float64','float32'):
            a.semilogy(range(len(keys)),[max(max(x[field],1e-18) for x in data['cases'] if x['reaction']==k and x['dtype']==dtype) for k in keys],'o-',label=dtype)
        a.set_xticks(range(len(keys)),keys,rotation=60);a.set_ylabel('Max normalized '+field);a.grid(alpha=.2);a.legend()
    save(fig,assets/'conservation.svg')
    fig,axes=plt.subplots(1,2,figsize=(8,3.6))
    for field,a in zip(('state_error','complementarity'),axes):
        for dtype in ('float64','float32'):
            xs=[x for x in data['coupled'] if x['dtype']==dtype]
            labels=[x['mode']+'/'+x['device'] for x in xs]
            a.semilogy(labels,[max(x[field],1e-18) for x in xs],'o-',label=dtype)
        a.set_ylabel(field);a.grid(alpha=.2);a.legend()
    save(fig,assets/'coupled.svg')
    # A measured case map, not interpolation or an unrun numerical grid.
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    cases=list(dict.fromkeys(x['case'] for x in data['cases']))
    for mode,a in zip(('TP','UV'),axes):
        z=[[max(x['complementarity'] for x in data['cases'] if x['reaction']==key and x['case']==case and x['mode']==mode) for key in keys] for case in cases]
        im=a.imshow(np.log10(np.maximum(z,1e-15)),aspect='auto',vmin=-15,vmax=-3)
        a.set_xticks(range(len(keys)),keys,rotation=60);a.set_yticks(range(len(cases)),cases);a.set_title(mode)
    fig.colorbar(im,ax=axes,label='log10 max violation across dtype/device',shrink=.8)
    save(fig,assets/'case-map.svg')
    (assets/'environment.json').write_text(json.dumps(data['environment'],indent=2)+'\n')


def generate(output,keys,validation):
    data=json.loads(validation.read_text())
    for key in keys:reaction(key,output,data)
    if len(keys)==len(REACTIONS):algorithms(output,data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all',action='store_true');group.add_argument('--reaction',choices=list(REACTIONS))
    parser.add_argument('--output-dir',type=Path,default=DEFAULT)
    parser.add_argument('--validation',type=Path,default=DEFAULT/'_static/reactions/validation.json')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();keys=list(REACTIONS) if args.all else [args.reaction]
    if args.check:
        recorded=json.loads(args.validation.read_text())["environment"]["source_sha256"]
        if recorded != source_fingerprint():
            raise SystemExit("Native measurements are stale; rerun reaction_validation.py.")
        with tempfile.TemporaryDirectory() as folder:
            tmp=Path(folder);generate(tmp,keys,args.validation)
            changed=[str(p.relative_to(tmp)) for p in tmp.rglob('*') if p.is_file() and (not (args.output_dir/p.relative_to(tmp)).exists() or p.read_bytes()!=(args.output_dir/p.relative_to(tmp)).read_bytes())]
            if changed:raise SystemExit('Stale generated files: '+', '.join(changed))
        print('Generated reaction documentation is current.')
    else:generate(args.output_dir,keys,args.validation)

if __name__=='__main__':main()
