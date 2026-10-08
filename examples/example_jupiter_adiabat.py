#!/usr/bin/env python3
"""Reversible 1D deep-Jupiter adiabat with KCl and MgSiO3 condensation.

Each temperature trial calls ThermoX.forward (TP equilibrium). A bracketed
outer solve conserves specific entropy, including retained condensates.
Run: python examples/example_jupiter_adiabat.py --output-dir /tmp/jupiter
The MgSiO3 curve and constant heat capacities are illustrative approximations.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from kintera import ThermoOptions, ThermoX, relative_humidity, constants

DEFAULT_CARD = Path(__file__).with_name('jupiter_kcl_mgsio3.yaml')


def run_column(card=DEFAULT_CARD, levels=None):
    """Return an equilibrium isentrope and independent conservation diagnostics."""
    config = yaml.safe_load(Path(card).read_text())
    settings = config['column']
    levels = levels if levels is not None else settings['levels']
    if levels < 2:
        raise ValueError('At least two pressure levels are required')
    pb = float(settings['bottom_pressure_bar']) * 1e5
    pt = float(settings['top_pressure_bar']) * 1e5
    if not 0 < pt < pb:
        raise ValueError('Require 0 < top pressure < bottom pressure')
    tmin, tmax = map(float, settings['temperature_bounds_K'])
    tbase = float(settings['bottom_temperature_K'])
    if not 0 < tmin <= tbase <= tmax:
        raise ValueError('Bottom temperature must be inside the positive bounds')
    gravity = float(settings['gravity_m_s2'])
    tolerance = float(settings['entropy_tolerance_J_kg_K'])
    if gravity <= 0 or tolerance <= 0:
        raise ValueError('Gravity and entropy tolerance must be positive')

    thermo = ThermoX(ThermoOptions.from_yaml(str(card)))
    names = thermo.options.species()
    ngas = len(thermo.options.vapor_ids())
    he = names.index('He')  # Inert amount reference; H2 participates in reaction.
    mole = settings['mole_fractions']
    unknown = set(mole) - set(names)
    if unknown:
        raise ValueError(f'Unknown initial species: {unknown}')
    x0 = np.array([float(mole.get(n, 0.)) if mole.get(n) != 'balance' else 0.
                   for n in names])
    balance = [i for i, name in enumerate(names) if mole.get(name) == 'balance']
    if len(balance) != 1:
        raise ValueError('Exactly one species must have mole fraction balance')
    x0[balance[0]] = 1. - x0.sum()
    if np.any(x0 < 0) or x0[he] <= 0:
        raise ValueError('Initial fractions must be nonnegative with positive He')
    seed = torch.tensor(x0[None, :], dtype=torch.float64)
    mu = thermo.buffer('mu').numpy()
    compositions = {s['name']: s['composition'] for s in config['species']}
    elements = sorted({e for c in compositions.values() for e in c})
    atoms = np.array([[compositions[n].get(e, 0.) for n in names] for e in elements])
    inventory = atoms @ x0
    mass0 = x0 @ mu
    stoich = thermo.buffer('stoich')
    pressures = np.geomspace(pb, pt, levels)
    rows, previous = [], seed

    def equilibrate(temperature, pressure, start):
        # Every trial starts from the same inventory, never the preceding
        # temperature trial. That avoids accumulating root-search history.
        x = start.clone()
        t = torch.tensor([temperature], dtype=torch.float64)
        p = torch.tensor([pressure], dtype=torch.float64)
        diag = torch.zeros((1, 1), dtype=torch.float64)
        thermo.forward(t, p, x, False, diag)
        if diag.item() < 0 or not torch.isfinite(x).all() or (x < 0).any():
            raise RuntimeError(f'TP failed at {temperature:g} K, {pressure/1e5:g} bar: {diag.item()}')
        # This card is ideal gas. Preserve depleted SiH4 below the general
        # TPX->V conversion's 1e-20 mol/m3 floor when checking equilibrium.
        conc = x * p[:, None] / (constants.Rgas*t[:, None]*x[:, :ngas].sum(-1, keepdim=True))
        rho = (conc * thermo.buffer('mu')).sum(-1)
        entropy = thermo.compute('TPV->S', [t, p, conc]) / rho
        return x, conc, float(rho.item()), float(entropy.item()), int(diag.item())

    with torch.no_grad():
        base = equilibrate(tbase, pb, seed)
        target = base[3]
        for level, pressure in enumerate(pressures):
            if level == 0:
                temperature, state, nouter = tbase, base, 0
            else:
                # An ascent cools the parcel. Use the previous equilibrium as
                # the inventory-preserving starting composition for all trials.
                lo, hi = max(tmin, rows[-1]['temperature_K']*.65), rows[-1]['temperature_K']
                low = equilibrate(lo, pressure, previous)
                high = equilibrate(hi, pressure, previous)
                if not low[3] <= target <= high[3]:
                    raise RuntimeError('Entropy root is outside temperature bounds; '
                                       'raise the top pressure or revise the model domain')
                for nouter in range(1, 65):
                    temperature = .5*(lo+hi)
                    state = equilibrate(temperature, pressure, previous)
                    if abs(state[3]-target) <= tolerance:
                        break
                    if state[3] < target:
                        lo = temperature
                    else:
                        hi = temperature
                else:
                    raise RuntimeError('Specific-entropy root failed to converge')
            x, conc, rho, entropy, diag = state
            amounts = x[0].numpy() * x0[he] / x[0, he].item()
            atom_scale = np.where(inventory > 0, inventory, 1.)
            atom_error = float(np.max(np.abs(atoms @ amounts-inventory)/atom_scale))
            mass_error = float(abs(amounts @ mu / mass0-1))
            rh = relative_humidity(torch.tensor([temperature], dtype=torch.float64),
                                   conc, stoich, thermo.options.nucleation(), ngas)[0].numpy()
            cloud = x[0, ngas:].numpy()
            with np.errstate(divide='ignore'):
                violation = np.where(cloud > 1e-15, abs(np.log(rh)), np.maximum(0, np.log(rh)))
            if not np.isfinite(rh).all() or np.max(violation) > 2e-7:
                raise RuntimeError(f'Phase equilibrium check failed at level {level}: T={temperature}, diag={diag}, RH={rh}, x={x}')
            row = dict(pressure_bar=pressure/1e5, temperature_K=temperature,
                       density_kg_m3=rho, entropy_J_kg_K=entropy,
                       entropy_error_J_kg_K=entropy-target,
                       element_relative_error=atom_error, mass_relative_error=mass_error,
                       tp_iterations=diag, entropy_iterations=nouter,
                       rh_KCl=float(rh[0]), rh_MgSiO3=float(rh[1]),
                       phase_residual=float(np.max(violation)))
            if level == 0:
                row['height_km'] = 0.
            else:
                # dP/dz=-rho*g, with bulk density including suspended clouds.
                h0 = pressures[level-1]/rows[-1]['density_kg_m3']/gravity
                h1 = pressure/rho/gravity
                row['height_km'] = rows[-1]['height_km']-.5*(h0+h1)*np.log(pressure/pressures[level-1])/1000
            for i, name in enumerate(names):
                row['x_'+name] = float(x[0, i])  # Fraction of all gas+cloud moles.
                row['q_'+name] = float(x[0, i]*mu[i]/(x[0].numpy()@mu))
            rows.append(row); previous = x

    # Only use the solid KCl branch where a cloud is actually present.
    if any(r['x_KCl(s)'] > 1e-15 and r['temperature_K'] >= 1044 for r in rows):
        raise RuntimeError('Solid KCl cloud encountered above its melting point')
    if max(r['element_relative_error'] for r in rows) > 1e-7:
        raise RuntimeError('Element conservation check failed')
    summary = dict(levels=levels, species=names, closure='reversible; condensates retained',
                   MgSiO3_fit='conditional Visscher table reconstruction; illustrative',
                   max_entropy_error_J_kg_K=max(abs(r['entropy_error_J_kg_K']) for r in rows),
                   max_element_relative_error=max(r['element_relative_error'] for r in rows),
                   max_mass_relative_error=max(r['mass_relative_error'] for r in rows),
                   max_phase_residual=max(r['phase_residual'] for r in rows))
    for name in ('KCl(s)', 'MgSiO3(s)'):
        cloudy = [r for r in rows if r['x_'+name] > 1e-12]
        summary[name+'_first_cloudy_level'] = ({k: cloudy[0][k] for k in ('pressure_bar','temperature_K','height_km')}
                                               if cloudy else None)
    return rows, summary


def write_outputs(rows, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with (output/'profile.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    p = np.array([r['pressure_bar'] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    axes[0].plot([r['temperature_K'] for r in rows], p)
    axes[0].set_xlabel('Temperature [K]'); axes[0].set_ylabel('Pressure [bar]')
    for name in ('KCl', 'KCl(s)', 'Mg', 'SiH4', 'MgSiO3(s)'):
        q = np.array([r['q_'+name] for r in rows])
        axes[1].plot(np.maximum(q, 1e-18), p, label=name)
    axes[1].set_xscale('log'); axes[1].set_xlim(1e-12, 1e-2)
    axes[1].set_xlabel('Mass fraction'); axes[1].legend(fontsize=8)
    for name in ('KCl', 'MgSiO3'):
        axes[2].plot([r['rh_'+name] for r in rows], p, label=name)
    axes[2].axvline(1, color='0.5', linestyle=':')
    axes[2].set_xscale('log'); axes[2].set_xlabel('Saturation ratio Q / Qsat')
    axes[2].legend()
    for ax in axes:
        ax.set_yscale('log'); ax.grid(alpha=.25)
    axes[0].invert_yaxis()
    fig.suptitle('Illustrative deep-Jupiter reversible TP adiabat')
    fig.tight_layout()
    plt.rcParams['svg.hashsalt'] = 'jupiter-tp-adiabat'
    fig.savefig(output/'profile.svg', metadata={'Date': None})
    svg = output/'profile.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(output/'profile.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', type=Path, default=DEFAULT_CARD)
    parser.add_argument('--levels', type=int)
    parser.add_argument('--output-dir', type=Path, default=Path('examples/output/jupiter_adiabat'))
    args = parser.parse_args()
    rows, summary = run_column(args.yaml, args.levels)
    write_outputs(rows, summary, args.output_dir)
    print(json.dumps(summary, indent=2))
    print(f'Wrote profile.csv, summary.json, profile.svg and profile.png in {args.output_dir}')


if __name__ == '__main__':
    main()
