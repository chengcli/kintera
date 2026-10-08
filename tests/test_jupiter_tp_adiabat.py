"""Integration checks for the runnable deep-Jupiter TP adiabat example."""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples'))
from example_jupiter_adiabat import DEFAULT_CARD, run_column


def test_reversible_cloud_column_and_grid_convergence():
    coarse, cdiag = run_column(levels=31)
    fine, fdiag = run_column(levels=61)
    for rows, diag in ((coarse, cdiag), (fine, fdiag)):
        assert diag['max_entropy_error_J_kg_K'] <= 1e-5
        assert diag['max_element_relative_error'] < 1e-10
        assert diag['max_mass_relative_error'] < 1e-12
        assert diag['max_phase_residual'] < 2e-7
        assert all(r['tp_iterations'] > 0 for r in rows)
        assert all(r['temperature_K'] >= 800 for r in rows)
        assert all(a['temperature_K'] > b['temperature_K'] for a,b in zip(rows, rows[1:]))
        assert all(a['height_km'] < b['height_km'] for a,b in zip(rows, rows[1:]))
        for name in ('KCl(s)', 'MgSiO3(s)'):
            assert rows[0]['x_'+name] == 0.
            assert rows[-1]['x_'+name] > 0.
            assert diag[name+'_first_cloudy_level'] is not None
    np.testing.assert_allclose([r['temperature_K'] for r in coarse],
                               [r['temperature_K'] for r in fine[::2]], rtol=0, atol=1e-4)
    np.testing.assert_allclose([r['q_MgSiO3(s)'] for r in coarse],
                               [r['q_MgSiO3(s)'] for r in fine[::2]], rtol=1e-6, atol=1e-13)


def test_no_condensable_inventory_recovers_analytic_dry_adiabat(tmp_path):
    config = yaml.safe_load(DEFAULT_CARD.read_text())
    for name in ('KCl','Mg','SiH4'):
        config['column']['mole_fractions'][name] = 0.
    path = tmp_path/'dry.yaml'; path.write_text(yaml.safe_dump(config))
    rows, diag = run_column(path, levels=17)
    initial = config['column']['mole_fractions']
    initial['H2'] = 1-sum(v for k,v in initial.items() if k!='H2')
    cp_R = sum(initial.get(s['name'],0.)*(s['cv_R']+1) for s in config['species'] if s['phase']=='gas')
    p = np.array([r['pressure_bar'] for r in rows])
    expected = rows[0]['temperature_K']*(p/p[0])**(1/cp_R)
    np.testing.assert_allclose([r['temperature_K'] for r in rows], expected, rtol=0, atol=1e-5)
    assert diag['max_element_relative_error'] < 1e-12
    assert all(r['x_KCl(s)']==r['x_MgSiO3(s)']==0 for r in rows)
