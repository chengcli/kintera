"""Transport-divergence audit at KB fort.7 SS state.

KB at fort.7 has dc/dt = 0 by definition. Since chemistry rates now match
KB at 1.000 across all altitudes (moses00-validated milestone), the
expected KB-side transport divergence at fort.7 is:

    KB_transport(s, L) = − KB_chem_net(s, L)

where KB_chem_net is the per-species production − loss summed over all
reactions (parsed from KB's "REACTION RATES:" section).

Compare against kintera's actual transport divergence at fort.7 using
`build_eddy_diffusion_matrix` with mr_diffusion form. The diff per
species per layer is the pure transport-form gap, isolated from any
chemistry confounding.

Writes /tmp/moses00_transport_audit.html with per-species profile plots.
"""
from __future__ import annotations
import sys, base64, io
sys.path.insert(0, '/home/sam2/dev/kintera/tests/kinetics_base/diagnostics')
exec(open('/home/sam2/dev/kintera/tests/kinetics_base/diagnostics/moses00_perturb.py').read().split('def main()')[0])

import os
os.environ.setdefault('KINTERA_DISABLE_CHEMB_OVERRIDES', '1')

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NALT_OFFSET = 8  # KB row N (1-indexed) → kintera 0-indexed L(N+8)
NLYR = 91

# Build kintera state and inject fort.7
ts, filtered, species = build_state_and_sources()
ref = kt.parse_kinetics_base_atmosphere(str(REF_PATH))
ts.state.concentration = inject_state(ts, ref)
ts.concentration = ts.state.concentration
species_index = {s: i for i, s in enumerate(species)}
altitude = ts.state.x1v.numpy() / 1.0e5
density = ts.density[0].numpy()

# Load KB rates and align to kintera 0-indexed
kb_raw = np.load("/tmp/kb_moses00_rates.npz")["rates"]  # (NREACT, 91), KB-row-indexed
NREACT_KB = kb_raw.shape[0]
kb_rates = np.zeros_like(kb_raw)
for kb_row_idx in range(NLYR):
    kt_layer = kb_row_idx + NALT_OFFSET + 1
    if 0 <= kt_layer < NLYR:
        kb_rates[:, kt_layer] = kb_raw[:, kb_row_idx]

# Parse PUN to get reactants/products + coefficients per reaction
pun = kt.parse_kinetics_base_pun(str(PUN_PATH))
sp_by_id = {s.id: s.name for s in pun.species}
reaction_stoich: dict[int, list[tuple[int, float]]] = {}  # rid → [(sp_idx, signed_coef), ...]
for rxn in pun.reactions:
    rid = rxn.id
    if rid > NREACT_KB:
        continue
    contrib: list[tuple[int, float]] = []
    for j, p in enumerate(rxn.participants[:rxn.n_reactants]):
        sp_name = sp_by_id.get(p.species_id)
        if sp_name is None or sp_name not in species_index:
            continue
        coef = int(p.coefficient) or 1
        contrib.append((species_index[sp_name], -float(coef)))
    for j, p in enumerate(
        rxn.participants[rxn.n_reactants : rxn.n_reactants + rxn.n_products]
    ):
        sp_name = sp_by_id.get(p.species_id)
        if sp_name is None or sp_name not in species_index:
            continue
        coef = int(p.coefficient) or 1
        contrib.append((species_index[sp_name], +float(coef)))
    reaction_stoich[rid] = contrib

# Build KB chemistry tendency per species per layer:
# dc_KB(s, L) = Σ_rxn stoich_rxn(s) × KB_rate(rxn, L)
kb_chem_tend = np.zeros((NLYR, len(species)))
for rid, stoich in reaction_stoich.items():
    if rid - 1 >= NREACT_KB:
        continue
    rate_profile = kb_rates[rid - 1]  # (NLYR,)
    for sp_idx, signed_coef in stoich:
        kb_chem_tend[:, sp_idx] += signed_coef * rate_profile

# KB transport (expected) = − KB chem
kb_transport_expected = -kb_chem_tend

# kintera transport divergence at fort.7
trans_mat = kt.build_eddy_diffusion_matrix(
    ts.state, ts.kzz, form="mr_diffusion", density=ts.density,
)
kt_transport = trans_mat.matvec(ts.state.concentration)[0].numpy()  # (NLYR, nspecies)

# Diff
gap = kt_transport - kb_transport_expected

# Report at key altitudes
LAYERS = [20, 30, 40, 50, 60, 65, 70, 75, 80]
REPORT_SPECIES = ["H", "H2", "CH4", "CH3", "C2H2", "C2H4", "C2H6",
                  "C2H", "C2", "C", "CN", "HCN", "HC3N", "C2N2", "CH"]

print(f"{'Species':<8} {'L':<4} {'z(km)':<7} {'c (cm-3)':<10}"
      f" {'kt_trans':<11} {'KB_trans':<11} {'gap':<11} {'gap/c':<9}")
for sp in REPORT_SPECIES:
    if sp not in species_index:
        continue
    i = species_index[sp]
    for L in LAYERS:
        c = ts.state.concentration[0, L, i].item()
        kt_t = kt_transport[L, i]
        kb_t = kb_transport_expected[L, i]
        g = gap[L, i]
        rel = g / c if c > 0 else float('nan')
        print(f"{sp:<8} L{L:<3d} {altitude[L]:<7.0f} {c:<10.2e}"
              f" {kt_t:<+11.2e} {kb_t:<+11.2e} {g:<+11.2e} {rel:<+9.2e}")
    print()

# Save raw arrays
np.savez("/tmp/moses00_transport_audit.npz",
         species=np.array(species, dtype=object),
         altitude=altitude,
         density=density,
         c_fort7=ts.state.concentration[0].numpy(),
         kt_transport=kt_transport,
         kb_transport_expected=kb_transport_expected,
         gap=gap)
print(f"\nSaved arrays to /tmp/moses00_transport_audit.npz")

# Generate HTML report
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=72, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

panel_imgs = {}
for sp in REPORT_SPECIES:
    if sp not in species_index:
        continue
    i = species_index[sp]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(kt_transport[:, i], altitude, 'o-', color='#cc3333',
            label='kintera transport', ms=2.5, lw=1.0)
    ax.plot(kb_transport_expected[:, i], altitude, 's-', color='#000080',
            label='KB transport (= −chem)', ms=2.5, lw=1.0)
    ax.axvline(0, color='gray', alpha=0.3)
    ax.set_xlabel("transport tendency (cm⁻³/s)")
    ax.set_ylabel("altitude (km)")
    ax.set_title(sp)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1500)
    # Symmetric x range for clarity
    mx = max(abs(kt_transport[:, i]).max(), abs(kb_transport_expected[:, i]).max())
    if mx > 0:
        ax.set_xlim(-1.2*mx, 1.2*mx)
    panel_imgs[sp] = fig_to_b64(fig)
    plt.close(fig)

html = ['''<!DOCTYPE html><html><head><meta charset="utf-8">
<title>moses00 transport audit at fort.7 SS</title>
<style>
body{font-family:-apple-system,sans-serif;margin:24px;max-width:1500px}
h1,h2{border-bottom:1px solid #888;padding-bottom:4px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.panel img{max-width:100%}
table{border-collapse:collapse;font-family:'SF Mono',Menlo,monospace;font-size:0.85em;margin-top:8px}
th,td{border:1px solid #ccc;padding:3px 7px;text-align:right}
th{background:#eee}
td.sp{text-align:left;font-weight:600;background:#fafafa}
</style></head><body>''']
html.append('<h1>moses00 transport-divergence audit at KB fort.7 SS</h1>')
html.append('<p>After the moses00-validated milestone, chemistry rates match KB at '
            'ratio = 1.000. So KB transport at fort.7 SS = −(KB chemistry tendency). '
            'Compare against kintera\'s actual transport divergence using '
            '<code>build_eddy_diffusion_matrix(form="mr_diffusion")</code>.</p>')
html.append('<h2>Per-species profiles</h2><div class="grid">')
for sp in REPORT_SPECIES:
    if sp in panel_imgs:
        html.append(f"<div class='panel'><b>{sp}</b><img src='data:image/png;base64,{panel_imgs[sp]}'/></div>")
html.append('</div>')

html.append('<h2>Gap at key altitudes (kintera − KB)</h2>')
html.append('<table><tr><th>Species</th><th>L20</th><th>L40</th><th>L60</th><th>L70</th><th>L80</th></tr>')
for sp in REPORT_SPECIES:
    if sp not in species_index:
        continue
    i = species_index[sp]
    row = [f"<td class='sp'>{sp}</td>"]
    for L in [20, 40, 60, 70, 80]:
        row.append(f"<td>{gap[L, i]:+.2e}</td>")
    html.append("<tr>" + "".join(row) + "</tr>")
html.append('</table></body></html>')
with open("/tmp/moses00_transport_audit.html", "w") as f:
    f.write("".join(html))
print("Wrote /tmp/moses00_transport_audit.html")
