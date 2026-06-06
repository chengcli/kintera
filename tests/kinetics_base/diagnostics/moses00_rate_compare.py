"""Compare per-reaction rates between kintera (at injected KB fort.7 state)
and KB (parsed from REACTION RATES section of kinetics.out)."""
from __future__ import annotations
import sys
sys.path.insert(0, '/home/sam2/dev/kintera/tests/kinetics_base/diagnostics')
exec(open('/home/sam2/dev/kintera/tests/kinetics_base/diagnostics/moses00_perturb.py').read().split('def main()')[0])

import numpy as np
import torch
import base64, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KB_RATES_PATH = "/tmp/kb_moses00_rates.npz"
OUT_HTML = "/tmp/moses00_rate_compare.html"

ts, filtered, species = build_state_and_sources()
ref = kt.parse_kinetics_base_atmosphere(str(REF_PATH))
ts.state.concentration = inject_state(ts, ref)
ts.concentration = ts.state.concentration
species_index = {s: i for i, s in enumerate(species)}

# Load KB rates
kb_data = np.load(KB_RATES_PATH)
kb_rates = kb_data["rates"]  # (521, 91)
nlyr = 91
altitudes = ts.state.x1v.numpy() / 1.0e5

# For each kintera reaction, compute its rate at each altitude
# Rate = k(T) × prod(reactant_concentrations) for thermal, with stoichiometry
# For photo, rate is the same (k×[parent])
# Use kintera's atm2d sources to get tendency, then back out per-reaction rate
# via linearize on individual source terms.

print("Computing kintera per-reaction rates at fort.7 state...")
NREACT_KB = kb_rates.shape[0]
kt_rates = np.zeros((NREACT_KB, nlyr), dtype=np.float64)
kt_term_info: dict[int, dict] = {}
for t in filtered:
    rid = t.reaction_id
    if rid is None or rid <= 0 or rid > NREACT_KB:
        continue
    # Build individual atm2d source for this term and compute rate
    atm_one = kt.build_kinetics_base_titan_atm2d_source_terms(ts, [t])
    if not atm_one:
        continue
    for src in atm_one:
        try:
            lin = src.linearize(ts.state)
        except Exception:
            continue
        # Rate = -tendency on first reactant divided by stoich coef
        # (for non-zero stoich; many reactions consume the first reactant 1:1)
        if not t.reactants:
            continue
        # For photo (1 reactant): rate = -tendency[reactant] / coef[0]
        # For thermal (≥2 reactants): rate = -tendency[reactant_0] / coef[0]
        r0 = t.reactants[0]
        if r0 not in species_index:
            continue
        i0 = species_index[r0]
        tend = lin.tendency[0, :, i0]  # (nlyr,)
        coef = (t.parameters.get("reactant_coefficients") or [1])[0]
        rate = -tend.numpy() / max(coef, 1)
        kt_rates[rid - 1] = np.maximum(kt_rates[rid - 1], rate)  # take max if multiple sources/branches share rid (e.g. photo branches)
        kt_term_info[rid] = {
            "reactants": list(t.reactants),
            "products": list(t.products),
            "kind": t.kind,
        }

print(f"Computed kintera rates for {len(kt_term_info)} reactions")
print(f"kt_rates shape: {kt_rates.shape}, max: {kt_rates.max():.3e}")
print(f"KB rates max: {kb_rates.max():.3e}")

# Compare at key altitudes
LAYERS_OF_INTEREST = [10, 20, 30, 40, 50, 60, 70, 80]

# Aggregate: for each reaction, ratio kt/KB at each layer
ratios = np.full_like(kt_rates, np.nan)
both_nonzero = (kt_rates > 1e-30) & (kb_rates > 1e-30)
ratios[both_nonzero] = kt_rates[both_nonzero] / kb_rates[both_nonzero]

# Top mismatches: large absolute rate AND large ratio
def reaction_label(rid):
    info = kt_term_info.get(rid)
    if not info:
        return f"rxn {rid} (unknown)"
    return f"rxn {rid}: {' + '.join(info['reactants'])} -> {' + '.join(info['products'])}"

# Sort by max(kt, kb) × max log-ratio
score = np.zeros(kt_rates.shape[0])
for rid in range(kt_rates.shape[0]):
    if rid >= kb_rates.shape[0]:
        break
    kt_max = kt_rates[rid].max()
    kb_max = kb_rates[rid].max()
    if max(kt_max, kb_max) < 1e-4:
        continue
    # log ratio magnitude (where both non-zero)
    valid_layers = (kt_rates[rid] > 1e-30) & (kb_rates[rid] > 1e-30)
    if not valid_layers.any():
        # If KB has rate but kintera 0 (or vice versa): big mismatch
        score[rid] = max(kt_max, kb_max) * 100
        continue
    log_ratio = np.log10(kt_rates[rid][valid_layers] / kb_rates[rid][valid_layers])
    score[rid] = max(kt_max, kb_max) * np.abs(log_ratio).max()

top_mismatches = np.argsort(-score)[:30]

print("\nTop 30 reactions with largest kt-vs-KB mismatch (weighted by rate magnitude):")
print(f"{'rid':<5} {'reaction':<55} {'max rate kt':<13} {'max rate KB':<13} {'max |log ratio|':<15}")
for rid in top_mismatches:
    if score[rid] == 0:
        continue
    info = kt_term_info.get(int(rid), {})
    if not info:
        continue
    kt_max = kt_rates[rid].max()
    kb_max = kb_rates[rid].max()
    valid = (kt_rates[rid] > 1e-30) & (kb_rates[rid] > 1e-30)
    if valid.any():
        max_logr = np.abs(np.log10(kt_rates[rid][valid] / kb_rates[rid][valid])).max()
    else:
        max_logr = float('inf')
    label = f"{' + '.join(info['reactants'])} -> {' + '.join(info['products'])}"
    print(f"{int(rid):<5} {label[:55]:<55} {kt_max:<13.2e} {kb_max:<13.2e} {max_logr:<15.3f}")

# Generate HTML
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=72, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

print("\nGenerating HTML report...")
parts = ["""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>moses00 kintera vs KB per-reaction rate comparison</title>
<style>
body { font-family: -apple-system, sans-serif; margin: 20px; max-width: 1500px; }
h1, h2 { border-bottom: 1px solid #888; padding-bottom: 4px; }
table { border-collapse: collapse; font-size: 0.85em; margin-top: 8px; }
th, td { border: 1px solid #ccc; padding: 3px 7px; text-align: right; }
th { background: #eee; }
td.label { text-align: left; font-weight: 600; }
.good { color: #1c7f1c; }
.ok { color: #b06a00; }
.bad { color: #a01010; }
.zero { color: #aaa; }
</style></head><body>
<h1>moses00: kintera vs KB per-reaction rates</h1>
<p>State: KB fort.7 SS injected into kintera. Both sides compute reaction
rate = k(T) × ∏[reactant_i] in cm⁻³/s. KB rates parsed from
<code>kinetics.out-revert</code> REACTION RATES section.</p>
"""]

# Top mismatches table
parts.append("<h2>Top 30 mismatches (weighted by rate magnitude × |log10(kt/KB)|)</h2>")
parts.append("<table><tr><th>rid</th><th>reaction</th>")
for L in LAYERS_OF_INTEREST:
    parts.append(f"<th colspan=3>L{L} (z={altitudes[L]:.0f}km)</th>")
parts.append("</tr><tr><th></th><th></th>")
for L in LAYERS_OF_INTEREST:
    parts.append("<th>kt</th><th>KB</th><th>ratio</th>")
parts.append("</tr>")
for rid in top_mismatches:
    if score[rid] == 0:
        continue
    info = kt_term_info.get(int(rid), {})
    if not info:
        continue
    label = f"{' + '.join(info['reactants'])} → {' + '.join(info['products'])}"
    parts.append(f"<tr><td>{rid}</td><td class='label'>{label}</td>")
    for L in LAYERS_OF_INTEREST:
        kt_v = kt_rates[rid, L]
        kb_v = kb_rates[rid, L]
        if kt_v < 1e-30 and kb_v < 1e-30:
            parts.append("<td class='zero'>·</td><td class='zero'>·</td><td class='zero'>·</td>")
        else:
            if kb_v > 1e-30:
                r = kt_v / kb_v
                cls = 'good' if 0.5 <= r <= 2.0 else ('ok' if 0.1 <= r <= 10.0 else 'bad')
                r_str = f"{r:.2f}"
            else:
                cls = 'bad'
                r_str = "∞"
            parts.append(f"<td>{kt_v:.2e}</td><td>{kb_v:.2e}</td>"
                         f"<td class='{cls}'>{r_str}</td>")
    parts.append("</tr>")
parts.append("</table>")

# Sum of |kt-kb| per layer, scaled
parts.append("<h2>Total chemistry tendency residual per altitude</h2>")
parts.append("<p>Sum across all reactions of (kt_rate − KB_rate) at each altitude.</p>")
kt_sum = kt_rates.sum(axis=0)
kb_sum = kb_rates.sum(axis=0)
parts.append("<table><tr><th>L</th><th>z (km)</th><th>kt total</th><th>KB total</th><th>diff</th></tr>")
for L in range(0, 91, 5):
    parts.append(f"<tr><td>L{L}</td><td>{altitudes[L]:.0f}</td>"
                 f"<td>{kt_sum[L]:.2e}</td><td>{kb_sum[L]:.2e}</td>"
                 f"<td>{kt_sum[L]-kb_sum[L]:+.2e}</td></tr>")
parts.append("</table>")

parts.append("</body></html>")
with open(OUT_HTML, "w") as f:
    f.write("".join(parts))
print(f"Wrote {OUT_HTML}")
np.savez("/tmp/moses00_rate_compare.npz",
         kt_rates=kt_rates,
         kb_rates=kb_rates,
         altitudes=altitudes,
         reaction_labels=np.array([f"{i}: {kt_term_info.get(i,{}).get('reactants','')}->{kt_term_info.get(i,{}).get('products','')}"
                                    for i in range(kt_rates.shape[0])], dtype=object))
print(f"Wrote /tmp/moses00_rate_compare.npz")
