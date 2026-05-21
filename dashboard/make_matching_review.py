"""Generate matching.html — a per-species, per-level interactive review.

Color-coded ratio table, sortable, with filter buttons (neutral / cation /
anion / grain / all). Click a species name to expand its full profile vs KB.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import kintera as kt


DASH = pathlib.Path("/home/sam2/dev/kintera/dashboard")

# ----- load -----
d = np.load("/tmp/baseline_g29.npz", allow_pickle=True)
SPECIES = [str(x) for x in d["species"]]
C = d["concentration"]
KB = kt.parse_kinetics_base_atmosphere("/tmp/kb_run_500/fort.7").species_profiles
INITIAL = kt.parse_kinetics_base_atmosphere(
    "/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare/examples/titan/"
    "kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz"
)
ALT = np.asarray(INITIAL.altitude)
NLYR = C.shape[0]


def species_kind(name: str) -> str:
    if name.endswith("+"):
        return "cation"
    if name.endswith("-"):
        return "anion"
    if name.startswith("G") and len(name) > 1 and name[1].isupper():
        return "grain"
    if name in {"N2", "M", "JDUST", "PROD", "SGA", "U", "RAYEAR", "AR"}:
        return "fixed"
    return "neutral"


def species_summary(name: str):
    """Per-species summary: levels with KB ≥ 1, % matched, max-ratio, min-ratio."""
    j = SPECIES.index(name)
    kt_prof = C[:, j]
    kb_prof = np.asarray(KB.get(name, np.zeros(NLYR)))
    levels = []
    matched = 0
    in10x = 0
    in100x = 0
    n_valid = 0
    max_ratio = 0.0
    min_ratio = 1e30
    for lev in range(NLYR):
        kbv = kb_prof[lev]
        if kbv < 1:
            levels.append({"L": lev, "alt": float(ALT[lev]), "kt": float(kt_prof[lev]),
                           "kb": float(kbv), "ratio": None})
            continue
        n_valid += 1
        v = float(kt_prof[lev])
        r = v / float(kbv) if kbv else 0.0
        levels.append({"L": lev, "alt": float(ALT[lev]), "kt": v, "kb": float(kbv), "ratio": r})
        if r > 0:
            max_ratio = max(max_ratio, r)
            min_ratio = min(min_ratio, r)
        if 0.3 < r < 3.0:
            matched += 1
        if 0.1 < r < 10.0:
            in10x += 1
        if 0.01 < r < 100.0:
            in100x += 1
    return {
        "name": name,
        "kind": species_kind(name),
        "n_valid": n_valid,
        "matched": matched,
        "in10x": in10x,
        "in100x": in100x,
        "pct_matched": (100 * matched / n_valid) if n_valid else 0,
        "max_ratio": max_ratio if max_ratio > 0 else None,
        "min_ratio": min_ratio if min_ratio < 1e29 else None,
        "levels": levels,
    }


summaries = [species_summary(s) for s in SPECIES]
# Drop species with no KB data at all
summaries = [s for s in summaries if s["n_valid"] > 0]

# ----- write HTML -----
html_path = DASH / "matching.html"

# Generate inline JSON data
data_json = json.dumps(summaries)

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>kintera × KB — Matching Detail</title>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    line-height: 1.5;
    max-width: 1500px;
    margin: 1em auto;
    padding: 0 1.2em;
    color: #222;
    background: #fafafa;
  }
  h1 { border-bottom: 3px solid #2a6; padding-bottom: 0.2em; }
  h2 { color: #2a6; margin-top: 1.5em; border-bottom: 1px solid #ddd; }
  code, .mono { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 0.92em; }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.5em 0;
    font-size: 0.88em;
    font-family: "SF Mono", Menlo, Consolas, monospace;
  }
  th, td { border: 1px solid #ccc; padding: 3px 8px; text-align: right; }
  th {
    background: #e8f3ec;
    cursor: pointer;
    user-select: none;
    position: sticky;
    top: 0;
    z-index: 2;
  }
  th:hover { background: #cce6d4; }
  td.name { text-align: left; font-weight: bold; cursor: pointer; }
  td.name:hover { color: #2a6; }
  .kind-neutral { color: #444; }
  .kind-cation  { color: #c70; }
  .kind-anion   { color: #b22; }
  .kind-grain   { color: #06a; }
  .kind-fixed   { color: #888; font-style: italic; }
  .ratio-perfect  { background: #c8e6c9; }
  .ratio-good     { background: #e8f3ec; }
  .ratio-fair     { background: #fff8e8; }
  .ratio-poor     { background: #ffe0e0; }
  .ratio-bad      { background: #ffb3b3; }
  .filter-bar {
    background: #fff;
    border: 1px solid #ddd;
    padding: 0.7em 1em;
    border-radius: 4px;
    margin: 1em 0;
    font-size: 0.93em;
  }
  .filter-bar button {
    background: #f0f0f0;
    border: 1px solid #aaa;
    padding: 4px 12px;
    margin: 0 4px;
    border-radius: 3px;
    cursor: pointer;
    font-size: 0.9em;
  }
  .filter-bar button.active { background: #2a6; color: white; border-color: #2a6; }
  .filter-bar input {
    padding: 4px 8px;
    border: 1px solid #aaa;
    border-radius: 3px;
    margin-left: 1em;
    width: 180px;
  }
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.6em;
    margin: 1em 0;
  }
  .stat {
    background: #fff;
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 0.6em;
    text-align: center;
  }
  .stat .v { font-size: 1.4em; font-weight: bold; color: #2a6; }
  .stat .l { font-size: 0.78em; color: #666; }
  .detail-row { background: #f4f4f7; display: none; }
  .detail-row.show { display: table-row; }
  .detail-cell { font-size: 0.84em; padding: 6px 12px; }
  .small { font-size: 0.85em; color: #666; }
  details summary {
    cursor: pointer;
    font-weight: bold;
    padding: 0.3em 0;
  }
</style>
</head>
<body>

<h1>kintera × KB — Matching Detail</h1>
<p class="small">
  Best config: coupled + chg fold + G29 + full grain + dt=1e+7 + NT=100 ·
  baseline dump <code>/tmp/baseline_g29.npz</code> ·
  <a href="index.html">← back to dashboard</a>
</p>

<div class="stat-grid" id="stats"></div>

<div class="filter-bar">
  <b>Filter:</b>
  <button class="active" onclick="setFilter('all')">all</button>
  <button onclick="setFilter('neutral')">neutrals</button>
  <button onclick="setFilter('cation')">cations</button>
  <button onclick="setFilter('anion')">anions</button>
  <button onclick="setFilter('grain')">grain</button>
  <button onclick="setFilter('matched')">matched ≥ 50%</button>
  <button onclick="setFilter('poor')">poor matches</button>
  <input type="text" id="search" placeholder="search species name..." oninput="rebuild()">
</div>

<table id="main">
  <thead>
    <tr>
      <th onclick="sortBy('name')">species</th>
      <th onclick="sortBy('kind')">kind</th>
      <th onclick="sortBy('n_valid')">KB levels</th>
      <th onclick="sortBy('matched')">matched (0.3-3×)</th>
      <th onclick="sortBy('pct')">match %</th>
      <th onclick="sortBy('in10x')">within 10×</th>
      <th onclick="sortBy('in100x')">within 100×</th>
      <th onclick="sortBy('min_ratio')">min ratio</th>
      <th onclick="sortBy('max_ratio')">max ratio</th>
    </tr>
  </thead>
  <tbody id="body"></tbody>
</table>

<script>
const DATA = __DATA__;
let filter = 'all';
let sortKey = 'pct';
let sortDir = -1;

function fmt(x) {
  if (x === null || x === undefined) return '—';
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 0.01 && a < 1e5) return x.toFixed(a < 1 ? 3 : 1);
  return x.toExponential(2);
}

function ratioClass(r) {
  if (r === null || r === undefined) return '';
  if (r > 0.5 && r < 2.0) return 'ratio-perfect';
  if (r > 0.3 && r < 3.0) return 'ratio-good';
  if (r > 0.1 && r < 10.0) return 'ratio-fair';
  if (r > 0.01 && r < 100.0) return 'ratio-poor';
  return 'ratio-bad';
}

function passFilter(s) {
  if (filter === 'all') return true;
  if (filter === 'matched') return s.pct_matched >= 50;
  if (filter === 'poor') return s.pct_matched < 30 && s.n_valid >= 3;
  return s.kind === filter;
}

function passSearch(s) {
  const q = document.getElementById('search').value.toLowerCase();
  if (!q) return true;
  return s.name.toLowerCase().includes(q);
}

function sortBy(key) {
  if (sortKey === key) { sortDir = -sortDir; } else { sortKey = key; sortDir = -1; }
  rebuild();
}

function setFilter(f) {
  filter = f;
  document.querySelectorAll('.filter-bar button').forEach(b => b.classList.toggle('active', b.textContent.includes(f) || (f === 'all' && b.textContent === 'all')));
  rebuild();
}

function getSortVal(s, key) {
  if (key === 'name') return s.name;
  if (key === 'kind') return s.kind;
  if (key === 'pct')  return s.pct_matched;
  if (key === 'min_ratio') return s.min_ratio !== null ? s.min_ratio : 1e30;
  if (key === 'max_ratio') return s.max_ratio !== null ? s.max_ratio : 0;
  return s[key];
}

function buildStats() {
  const visible = DATA.filter(s => passFilter(s) && passSearch(s));
  const total = visible.reduce((a, s) => a + s.n_valid, 0);
  const matched = visible.reduce((a, s) => a + s.matched, 0);
  const in10 = visible.reduce((a, s) => a + s.in10x, 0);
  const in100 = visible.reduce((a, s) => a + s.in100x, 0);
  const ge50 = visible.filter(s => s.pct_matched >= 50).length;
  const html = `
    <div class="stat"><div class="v">${visible.length}</div><div class="l">species shown</div></div>
    <div class="stat"><div class="v">${matched}/${total}</div><div class="l">levels matched (0.3-3×)</div></div>
    <div class="stat"><div class="v">${in10}/${total}</div><div class="l">levels within 10×</div></div>
    <div class="stat"><div class="v">${in100}/${total}</div><div class="l">levels within 100×</div></div>
    <div class="stat"><div class="v">${ge50}</div><div class="l">species ≥50% matched</div></div>
  `;
  document.getElementById('stats').innerHTML = html;
}

function toggleDetail(idx) {
  const row = document.getElementById('detail-' + idx);
  if (row) row.classList.toggle('show');
}

function buildRow(s, idx) {
  const cls = ratioClass(s.max_ratio);
  let detail = '<tr id="detail-' + idx + '" class="detail-row"><td class="detail-cell" colspan="9">';
  detail += '<details open><summary>per-level profile</summary>';
  detail += '<table style="margin: 0.3em 0; font-size: 0.82em"><tr><th>L</th><th>alt(km)</th><th>kt</th><th>KB</th><th>ratio</th></tr>';
  for (const lev of s.levels) {
    if (lev.kb < 1) continue;
    const rcls = ratioClass(lev.ratio);
    detail += `<tr class="${rcls}"><td>${lev.L}</td><td>${lev.alt.toFixed(1)}</td><td>${fmt(lev.kt)}</td><td>${fmt(lev.kb)}</td><td>${fmt(lev.ratio)}×</td></tr>`;
  }
  detail += '</table></details></td></tr>';
  return `<tr>
    <td class="name kind-${s.kind}" onclick="toggleDetail(${idx})">${s.name}</td>
    <td class="kind-${s.kind}">${s.kind}</td>
    <td>${s.n_valid}</td>
    <td>${s.matched}</td>
    <td><span style="display:inline-block;width:60%;background:#f0f0f0">
        <span style="display:inline-block;width:${s.pct_matched}%;background:#2a6;height:8px"></span></span> ${s.pct_matched.toFixed(0)}%</td>
    <td>${s.in10x}</td>
    <td>${s.in100x}</td>
    <td class="${ratioClass(s.min_ratio)}">${fmt(s.min_ratio)}×</td>
    <td class="${ratioClass(s.max_ratio)}">${fmt(s.max_ratio)}×</td>
  </tr>` + detail;
}

function rebuild() {
  const visible = DATA.filter(s => passFilter(s) && passSearch(s));
  visible.sort((a, b) => {
    const av = getSortVal(a, sortKey);
    const bv = getSortVal(b, sortKey);
    if (typeof av === 'string') return sortDir * av.localeCompare(bv);
    return sortDir * (av - bv);
  });
  document.getElementById('body').innerHTML = visible.map((s, i) => buildRow(s, i)).join('');
  buildStats();
}

rebuild();
</script>

</body>
</html>
"""

html = html.replace("__DATA__", data_json)
html_path.write_text(html)
print(f"[done] wrote {html_path} ({len(html)/1024:.0f} KB)")
