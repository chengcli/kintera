"""Run moses00 to steady state, generate HTML report comparing to KB fort.7.

Reads kintera SS from `/tmp/kt_moses00_ss.npz` (produced by moses00_match.py)
and KB fort.7 SS from the .kt-converted reference file, then writes a
self-contained HTML with per-species altitude-profile plots and a summary
table at key altitudes.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kintera as kt

EX_DIR = Path(
    "/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare/examples/titan_moses00"
)
KB_SS_PATH = EX_DIR / "case" / "fort.7.kt"
KT_SS_PATH = Path("/tmp/kt_moses00_ss.npz")
OUT_HTML = Path("/tmp/moses00_ss_report.html")

# Species to highlight (one panel each)
PANEL_SPECIES = [
    "H", "H2", "CH4", "CH3",
    "C2H2", "C2H4", "C2H6", "C2H5",
    "C3H8", "C3H6", "1-C4H6", "C4H10",
    "HCN", "HC3N", "C2N2", "CN",
    "C6H2", "C6H3", "C", "C2H",
]
SUMMARY_LAYERS = [20, 30, 40, 50, 60, 70]


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def main():
    # Load kintera SS
    if not KT_SS_PATH.exists():
        raise SystemExit(
            f"Need {KT_SS_PATH}. Run `diagnostics/moses00_match.py` first."
        )
    kt_ss = np.load(KT_SS_PATH, allow_pickle=True)
    species = [str(s) for s in kt_ss["species"]]
    altitude = kt_ss["altitude"]  # km
    density = kt_ss["density"]
    c_kintera = kt_ss["c_kintera_ss"]  # (nlyr, nspecies) concentrations

    # Load KB fort.7 (mixing ratios; multiply by density)
    ref = kt.parse_kinetics_base_atmosphere(str(KB_SS_PATH))
    kb_mr = ref.species_profiles  # dict[str -> list[float]]

    nlyr = len(altitude)
    c_kb = np.zeros((nlyr, len(species)))
    for i, sp in enumerate(species):
        if sp not in kb_mr:
            continue
        prof = np.asarray(kb_mr[sp])
        if prof.size != nlyr:
            continue
        # KB stores MR (except for synthetic species we skip below)
        if sp in ("M", "JDUST"):
            continue
        c_kb[:, i] = prof * density

    # Convert kintera SS to MR for comparison
    with np.errstate(divide="ignore", invalid="ignore"):
        mr_kt = np.where(density[:, None] > 0, c_kintera / density[:, None], 0.0)
        mr_kb = np.where(density[:, None] > 0, c_kb / density[:, None], 0.0)

    # ---- Summary table at SUMMARY_LAYERS ----
    rows = []
    for sp in PANEL_SPECIES:
        if sp not in species:
            continue
        i = species.index(sp)
        cells = []
        for L in SUMMARY_LAYERS:
            kb_v = mr_kb[L, i]
            kt_v = mr_kt[L, i]
            if abs(kb_v) > 1e-30:
                ratio = kt_v / kb_v
                ratio_str = f"{ratio:.2f}"
                if 0.5 <= ratio <= 2.0:
                    color = "#1c7f1c"  # green
                elif 0.2 <= ratio <= 5.0:
                    color = "#b06a00"  # orange
                else:
                    color = "#a01010"  # red
            else:
                ratio_str = "—"
                color = "#777"
            cells.append((kb_v, kt_v, ratio_str, color))
        rows.append((sp, cells))

    # ---- Per-species profile plots ----
    panel_imgs = {}
    for sp in PANEL_SPECIES:
        if sp not in species:
            continue
        i = species.index(sp)
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        kb_p = mr_kb[:, i]
        kt_p = mr_kt[:, i]
        # Mask zeros for log axis
        kb_mask = np.where(kb_p > 1e-30, kb_p, np.nan)
        kt_mask = np.where(kt_p > 1e-30, kt_p, np.nan)
        ax.plot(kb_mask, altitude, "o-", color="#000080", label="KB fort.7", ms=3, lw=1.2)
        ax.plot(kt_mask, altitude, "s--", color="#cc3333", label="kintera SS", ms=3, lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("mixing ratio")
        ax.set_ylabel("altitude (km)")
        ax.set_title(sp)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        panel_imgs[sp] = fig_to_base64(fig)
        plt.close(fig)

    # ---- Compose HTML ----
    parts = []
    parts.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>moses00 SS: kintera vs KB</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         margin: 24px; max-width: 1400px; color: #222; }
  h1 { border-bottom: 2px solid #444; padding-bottom: 8px; }
  h2 { margin-top: 32px; border-bottom: 1px solid #aaa; padding-bottom: 4px; }
  table.summary { border-collapse: collapse; font-size: 0.9em; margin-top: 12px; }
  table.summary th, table.summary td { border: 1px solid #bbb;
         padding: 4px 8px; text-align: right; }
  table.summary th { background: #f0f0f0; font-weight: 600; }
  table.summary td.sp { text-align: left; font-weight: 600; background: #fafafa; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 12px; }
  .panel img { max-width: 100%; }
  .legend-color { display: inline-block; width: 12px; height: 12px;
         vertical-align: middle; margin-right: 4px; border: 1px solid #999; }
  .meta { color: #666; font-size: 0.9em; }
</style></head>
<body>
""")
    parts.append(f"<h1>moses00 steady state: kintera vs KB fort.7</h1>")
    parts.append(
        f"<p class='meta'>Generated 2026-05-31. "
        f"Reference: <code>{KB_SS_PATH.name}</code>. "
        f"kintera SS dump: <code>{KT_SS_PATH.name}</code>. "
        f"Species: {len(species)}, altitudes: {nlyr}.</p>"
    )
    parts.append("<h2>Ratio kintera/KB at key altitudes (mixing ratio)</h2>")
    parts.append(
        "<p><span class='legend-color' style='background:#1c7f1c'></span> 0.5–2× &nbsp;"
        "<span class='legend-color' style='background:#b06a00'></span> 0.2–5× &nbsp;"
        "<span class='legend-color' style='background:#a01010'></span> &lt;0.2 or &gt;5×</p>"
    )
    parts.append("<table class='summary'>")
    parts.append("<tr><th>species</th>")
    for L in SUMMARY_LAYERS:
        parts.append(f"<th>L{L}<br>z={altitude[L]:.0f} km</th>")
    parts.append("</tr>")
    for sp, cells in rows:
        parts.append(f"<tr><td class='sp'>{sp}</td>")
        for kb_v, kt_v, ratio_str, color in cells:
            parts.append(
                f"<td style='color:{color}'>"
                f"<small>{kb_v:.1e} → {kt_v:.1e}</small><br><b>{ratio_str}</b></td>"
            )
        parts.append("</tr>")
    parts.append("</table>")

    parts.append("<h2>Mixing-ratio altitude profiles</h2>")
    parts.append("<div class='grid'>")
    for sp in PANEL_SPECIES:
        if sp in panel_imgs:
            parts.append(
                f"<div class='panel'><img src='data:image/png;base64,{panel_imgs[sp]}'/></div>"
            )
    parts.append("</div>")
    parts.append("</body></html>")

    OUT_HTML.write_text("".join(parts))
    print(f"Wrote {OUT_HTML} ({OUT_HTML.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
