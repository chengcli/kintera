"""Parse KB's REACTION RATES section from kinetics.out log and compare
to kintera's per-reaction rates at the same state (KB fort.7).

KB log format (around line 119241+):
    REACTION RATES:
            PRESSURE   RATE(  1)  RATE(  2)  ...  RATE( 10)
       1)  7.645E+01   0.000E+00  2.802E-05  ...
       ...
      91)  ...
            PRESSURE   RATE( 11)  RATE( 12)  ...  RATE( 20)
       1)  ...
       ...

Each batch is 10 reactions × 91 altitudes. KB has 521 reactions so 53 batches.
"""
from __future__ import annotations
import sys
import re
from pathlib import Path
import numpy as np

KB_LOG = Path("/tmp/kb_m00/kinetics.out-revert")
NLYR = 91
NREACT = 521
# KB output rows are 1-indexed and start at atmospheric layer NALT1=10 (kintera
# 0-indexed L9). So KB row N maps to kintera 0-indexed L(N+8). The output array
# is sized NLYR but with first 9 entries zero (KB doesn't simulate L0-L8).
NALT1_KB_TO_KT_OFFSET = 8  # KB row N → kintera 0-indexed L(N + 8)

assert KB_LOG.exists(), f"need {KB_LOG}"

with KB_LOG.open() as f:
    lines = f.readlines()

# Find "REACTION RATES:" line
start = None
for i, ln in enumerate(lines):
    if "REACTION RATES:" in ln:
        start = i
        break

if start is None:
    raise SystemExit("REACTION RATES: section not found in log")
print(f"REACTION RATES section starts at line {start+1}")

# Parse batches. Each batch:
#   header: "        PRESSURE   RATE(  N1)  RATE(  N2) ..."
#   91 data rows: "  K)  P    R1 R2 R3 ... R10"
rates = np.zeros((NREACT, NLYR), dtype=np.float64)
header_re = re.compile(r'\s*PRESSURE\s+(RATE\(\s*\d+\))')
rate_re = re.compile(r'RATE\(\s*(\d+)\)')
data_re = re.compile(r'^\s*(\d+)\)\s+([-+0-9.eE]+)\s+(.*)$')

i = start + 1
batches_parsed = 0
last_header_line = start
while i < len(lines):
    ln = lines[i]
    # New section break? Stop if we see "PRODUCTION AND LOSS" or end markers
    if "PRODUCTION" in ln or "PROD AND LOSS" in ln or "FORTRAN STOP" in ln:
        print(f"  hit section break at line {i+1}: {ln.strip()[:60]}")
        break
    if "PRESSURE" in ln and "RATE(" in ln:
        ids_in_batch = [int(m.group(1)) for m in rate_re.finditer(ln)]
        if not ids_in_batch:
            i += 1
            continue
        last_header_line = i
        i += 1
        rows_read = 0
        while rows_read < NLYR and i < len(lines):
            m = data_re.match(lines[i])
            if m is None:
                # Stop if next batch header
                if "PRESSURE" in lines[i] and "RATE(" in lines[i]:
                    break
                i += 1
                continue
            lyr_idx = int(m.group(1))
            rest = m.group(3)
            values = []
            for tok in rest.split():
                try:
                    values.append(float(tok))
                except ValueError:
                    break
            if len(values) >= len(ids_in_batch):
                for j, rid in enumerate(ids_in_batch):
                    rates[rid - 1, lyr_idx - 1] = values[j]
            i += 1
            rows_read += 1
        batches_parsed += 1
        if batches_parsed % 10 == 0:
            print(f"  parsed {batches_parsed} batches, last reaction id = {ids_in_batch[-1]}")
    else:
        i += 1

print(f"Total batches parsed: {batches_parsed}")
print(f"Non-zero rates: {(rates != 0).sum()} / {rates.size}")
print(f"Max rate: {rates.max():.3e} at reaction {rates.argmax() // NLYR + 1} layer {rates.argmax() % NLYR + 1}")

np.savez("/tmp/kb_moses00_rates.npz", rates=rates, nreact=NREACT, nlyr=NLYR)
print(f"\nSaved to /tmp/kb_moses00_rates.npz: shape {rates.shape}")
