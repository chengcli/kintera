# Titan C2 photochemistry for the snapy GCM (kintera × py2sess)

Reusable chemistry assets that let the snapy GCM stack (see `~/dev/UM-TITAN`)
run the validated **moses05.C2** Titan photochemical network (15 C2-chain
species + N2) as passive scalars, with photolysis driven by **py2sess**
two-stream radiative transfer — CPU and CUDA end-to-end.

## Pipeline

```
KINETICS-base moses05 pun ──make_c2_network.py──► titan_c2_chem.yaml   (42 arrhenius + 4 Troe falloff)
 (dev kintera, py3.10)                            titan_c2_data.npz    (photo σ/flux, absorbers, atm, custom rates)
                                                  c2_ref_rates.npz     (dev reference for the gate)
                                                        │
                                   validate_c2_network.py (pyenv, Gate A: ≤1e-6)
                                                        │
            titan_c2_chem.py  ◄─────────────────────────┘
            (TitanC2Chemistry: kintera Kinetics + 11 vendored KB
             UPDATE_CHEMB custom rates + photolysis J; implicit advance)
            py2sess_rt.py
            (TitanC2Radiation: τ from absorbers → forward_flux → actinic flux)
                                                        │
            UM-TITAN/titan/run_titan_c2.py  (3D cubed-sphere case)
```

## Key implementation notes

- **Old-kintera compatibility**: snapy's pyenv ships kintera 0.0.0 (no
  `kinetics_base` package, no `kb_falloff`). The generator emits YAML the old
  API accepts: KB falloff maps EXACTLY onto `type: falloff` + simplified-Troe
  with synthetic `Fc≡0.6`; the compiled default `Tref=300 K` (the yaml
  `reference-state` is not propagated into rate options in that version) is
  folded into the pre-exponentials (`A_yaml = A_kb·300^b`).
- **11 KB `UPDATE_CHEMB` reactions** (caps / zkcalc forms, e.g. 2H+M→H2,
  CH3+CH3+M→C2H6) are not YAML-expressible; they are vendored verbatim from
  `python/kinetics_base/titan/chemb_overrides.py` into `titan_c2_chem.py` and
  validated through the same gate.
- **Implicit step**: batched pure-torch backward-Euler
  (`evolve_implicit_torch`, 16×16 `linalg.solve`); kintera's compiled CUDA
  `evolve_implicit` kernel requests more shared memory than available at
  73 reactions × 16 species.
- **Actinic flux**: Titan's UV photolysis region is absorption-dominated, so
  the per-column slant attenuation is folded into the optical depth
  (`exp(−τ/μ0)` ≡ overhead attenuation of `τ/μ0`) and ALL day columns go
  through ONE batched `forward_flux` call — exact in the ssa→0 limit and the
  key to GPU throughput. A fixed per-bin transmission for the atmosphere
  ABOVE the model lid (from the moses05 profile) multiplies the TOA flux.

## Gates (all must pass)

| gate | what | tolerance |
|---|---|---|
| A1 | per-reaction rate, pyenv runtime vs dev KINETICS-base, 91 Titan levels | ≤ 1e-6 (measured 3e-11) |
| A2 | total thermal dC/dt per species | ≤ 1e-8 |
| A3 | unattenuated TOA photolysis J | ≤ 1e-6 (3e-16) |
| B0 | torch implicit step ≡ kintera.evolve_implicit (residual) | 1e-10·|b| |
| B1 | CPU vs CUDA rates/jac/advance | 1e-12 / 1e-7 |
| B2 | 0-D box: C/H-atom conservation over 1000 h forcing | < 1e-4 (clamp-limited) |
| C1 | actinic flux vs analytic Beer–Lambert | ≤ 2 % / bin |
| C2 | CPU vs CUDA actinic flux | 1e-10 |

## Commands

```bash
# regenerate + validate the network (dev environment)
/opt/anaconda3/bin/python3.10 examples/snapy_titan_c2/make_c2_network.py

# install the RT dependency into snapy's env (one-time)
~/pyenv/bin/pip install "py2sess[torch]"

# run all gates
cd examples/snapy_titan_c2
VIRTUAL_ENV=~/pyenv ~/pyenv/bin/python -m pytest test_titan_c2.py -v

# the 3D case lives in the UM-TITAN repo:
cd ~/dev/UM-TITAN/titan
VIRTUAL_ENV=~/pyenv ~/pyenv/bin/python -u run_titan_c2.py -c titan_c2_dry.yaml --output-dir ./out_c2
torchrun --nproc_per_node=8 run_titan_c2.py -c titan_c2_dry.8gpu.yaml --output-dir ./out_c2_8gpu

# report (figures + HTML)
VIRTUAL_ENV=~/pyenv ~/pyenv/bin/python make_report.py \
    --run-dir ./out_c2 --config ~/dev/UM-TITAN/titan/titan_c2_dry.yaml
```

## Performance (one cubed-sphere block: 2916 columns × 40 layers × 176 λ, H100)

| | CPU | GPU | speedup |
|---|---|---|---|
| RT (py2sess actinic flux) | 474 ms | 22 ms | **21×** |
| chemistry (73 rxns, one implicit solve) | 1565 ms | 27 ms | **58×** |
| total per step | 2038 ms | 50 ms | **41×** |

The chemistry row is **one** implicit solve. The adaptive ROS2 integrator takes
several sub-steps per hydro step (one per stiff cell, via compaction), so the
real per-step chemistry cost scales with the sub-step count — see
[INTEGRATOR.md](INTEGRATOR.md) for the measured GCM breakdown and tuning.

## Chemistry integrator

A per-cell adaptive, compacting **ROS2** integrator (2nd-order, L-stable,
element-conserving). Tuning — the `chem_rtol` / `max_substeps` knobs, the
stiffness cliff, and how the recommendation changes with the dynamics grid —
is documented in **[INTEGRATOR.md](INTEGRATOR.md)**. Short version: the default
`chem_rtol = 0.1` is a coarse-grid (dt≈66 s) choice; tighten to 0.05 on finer
grids, where chemistry stops being the bottleneck.

## Caveats

- The positivity clamp is the only source of atom drift (ROS2 itself is exactly
  element-conserving); it fires when a stiff radical's linearized increment
  overshoots below zero, and is negligible at GCM dt.
- The dense lower-boundary cells (~100 km) are floor-limited at the current grid
  (~7–13% error there, below the 150–300 km science region) — see INTEGRATOR.md.
- py2sess is GPL-3.0 — used as a pip runtime dependency of this example only
  (no code vendored).
- Pure absorption (ssa=0): adding Titan haze scattering needs a
  mean-intensity output from the RT, not the geometry-folding shortcut.
