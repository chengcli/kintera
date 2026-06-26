# Paper figure — Chapman cycle kinetics validation

`fig_chapman_match.{png,pdf}` (300-dpi PNG + vector PDF).

kintera's `Photolysis` + `Arrhenius` modules compute the photolysis rates J(z)
and rate constants k(z) on a stratospheric column (the TOA solar flux is
attenuated through the overhead O2 column); the Chapman steady state obtained by
kintera's implicit solver (`evolve_implicit`) is compared to the closed-form
analytic Chapman steady state of the same J, k. The panel shows the O3/O
profiles (the ozone layer) with kintera markers on the analytic curves; the
inset reports the max relative difference (~1e-12).

```
/opt/anaconda3/bin/python3.10 paper_figures/fig_chapman_match.py
```

Note: idealized O2/O3 cross-sections (strong Schumann-Runge absorbed high in
the mesosphere + a weak Herzberg continuum that makes the stratospheric ozone
layer near 25 km); the point is the kintera-vs-analytic agreement, not a
quantitative ozone climatology.

The Titan validation figures (kintera vs KINETICS-base + Cassini/Huygens data)
live in the **UM-TITAN** repo under `paper_figures/`.
