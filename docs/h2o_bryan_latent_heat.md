# h2o_bryan: latent heat at 273.15 K

`h2o_bryan` (`src/vapors/vapor_functions.h`) is the saturation vapor pressure
used for the Bryan and Fritsch (2002, MWR 130, appendix) moist benchmark. That
case takes a constant-capacity latent heat with L_v0 = 2.5e6 J/kg at
T_0 = 273.15 K. Up to 192d724, `h2o_bryan` reused the liquid beta of
`h2o_ideal` (24.845). That beta gives L(273.15) about 0.15 % too high.

## Kirchhoff form

Both functions use the same ideal form, with t = T / t_r:

    ln p(T) = ln p_r + beta (1 - t_r / T) - delta ln(T / t_r)
    d ln p / dT = beta t_r / T^2 - delta / T

Clausius-Clapeyron with an ideal vapor, L = R_v T^2 d ln p / dT, gives

    L(T) = R_v (beta t_r - delta T)

This is linear in T with slope -R_v delta = c_pv - c_l, so beta fixes L at
one temperature and delta fixes the slope.
`h2o_bryan` has t_r = 273.16 K, p_r = 611.7 Pa and delta = 4.986009.

## R_v

kintera has no water-specific constant. It builds R_v from the gas constant
and the harp atomic weights that `molar_mass` uses:

| quantity | value | source |
|---|---|---|
| R | 8.31446 J/(mol K) | `src/constants.h:6` (`constants::Rgas`) |
| M(H2O) | 2 x 1.008 + 15.999 = 18.015 g/mol | pyharp 2.6.5 `harp/element.cpp:54,61` via `kintera::molar_mass` (`src/utils/molar_mass.cpp:17`) |
| R_v | 461.5298362475715 J/(kg K) | R / M(H2O) |

## beta

    beta = (L_v0 / R_v + delta T_0) / t_r = 24.815845124764618

| beta | L(273.15) [J/kg] | L / 2.5e6 - 1 |
|---|---|---|
| 24.845 (old, = h2o_ideal liquid) | 2503675.599 | +1.470e-3 |
| 24.816 | 2500019.5 | +7.8e-6 |
| 24.81585 | 2500000.6 | +2.5e-7 |
| 24.815845 (new) | 2499999.984 | -6.3e-9 |

At least five decimals are needed to reach 1e-6. The code uses 24.815845.
`h2o_ideal` keeps its own literal 24.845 and is bitwise unchanged
(`VaporFunctions.h2o_ideal_is_unchanged`).

## SVP against Bolton (1980)

The benchmark's moisture formulas use Bolton's
e_s = 611.2 exp(17.67 (T - 273.15) / (T - 29.65)) Pa.

| T [K] | Bolton [Pa] | old h2o_bryan [Pa] | old diff | new h2o_bryan [Pa] | new diff |
|---|---|---|---|---|---|
| 250 | 95.4891 | 95.2348 | -0.266 % | 95.4924 | +0.003 % |
| 290 | 1917.9970 | 1921.1636 | +0.165 % | 1917.9138 | -0.004 % |
| 300 | 3534.5197 | 3539.4575 | +0.140 % | 3530.2372 | -0.121 % |

The values come from compiling the header before and after the change and
evaluating `exp(h2o_bryan(T))` directly. p_r = 611.7 Pa at 273.16 K is
unchanged; Bolton gives 611.64 Pa there (+0.009 %). The cause of the
remaining -0.12 % at 300 K has not been analysed here. It is not the
reference pressure.

## Changed expectations

Three existing expectations pin the h2o_bryan constant, not the physics.
They change in the same commit as the header. Every other test is
unchanged, and `h2o_ideal` is bitwise identical.

1. `tests/test_vapor_functions.cpp`, `h2o_bryan_expected`:
   beta 24.845 -> 24.815845. Values at the tested temperatures (ln Pa):

   | T [K] | old | new |
   |---|---|---|
   | 250 | 4.556345539765576 | 4.559046458965575 |
   | 273.16 | 6.416241966248519 | 6.416241966248519 |
   | 289.85 | 7.551155054063129 | 7.549476265206824 |
   | 300 | 8.171728750030239 | 8.169120349363572 |

2. `tests/test_vapor_functions.cpp`, `h2o_bryan_ddT_expected`:
   beta 24.845 -> 24.815845. Values (1/K):

   | T [K] | old | new |
   |---|---|---|
   | 250 | 0.0886425272 | 0.08851510352320001 |
   | 273.16 | 0.07270094816224923 | 0.07259421584419387 |
   | 289.85 | 0.0635790182569613 | 0.06348422366961026 |
   | 300 | 0.058787305555555565 | 0.058698816891111116 |

3. `docs/reaction_catalogue.py`, `h2o_bryan` entry, checked by
   `tests/test_reaction_equilibrium.py::test_documented_native_curves`:
   `[273.16, 611.7, 24.845, 4.986009, 24.845, 4.986009]` ->
   `[273.16, 611.7, 24.815845, 4.986009, 24.815845, 4.986009]`.
   The documented ln Q values change as follows:

   | T [K] | old | new |
   |---|---|---|
   | 230 | 2.6115095315005963 | 2.6169805306310305 |
   | 266.5 | 5.918421989930005 | 5.919150591430943 |
   | 303 | 8.346098495661849 | 8.34322725737802 |

These values are float64 evaluations of the helper formula.

## Generated pages

`docs/plot_reaction_comparisons.py --all` regenerates
`docs/source/reactions/h2o.rst`, `docs/source/_static/reactions/h2o-values.csv`
and `h2o-comparison.svg` from the catalogue. With numpy 2.4.4 and
matplotlib 3.10.8 (the recorded environment) it reproduces every page at
192d724 byte for byte. On the new catalogue, only the h2o_bryan parameter row
and the h2o_bryan ln Q values change in the rst and csv. The SVG is a single
figure, so the lower panel's axis range moves with the h2o_bryan ratio curve.

`validation.json` is stale for h2o_bryan only: it fingerprints
`src/vapors/vapor_functions.h`, and only its two h2o_bryan `native_curves`
entries (cpu, cuda) depend on this constant, since the equilibrium cases use
a manufactured inline curve. Re-run `python docs/reaction_validation.py --cuda`
to refresh it. No CI or pre-commit gate runs `--check`.

Downstream inputs that set u0_R = -beta t_r for this curve (for example
snapy `bryan.yaml`) need the new beta.
