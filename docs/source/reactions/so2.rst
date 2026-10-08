Sulfur dioxide
==============

Reaction::

   SO2(g) <=> SO2(condensed)

All plotted functions return natural log Q, with each partial pressure expressed numerically in Pa. The net pressure exponent is 1; condensed-phase activity is one. See :doc:`../equilibrium_algorithms` for the signed quotient and H2 treatment.

Data, provenance and limitations
--------------------------------

The registered function is only the low-temperature fit. It does not select the high-temperature branch automatically.

The coefficient audit was performed on 2026-10-07. Primary data links:

* `Stull (1947), NIST Antoine table <https://webbook.nist.gov/cgi/cbook.cgi?ID=C7446095&Mask=4&Type=ANTOINE&Plot=on>`_

Legacy values are transcribed from ``src/vapors/vapor_functions.h``. PR-only candidates are transcribed from PR 108, revision ``17ba9fb88bf1e50b06b5721b9bb604a1a0691395``. An unresolved provenance label means the fit has not been certified by this audit.

Coefficient conventions
-----------------------

``ideal`` parameters are [T3, P3, beta_liquid, gamma_liquid, beta_solid, gamma_solid]: L = ln(P3) + beta(1 - T3/T) - gamma ln(T/T3). The solid branch is selected at T <= T3. ``antoine`` parameters are [A, B, C]: L = ln(100000) + ln(10)(A - B/(T+C)). ``linear`` parameters are [A, B, base]: L = ln(base)(A - B/T). Temperatures and B/C offsets are in kelvin. ``table`` contains [temperatures, ln Q] with linear interpolation used only for display.

.. list-table::
   :header-rows: 1

   * - Formula
     - Form
     - Parameters
     - Plot interval (K)
     - Status
   * - so2_antoine
     - antoine
     - [3.48586, 668.225, -72.252]
     - 177.7–263
     - NIST WebBook Antoine table
   * - NIST high T (comparison only)
     - antoine
     - [4.37798, 966.575, -42.071]
     - 263–414.9
     - NIST WebBook Antoine table

.. figure:: ../_static/reactions/so2-comparison.svg
   :alt: Coefficient curves and their log pressure-quotient ratios

   The ratio reference is so2_antoine. Curves are compared only where their displayed intervals overlap; disagreement is not an uncertainty estimate.

.. list-table::
   :header-rows: 1

   * - Formula
     - T (K)
     - ln Q (Pa convention)
   * - so2_antoine
     - 177.7
     - 4.947910643
   * - so2_antoine
     - 220.35
     - 9.1500447
   * - so2_antoine
     - 263
     - 11.4730396
   * - NIST high T (comparison only)
     - 263
     - 11.51967643
   * - NIST high T (comparison only)
     - 338.95
     - 14.0968687
   * - NIST high T (comparison only)
     - 414.9
     - 15.62404741

:download:`Numerical comparison CSV <../_static/reactions/so2-values.csv>`

Native formula evaluation agrees with the offline coefficient record as follows (float64; mathematical transcription checks, not physical uncertainty):

.. list-table::
   :header-rows: 1

   * - Formula
     - Device
     - Maximum absolute ln Q difference
   * - so2_antoine
     - cpu
     - 0.000e+00
   * - so2_antoine
     - cuda
     - 1.776e-15

Isolated equilibrium validation
-------------------------------

This reaction is tested independently with its balanced stoichiometry and explicit gas/solid phases. The native TP and UV solvers are compared with a 100-step bracketed scalar extent solve. Manufactured caloric data and a consistent inline equilibrium curve isolate solver correctness; these results do **not** validate the physical fit above. See the algorithm page for the exact construction and acceptance thresholds.

.. list-table::
   :header-rows: 1

   * - Mode
     - Initial state
     - Runs
     - Max state error
     - Max residual violation
     - Max conservation error
     - Max energy error
     - Status
   * - TP
     - condense
     - 4
     - 5.672e-08
     - 2.852e-07
     - 1.673e-08
     - 0.000e+00
     - pass
   * - TP
     - nucleate
     - 4
     - 5.651e-08
     - 2.841e-07
     - 1.743e-08
     - 0.000e+00
     - pass
   * - TP
     - evaporate
     - 4
     - 1.116e-09
     - 0.000e+00
     - 1.116e-09
     - 0.000e+00
     - pass
   * - TP
     - clear
     - 4
     - 2.324e-10
     - 0.000e+00
     - 2.324e-10
     - 0.000e+00
     - pass
   * - TP
     - zero_h2
     - 4
     - 5.651e-08
     - 2.841e-07
     - 1.743e-08
     - 0.000e+00
     - pass
   * - TP
     - trace_h2
     - 4
     - 5.672e-08
     - 2.852e-07
     - 1.673e-08
     - 0.000e+00
     - pass
   * - TP
     - absent_reactants
     - 4
     - 6.840e-10
     - 0.000e+00
     - 6.840e-10
     - 0.000e+00
     - pass
   * - TP
     - rich_h2
     - 4
     - 5.672e-08
     - 2.852e-07
     - 1.673e-08
     - 0.000e+00
     - pass
   * - UV
     - condense
     - 4
     - 1.907e-07
     - 2.547e-07
     - 1.907e-07
     - 1.193e-07
     - pass
   * - UV
     - nucleate
     - 4
     - 9.537e-08
     - 1.861e-07
     - 9.537e-08
     - 1.123e-07
     - pass
   * - UV
     - evaporate
     - 4
     - 9.537e-08
     - 0.000e+00
     - 9.537e-08
     - 9.875e-08
     - pass
   * - UV
     - clear
     - 4
     - 1.341e-10
     - 0.000e+00
     - 1.341e-10
     - 6.117e-08
     - pass
   * - UV
     - zero_h2
     - 4
     - 9.537e-08
     - 1.861e-07
     - 9.537e-08
     - 1.123e-07
     - pass
   * - UV
     - trace_h2
     - 4
     - 1.907e-07
     - 2.547e-07
     - 1.907e-07
     - 1.193e-07
     - pass
   * - UV
     - absent_reactants
     - 4
     - 9.537e-08
     - 0.000e+00
     - 9.537e-08
     - 1.083e-07
     - pass
   * - UV
     - rich_h2
     - 4
     - 1.907e-07
     - 2.547e-07
     - 1.907e-07
     - 1.193e-07
     - pass

.. figure:: ../_static/reactions/so2-validation.svg
   :alt: Native and independent scalar-reference cloud amounts

   Each marker is a recorded native solve. TP amounts are restored using conserved helium; UV amounts are concentrations.

Reproduce this reaction independently::

   python docs/reaction_validation.py --reaction so2 --cuda --output /tmp/so2.json
   pytest -q tests/test_reaction_equilibrium.py -k so2
