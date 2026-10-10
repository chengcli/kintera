Water
========

Reaction::

   H2O(g) <=> H2O(condensed)

All plotted functions return natural log Q, with each partial pressure expressed numerically in Pa. The net pressure exponent is 1; condensed-phase activity is one. See :doc:`../equilibrium_algorithms` for the signed quotient and H2 treatment.

Data, provenance and limitations
--------------------------------

The ideal branches switch at 273.16 K. h2o_bryan continues the liquid branch below the triple point; it is not an ice fit. The displayed legacy interval is a plotting interval, not a verified validity range.

The coefficient audit was performed on 2026-10-07. Primary data links:

* `Bridgeman and Aldrich (1964), NIST Antoine table, 273–303 K <https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=4&Type=ANTOINE&Plot=on>`_

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
   * - h2o_ideal
     - ideal
     - [273.16, 611.7, 24.845, 4.986009, 22.98, 0.52]
     - 230–303
     - legacy; coefficient provenance unresolved
   * - h2o_bryan
     - ideal
     - [273.16, 611.7, 24.815845, 4.986009, 24.815845, 4.986009]
     - 230–303
     - legacy; coefficient provenance unresolved
   * - NIST liquid
     - antoine
     - [5.40221, 1838.675, -31.737]
     - 273–303
     - NIST WebBook Antoine table

.. figure:: ../_static/reactions/h2o-comparison.svg
   :alt: Coefficient curves and their log pressure-quotient ratios

   The ratio reference is NIST liquid. Curves are compared only where their displayed intervals overlap; disagreement is not an uncertainty estimate.

.. list-table::
   :header-rows: 1

   * - Formula
     - T (K)
     - ln Q (Pa convention)
   * - h2o_ideal
     - 230
     - 2.193423775
   * - h2o_ideal
     - 266.5
     - 5.854792938
   * - h2o_ideal
     - 303
     - 8.346098496
   * - h2o_bryan
     - 230
     - 2.616980531
   * - h2o_bryan
     - 266.5
     - 5.919150591
   * - h2o_bryan
     - 303
     - 8.343227257
   * - NIST liquid
     - 273
     - 6.40388033
   * - NIST liquid
     - 288
     - 7.431033685
   * - NIST liquid
     - 303
     - 8.344590271

:download:`Numerical comparison CSV <../_static/reactions/h2o-values.csv>`

Native formula evaluation agrees with the offline coefficient record as follows (float64; mathematical transcription checks, not physical uncertainty):

.. list-table::
   :header-rows: 1

   * - Formula
     - Device
     - Maximum absolute ln Q difference
   * - h2o_ideal
     - cpu
     - 5.329e-15
   * - h2o_bryan
     - cpu
     - 4.441e-15
   * - h2o_ideal
     - cuda
     - 5.329e-15
   * - h2o_bryan
     - cuda
     - 5.329e-15

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
     - 9.537e-08
     - 6.882e-07
     - 9.537e-08
     - 1.142e-07
     - pass
   * - UV
     - nucleate
     - 4
     - 9.537e-08
     - 7.256e-08
     - 9.537e-08
     - 8.678e-08
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
     - 9.537e-08
     - 0.000e+00
     - 9.537e-08
     - 9.456e-08
     - pass
   * - UV
     - zero_h2
     - 4
     - 9.537e-08
     - 7.256e-08
     - 9.537e-08
     - 8.678e-08
     - pass
   * - UV
     - trace_h2
     - 4
     - 9.537e-08
     - 6.882e-07
     - 9.537e-08
     - 1.142e-07
     - pass
   * - UV
     - absent_reactants
     - 4
     - 2.980e-10
     - 0.000e+00
     - 2.980e-10
     - 4.927e-08
     - pass
   * - UV
     - rich_h2
     - 4
     - 9.537e-08
     - 6.882e-07
     - 9.537e-08
     - 1.142e-07
     - pass

.. figure:: ../_static/reactions/h2o-validation.svg
   :alt: Native and independent scalar-reference cloud amounts

   Each marker is a recorded native solve. TP amounts are restored using conserved helium; UV amounts are concentrations.

Reproduce this reaction independently::

   python docs/reaction_validation.py --reaction h2o --cuda --output /tmp/h2o.json
   pytest -q tests/test_reaction_equilibrium.py -k h2o
