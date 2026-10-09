Carbon dioxide
==============

Reaction::

   CO2(g) <=> CO2(condensed)

All plotted functions return natural log Q, with each partial pressure expressed numerically in Pa. The net pressure exponent is 1; condensed-phase activity is one. See :doc:`../equilibrium_algorithms` for the signed quotient and H2 treatment.

Data, provenance and limitations
--------------------------------

C = -3.494 K replaces the erroneous -34.94 K in both value and derivative. This is the solid sublimation temperature range.

The coefficient audit was performed on 2026-10-07. Primary data links:

* `Giauque and Egan (1937), NIST Antoine table <https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Mask=4&Type=ANTOINE&Plot=on>`_

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
   * - co2_antoine
     - antoine
     - [6.81228, 1301.679, -3.494]
     - 154.26–195.89
     - NIST WebBook Antoine table
   * - old transcription (rejected)
     - antoine
     - [6.81228, 1301.679, -34.94]
     - 154.26–195.89
     - incorrect C; comparison only

.. figure:: ../_static/reactions/co2-comparison.svg
   :alt: Coefficient curves and their log pressure-quotient ratios

   The ratio reference is co2_antoine. Curves are compared only where their displayed intervals overlap; disagreement is not an uncertainty estimate.

.. list-table::
   :header-rows: 1

   * - Formula
     - T (K)
     - ln Q (Pa convention)
   * - co2_antoine
     - 154.26
     - 7.318789252
   * - co2_antoine
     - 175.075
     - 9.730489873
   * - co2_antoine
     - 195.89
     - 11.62035482
   * - old transcription (rejected)
     - 154.26
     - 2.079548689
   * - old transcription (rejected)
     - 175.075
     - 5.810642252
   * - old transcription (rejected)
     - 195.89
     - 8.576681916

:download:`Numerical comparison CSV <../_static/reactions/co2-values.csv>`

Native formula evaluation agrees with the offline coefficient record as follows (float64; mathematical transcription checks, not physical uncertainty):

.. list-table::
   :header-rows: 1

   * - Formula
     - Device
     - Maximum absolute ln Q difference
   * - co2_antoine
     - cpu
     - 0.000e+00
   * - co2_antoine
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
     - 9.537e-08
     - 7.484e-07
     - 9.537e-08
     - 1.672e-07
     - pass
   * - UV
     - nucleate
     - 4
     - 9.537e-08
     - 2.216e-07
     - 9.537e-08
     - 1.433e-07
     - pass
   * - UV
     - evaporate
     - 4
     - 9.537e-10
     - 0.000e+00
     - 9.537e-10
     - 6.983e-08
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
     - 2.216e-07
     - 9.537e-08
     - 1.433e-07
     - pass
   * - UV
     - trace_h2
     - 4
     - 9.537e-08
     - 7.484e-07
     - 9.537e-08
     - 1.672e-07
     - pass
   * - UV
     - absent_reactants
     - 4
     - 1.788e-09
     - 0.000e+00
     - 1.788e-09
     - 5.073e-08
     - pass
   * - UV
     - rich_h2
     - 4
     - 9.537e-08
     - 7.484e-07
     - 9.537e-08
     - 1.672e-07
     - pass

.. figure:: ../_static/reactions/co2-validation.svg
   :alt: Native and independent scalar-reference cloud amounts

   Each marker is a recorded native solve. TP amounts are restored using conserved helium; UV amounts are concentrations.

Reproduce this reaction independently::

   python docs/reaction_validation.py --reaction co2 --cuda --output /tmp/co2.json
   pytest -q tests/test_reaction_equilibrium.py -k co2
