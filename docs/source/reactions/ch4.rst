Methane
========

Reaction::

   CH4(g) <=> CH4(condensed)

All plotted functions return natural log Q, with each partial pressure expressed numerically in Pa. The net pressure exponent is 1; condensed-phase activity is one. See :doc:`../equilibrium_algorithms` for the signed quotient and H2 treatment.

Data, provenance and limitations
--------------------------------

The ideal branches switch at 90.67 K. No uncertainty or validity range for the legacy beta/gamma coefficients has been verified.

The coefficient audit was performed on 2026-10-07. Primary data links:

* `Prydz and Goodwin (1972), NIST Antoine table <https://webbook.nist.gov/cgi/cbook.cgi?ID=C74828&Mask=4&Type=ANTOINE&Plot=on>`_

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
   * - ch4_ideal
     - ideal
     - [90.67, 11690, 10.15, 2.1, 10.41, 0.9]
     - 70–189.99
     - legacy; coefficient provenance unresolved
   * - NIST liquid
     - antoine
     - [3.9895, 443.028, -0.49]
     - 90.99–189.99
     - NIST WebBook Antoine table

.. figure:: ../_static/reactions/ch4-comparison.svg
   :alt: Coefficient curves and their log pressure-quotient ratios

   The ratio reference is NIST liquid. Curves are compared only where their displayed intervals overlap; disagreement is not an uncertainty estimate.

.. list-table::
   :header-rows: 1

   * - Formula
     - T (K)
     - ln Q (Pa convention)
   * - ch4_ideal
     - 70
     - 6.525422938
   * - ch4_ideal
     - 129.995
     - 11.68041631
   * - ch4_ideal
     - 189.99
     - 13.11908298
   * - NIST liquid
     - 90.99
     - 9.427158654
   * - NIST liquid
     - 140.49
     - 13.41259106
   * - NIST liquid
     - 189.99
     - 15.31592422

:download:`Numerical comparison CSV <../_static/reactions/ch4-values.csv>`

Native formula evaluation agrees with the offline coefficient record as follows (float64; mathematical transcription checks, not physical uncertainty):

.. list-table::
   :header-rows: 1

   * - Formula
     - Device
     - Maximum absolute ln Q difference
   * - ch4_ideal
     - cpu
     - 1.776e-15
   * - ch4_ideal
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
     - 1.600e-07
     - 9.537e-08
     - 7.574e-08
     - pass
   * - UV
     - nucleate
     - 4
     - 9.537e-08
     - 3.421e-07
     - 9.537e-08
     - 3.584e-08
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
     - 3.421e-07
     - 9.537e-08
     - 3.584e-08
     - pass
   * - UV
     - trace_h2
     - 4
     - 9.537e-08
     - 1.600e-07
     - 9.537e-08
     - 7.574e-08
     - pass
   * - UV
     - absent_reactants
     - 4
     - 9.537e-08
     - 0.000e+00
     - 9.537e-08
     - 1.097e-07
     - pass
   * - UV
     - rich_h2
     - 4
     - 9.537e-08
     - 1.600e-07
     - 9.537e-08
     - 7.574e-08
     - pass

.. figure:: ../_static/reactions/ch4-validation.svg
   :alt: Native and independent scalar-reference cloud amounts

   Each marker is a recorded native solve. TP amounts are restored using conserved helium; UV amounts are concentrations.

Reproduce this reaction independently::

   python docs/reaction_validation.py --reaction ch4 --cuda --output /tmp/ch4.json
   pytest -q tests/test_reaction_equilibrium.py -k ch4
