Hydrogen sulfide
================

Reaction::

   H2S(g) <=> H2S(condensed)

All plotted functions return natural log Q, with each partial pressure expressed numerically in Pa. The net pressure exponent is 1; condensed-phase activity is one. See :doc:`../equilibrium_algorithms` for the signed quotient and H2 treatment.

Data, provenance and limitations
--------------------------------

Both Antoine C signs were corrected to match NIST. The legacy ideal solid branch repeats the liquid coefficients and remains unverified.

The coefficient audit was performed on 2026-10-07. Primary data links:

* `Stull (1947), NIST Antoine table <https://webbook.nist.gov/cgi/cbook.cgi?ID=C7783064&Mask=4&Type=ANTOINE&Plot=on>`_

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
   * - h2s_ideal
     - ideal
     - [187.63, 23300, 11.89, 5.04, 11.89, 5.04]
     - 138.8–349.5
     - legacy; coefficient provenance unresolved
   * - h2s_antoine low T
     - antoine
     - [4.43681, 829.439, -25.412]
     - 138.8–212.8
     - NIST WebBook Antoine table
   * - h2s_antoine high T
     - antoine
     - [4.52887, 958.587, -0.539]
     - 212.8–349.5
     - NIST WebBook Antoine table

.. figure:: ../_static/reactions/h2s-comparison.svg
   :alt: Coefficient curves and their log pressure-quotient ratios

   The ratio reference is h2s_antoine low T. Curves are compared only where their displayed intervals overlap; disagreement is not an uncertainty estimate.

.. list-table::
   :header-rows: 1

   * - Formula
     - T (K)
     - ln Q (Pa convention)
   * - h2s_ideal
     - 138.8
     - 7.392539901
   * - h2s_ideal
     - 244.15
     - 11.48162164
   * - h2s_ideal
     - 349.5
     - 12.42799129
   * - h2s_antoine low T
     - 138.8
     - 4.885530701
   * - h2s_antoine low T
     - 175.8
     - 9.02954825
   * - h2s_antoine low T
     - 212.8
     - 11.53708268
   * - h2s_antoine high T
     - 212.8
     - 11.54238265
   * - h2s_antoine high T
     - 281.15
     - 14.07524067
   * - h2s_antoine high T
     - 349.5
     - 15.61589127

:download:`Numerical comparison CSV <../_static/reactions/h2s-values.csv>`

Native formula evaluation agrees with the offline coefficient record as follows (float64; mathematical transcription checks, not physical uncertainty):

.. list-table::
   :header-rows: 1

   * - Formula
     - Device
     - Maximum absolute ln Q difference
   * - h2s_ideal
     - cpu
     - 3.553e-15
   * - h2s_antoine low T
     - cpu
     - 0.000e+00
   * - h2s_antoine high T
     - cpu
     - 0.000e+00
   * - h2s_ideal
     - cuda
     - 3.553e-15
   * - h2s_antoine low T
     - cuda
     - 1.776e-15
   * - h2s_antoine high T
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
     - 2.861e-07
     - 1.103e-06
     - 2.861e-07
     - 1.073e-07
     - pass
   * - UV
     - nucleate
     - 4
     - 9.537e-08
     - 7.212e-07
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
     - 9.382e-08
     - pass
   * - UV
     - zero_h2
     - 4
     - 9.537e-08
     - 7.212e-07
     - 9.537e-08
     - 8.678e-08
     - pass
   * - UV
     - trace_h2
     - 4
     - 2.861e-07
     - 1.103e-06
     - 2.861e-07
     - 1.073e-07
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
     - 2.861e-07
     - 1.103e-06
     - 2.861e-07
     - 1.073e-07
     - pass

.. figure:: ../_static/reactions/h2s-validation.svg
   :alt: Native and independent scalar-reference cloud amounts

   Each marker is a recorded native solve. TP amounts are restored using conserved helium; UV amounts are concentrations.

Reproduce this reaction independently::

   python docs/reaction_validation.py --reaction h2s --cuda --output /tmp/h2s.json
   pytest -q tests/test_reaction_equilibrium.py -k h2s
