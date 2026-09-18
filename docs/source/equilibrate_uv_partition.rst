UV Saturation Adjustment by Partitioning
=========================================

The ``partition`` branch of ``equilibrate_uv`` adjusts composition and
temperature at fixed volume and internal-energy density. It applies when each
nucleation reaction contains exactly one vapor and one cloud species and no
species occurs in more than one such reaction. Other species keep their initial
concentrations during the adjustment.

Conserved inventories
---------------------

For reaction :math:`j`, write its stoichiometry as

.. math::

   a_j V_j \longrightarrow b_j C_j, \qquad a_j,b_j>0.

Let :math:`c_s^0` be the initial molar concentration of species :math:`s`
after any negative input concentrations have been clamped to zero. The
reaction-invariant inventory is

.. math::

   N_j = c_{V_j}^0 + \frac{a_j}{b_j}c_{C_j}^0.

Thus the reaction can redistribute material between :math:`V_j` and
:math:`C_j` without changing :math:`N_j`.

Partition at a trial temperature
--------------------------------

Let :math:`S_j(T)` be the value returned by reaction :math:`j`'s ``logsvp``
function, and let :math:`R` be the universal gas constant. The code evaluates

.. math::

   \begin{aligned}
   c_{V_j}^{\mathrm{sat}}(T)
     &= \frac{\exp[S_j(T)/a_j]}{RT}, \\
   c_{V_j}(T)
     &= \min\!\left(N_j,c_{V_j}^{\mathrm{sat}}(T)\right), \\
   c_{C_j}(T)
     &= \frac{b_j}{a_j}\left[N_j-c_{V_j}(T)\right].
   \end{aligned}

If :math:`N_j=0`, both concentrations are set to zero. For the usual
one-to-one vapor-cloud reaction, :math:`S_j=\ln p_{\mathrm{sat},j}` and
:math:`c_{V_j}^{\mathrm{sat}}=p_{\mathrm{sat},j}/(RT)`. If the inventory is
below saturation, all of it remains vapor; otherwise, the excess forms cloud.
Each pair is partitioned independently at the same trial temperature.

Fixed-energy temperature
------------------------

For each species, Kintera evaluates the molar internal energy as

.. math::

   e_s(T,c_s)
     = u_{0,s} + c_{v,s}^{(0)}T
       + R\,e_s^{\mathrm{extra}}(T,c_s).

The partitioned state must reproduce the supplied internal-energy density
:math:`U_0`:

.. math::

   F(T) = \sum_s c_s(T)e_s\bigl(T,c_s(T)\bigr)-U_0 = 0.

The sum includes unpartitioned species, which retain their initial
concentrations but still contribute to the energy. The implementation uses
the following slope for its Newton step:

.. math::

   \begin{aligned}
   F'_{\mathrm{used}}(T)
     &= \sum_s c_s(T)
        \left[c_{v,s}^{(0)}+R\,c_{v,s}^{\mathrm{extra}}(T,c_s)\right] \\
     &\quad + \sum_{j:\,c_{V_j}<N_j}
        \frac{dc_{V_j}}{dT}
        \left[e_{V_j}-\frac{b_j}{a_j}e_{C_j}\right], \\
   \frac{dc_{V_j}}{dT}
     &= c_{V_j}\left[\frac{S'_j(T)}{a_j}-\frac{1}{T}\right]
        \quad\text{when }c_{V_j}<N_j.
   \end{aligned}

In the all-vapor branch, :math:`c_{V_j}=N_j` is constant and that pair's
redistribution term is zero. This slope is the exact derivative when the
provided extra-energy and extra-heat-capacity functions are thermodynamically
consistent and have no explicit concentration dependence; otherwise it is
the slope used by the implementation, not necessarily the full derivative.

Scalar solve and solver selection
---------------------------------

Starting from the temperature inferred from the input energy, the algorithm
brackets a sign change in :math:`F` by halving or doubling the temperature.
It then uses Newton steps inside the bracket, replacing an invalid or
out-of-bracket step with the midpoint. The convergence test is

.. math::

   |F(T)| \leq 64\,\epsilon_{\mathrm{machine}}
                  \max\left(|U_0|,1\right),

with at most ``max_iter`` iterations counting the initial evaluation. The
``ftol`` option applies to the KKT solver, not this partition energy test.
If the partition solve fails, it restores the pre-partition temperature and
concentrations (after nonnegative clamping). ``uv_solver: auto`` then tries
the KKT solver, whereas ``uv_solver: partition`` reports a convergence
warning without a fallback.

For the Uranus CH4 and H2S reactions, :math:`a_j=b_j=1` and the two
vapor-cloud pairs are disjoint, so this direct partition branch applies.
