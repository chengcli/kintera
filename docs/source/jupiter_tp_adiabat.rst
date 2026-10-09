Deep-Jupiter TP adiabat with KCl and MgSiO3
===========================================

This runnable example constructs a one-dimensional, hydrostatic, reversible
adiabat for an **illustrative deep-Jupiter mixture**. It includes explicit H2
and He, KCl condensation, and
Mg + SiH4 + 3H2O <=> MgSiO3(s) + 5H2. Both condensates remain suspended;
there is no rainout, mixing, radiative transfer or external heat source.

The YAML defines the chemistry and column boundary conditions. The Python
script calls ``ThermoX.forward`` for every temperature trial, then finds the
temperature that preserves entropy per unit mass at each pressure level.
It writes the profile, diagnostics and plots. It does not use the UV solver.

Run the example
---------------

From the repository root, with the current kintera extension installed and
NumPy, PyYAML, Torch and Matplotlib available::

   python examples/example_jupiter_adiabat.py --output-dir examples/output/jupiter_adiabat
   pytest -q tests/test_jupiter_tp_adiabat.py

Optional arguments are ``--yaml PATH``, ``--levels N`` and ``--output-dir PATH``.
The default card spans 6000 to 200 bar, starting at 2300 K, on 121 logarithmically
spaced pressure levels. This domain keeps the resulting temperatures inside
800–2500 K. It covers the two deep cloud transitions, not the visible weather
layers or the entire planetary interior. The boundary state and abundances
are illustrative inputs, not a fit to observations.

.. figure:: _static/jupiter_adiabat/profile.svg
   :alt: Deep-Jupiter temperature, condensate mass fractions and saturation ratios versus pressure

   Reversible TP column. The clouds are retained and H2 evolves through the
   MgSiO3 reaction. Zero mass fractions are displayed at the plotting floor;
   no species floor is applied to the physical inventories.

Outputs and interpretation
--------------------------

``profile.csv`` contains pressure, temperature, height above the lower boundary,
bulk density, specific entropy, iteration counts, saturation ratios and
conservation errors. ``x_NAME`` is the mole fraction of all gas and condensed
species combined; it is **not** a gas-only mixing ratio. ``q_NAME`` is the mass
fraction of the total parcel. ``summary.json`` records maximum residuals and
the first sampled cloudy level. ``profile.svg`` and ``profile.png`` contain
the three-panel plot.

.. list-table:: Recorded default run
   :header-rows: 1

   * - Quantity
     - Value
   * - First sampled MgSiO3 cloud level
     - 4519.16 bar; 2114.25 K
   * - First sampled KCl cloud level
     - 314.760 bar; 957.561 K
   * - Maximum specific-entropy error
     - 9.99e-6 J kg^-1 K^-1
   * - Maximum relative element-inventory error
     - 2.85e-15
   * - Maximum relative mass error
     - 2.23e-15
   * - Maximum log saturation/complementarity residual
     - 9.73e-9

Cloud locations are grid samples at a 1e-12 cloud mole-fraction threshold,
not a refined root of the cloud-base condition. The reported error values are
numerical diagnostics for this approximation; they are not uncertainties on
Jupiter's cloud locations.

* :download:`Complete profile CSV <_static/jupiter_adiabat/profile.csv>`
* :download:`Recorded diagnostics JSON <_static/jupiter_adiabat/summary.json>`

Thermodynamic construction
--------------------------

At each new pressure, a bracketed scalar search calls TP equilibrium and solves

.. math::

   s(T,P,\boldsymbol{x}_{\rm eq})=s_0,\qquad
   s=\frac{S_{\rm volume}}{\rho},\qquad
   \rho=\sum_i c_i M_i.

The density and entropy include the retained solids. Conserving specific
entropy is necessary here: MgSiO3 formation changes the total number of
molecules, so holding entropy per mole fixed would describe a different path.
All temperature trials at one pressure restart from the same preceding parcel
composition, preserving its elemental inventory. The first He species is an
inert amount reference; H2 is a reacting gas, not a fixed reservoir.

For the ideal-gas card, concentrations are calculated directly as

.. math::

   c_i=\frac{x_i P}{RT\sum_{k\in g}x_k}.

This preserves trace SiH4 below the general ``TPX->V`` conversion's default
1e-20 mol/m³ floor. Such trace concentrations still matter in the reaction
quotient. The same concentration scale applies to the suspended condensates.
Native ``TPV->S`` supplies the entropy, and ``relative_humidity(..., ngas)``
checks the signed gas quotient including the five H2 product factors.

Altitude is a diagnostic obtained from hydrostatic balance with the specified
constant gravity:

.. math::

   \frac{dz}{d\ln P}=-\frac{P}{\rho g}.

The script integrates this equation by the trapezoidal rule on the pressure
grid. Temperature is found from entropy directly, rather than integrating an
assumed dry lapse rate. The tests check 31/61-level agreement at shared
pressure levels and recover the analytic constant-cp dry adiabat when all
condensable inventories are removed.

Data choices and scope
----------------------

The KCl curve is the registered ``kcl_lodders`` function: the bar-pressure fit
in `Morley et al. (2012), Eq. 18 <https://arxiv.org/abs/1206.4313>`_, converted
to natural logarithms and Pa. The script checks that a KCl solid cloud never
appears above 1044 K. Evaluating the curve at cloud-free hotter levels is a
mathematical continuation, not a model of liquid KCl.

For MgSiO3 the example deliberately uses the **conditional reconstruction**
documented in :doc:`reactions/mgsio3`, from
`Visscher et al. (2010), Tables 2 and 3 <https://arxiv.org/html/1001.3639>`_.
It is supplied inline in the YAML; no unverified named formula is added to
the production registry. The expression is

.. math::

   \log_{10}Q=6.238603569690592-49895/T,\qquad
   Q=\frac{p_{\rm Mg}p_{\rm SiH4}p_{\rm H2O}^{3}}{p_{\rm H2}^{5}}.

Its reconstruction assumes an H2-dominated mixture with X_H2 approximately
0.84. The explicit H2 inventory in this example is allowed to evolve. The
full reaction quotient has net pressure exponent zero. The generic Antoine
evaluator adds ln(100000), so the YAML subtracts 5 from its A coefficient
to represent this dimensionless quotient correctly.

Heat capacities are an internally consistent **idealized constant-cp closure**,
not measured species thermochemistry. For both curves L = ln(10)(A - B/T),
the card enforces

.. math::

   \Delta C_p=0,\qquad \Delta h/R=-B\ln 10.

The solid energy offsets and heat capacities are chosen accordingly, so the
latent heat used in the entropy calculation agrees with the temperature
derivative of the equilibrium curve. H2 rotational/vibrational variation,
nonideal EOS effects at thousands of bars, and competing silicates such as
forsterite are omitted. This example demonstrates the coupled numerical
method; quantitative Jupiter predictions need a broader reaction network,
verified caloric data, and an appropriate nonideal EOS.

YAML configuration
------------------

.. literalinclude:: ../../examples/jupiter_kcl_mgsio3.yaml
   :language: yaml
   :caption: examples/jupiter_kcl_mgsio3.yaml

Python implementation
---------------------

.. literalinclude:: ../../examples/example_jupiter_adiabat.py
   :language: python
   :caption: examples/example_jupiter_adiabat.py
