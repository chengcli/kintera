# Kinetics API Reference

The kinetics module lives under the `kintera` namespace (C++) / `kintera` package (Python) and provides differentiable chemical kinetics powered by libtorch. Configuration can be built programmatically or loaded from YAML files.

---

## Table of Contents

- [File Layout (C++)](#file-layout-c)
- [Core Types](#core-types)
- [Main Kinetics Module](#main-kinetics-module)
- [Rate Modules](#rate-modules)
  - [Arrhenius](#arrhenius)
  - [Three-Body](#three-body)
  - [Lindemann Falloff](#lindemann-falloff)
  - [Troe Falloff](#troe-falloff)
  - [SRI Falloff](#sri-falloff)
  - [Photolysis](#photolysis)
  - [Evaporation](#evaporation)
  - [Coagulation](#coagulation)
- [Actinic Flux](#actinic-flux)
- [Time Stepping](#time-stepping)
- [Global Species State](#global-species-state)
- [Utility Functions](#utility-functions)
- [Typical Usage (Python)](#typical-usage-python)

---

## File Layout (C++)

| File | Purpose |
|------|---------|
| `kinetics.hpp / .cpp` | Main `KineticsImpl` module |
| `rate_constant.hpp / .cpp` | `RateConstantImpl` evaluator |
| `arrhenius.hpp / .cpp` | Arrhenius rate constants |
| `three_body.hpp / .cpp` | Three-body reactions |
| `lindemann_falloff.hpp / .cpp` | Lindemann falloff |
| `troe_falloff.hpp / .cpp` | Troe falloff |
| `sri_falloff.hpp / .cpp` | SRI falloff |
| `photolysis.hpp / .cpp` | Photolysis rates |
| `actinic_flux.hpp` | Actinic flux data and factory functions |
| `evaporation.hpp / .cpp` | Evaporation rates |
| `coagulation.hpp / .cpp` | Coagulation (extends Arrhenius) |
| `jacobian.hpp / jacobian_*.cpp` | Jacobian computation |
| `species_rate.hpp` | Species production rate from reaction rates |
| `evolve_implicit.hpp` | Implicit and Rosenbrock-2 time stepping |
| `kinetics_formatter.hpp` | Formatting helpers |

Python bindings: `python/csrc/pykinetics.cpp`, `python/csrc/pyphotolysis.cpp`, `python/csrc/kintera.cpp`.

---

## Core Types

### `Composition`

```cpp
// C++
using Composition = std::map<std::string, double>;
```

```python
# Python — exposed as Dict[str, float]
{"O2": 1.0, "N2": 0.4}
```

### `Reaction`

Represents a single chemical reaction.

```cpp
// C++
struct Reaction {
  explicit Reaction(const std::string& equation);  // e.g. "O + O2 => O3"
  std::string equation() const;

  Composition reactants;
  Composition products;
  Composition orders;
  Composition efficiencies;
  std::string falloff_type;   // "none" | "Troe" | "SRI"
  bool reversible;            // default: false
};
```

```python
# Python
r = kt.Reaction("O + O2 => O3")
r.equation()            # -> "O + O2 => O3"
r.reactants             # -> {"O": 1.0, "O2": 1.0}   (get/set)
r.products              # -> {"O3": 1.0}               (get/set)
```

### `SpeciesThermo`

Base class for species thermodynamic data. `KineticsOptions` inherits from it.

```python
sp = opts.species()             # -> List[str]
opts.vapor_ids                  # List[int]   (get/set)
opts.cloud_ids                  # List[int]   (get/set)
opts.cref_R                     # List[float] (get/set)
opts.uref_R                     # List[float] (get/set)
opts.sref_R                     # List[float] (get/set)
opts.narrow_copy(indices)       # subset by species index
opts.accumulate(other)          # merge another SpeciesThermo
```

---

## Main Kinetics Module

### `KineticsOptions`

```cpp
// C++ — KineticsOptionsImpl, alias KineticsOptions = shared_ptr<KineticsOptionsImpl>
struct KineticsOptionsImpl : SpeciesThermoImpl {
  static KineticsOptions create();
  static KineticsOptions from_yaml(const std::string& filename, bool verbose = false);

  void report(std::ostream& os) const;
  std::vector<Reaction> reactions() const;

  double Tref = 298.15;             // Reference temperature [K]
  double Pref = 1.e5;               // Reference pressure [Pa]

  ArrheniusOptions        arrhenius;
  CoagulationOptions      coagulation;
  EvaporationOptions      evaporation;
  ThreeBodyOptions        three_body;
  LindemannFalloffOptions lindemann_falloff;
  TroeFalloffOptions      troe_falloff;
  SRIFalloffOptions       sri_falloff;
  PhotolysisOptions       photolysis;

  bool evolve_temperature = false;
  bool verbose = false;
  bool offset_zero = false;
};
```

```python
# Python
opts = kt.KineticsOptions()
opts = kt.KineticsOptions.from_yaml("mechanism.yaml", verbose=False)

opts.Tref                       # float (get/set)
opts.Pref                       # float (get/set)
opts.arrhenius                  # ArrheniusOptions (get/set)
opts.three_body                 # ThreeBodyOptions (get/set)
opts.lindemann_falloff          # LindemannFalloffOptions (get/set)
opts.troe_falloff               # TroeFalloffOptions (get/set)
opts.sri_falloff                # SRIFalloffOptions (get/set)
opts.evaporation                # EvaporationOptions (get/set)
opts.coagulation                # CoagulationOptions (get/set)
opts.evolve_temperature         # bool (get/set)
opts.reactions()                # -> List[Reaction]
opts.species()                  # -> List[str]
```

### `Kinetics`

The main module. Inherits from `torch::nn::Cloneable` (C++) / `torch.nn.Module` (Python).

```cpp
// C++
class KineticsImpl : torch::nn::Cloneable<KineticsImpl> {
public:
  explicit KineticsImpl(KineticsOptions const& options);

  // Forward — compute reaction rates
  std::tuple<Tensor, Tensor, optional<Tensor>>
  forward(Tensor temp, Tensor pres, Tensor conc);

  std::tuple<Tensor, Tensor, optional<Tensor>>
  forward(Tensor temp, Tensor pres, Tensor conc,
          std::map<std::string, Tensor> const& extra);

  // Jacobian of reaction rates w.r.t. species concentrations
  Tensor jacobian(Tensor temp, Tensor conc, Tensor cvol,
                  Tensor rate, Tensor rc_ddC,
                  optional<Tensor> rc_ddT = nullopt,
                  optional<Tensor> rate_constants = nullopt) const;

  // Buffers / data
  Tensor stoich;           // (nspecies, nreaction) stoichiometric matrix
  KineticsOptions options;
};
```

```python
# Python
kinet = kt.Kinetics(opts)

# Forward pass
rate, rc_ddC, rc_ddT = kinet.forward(temp, pres, conc)
rate, rc_ddC, rc_ddT = kinet.forward(temp, pres, conc, extra)

# Jacobian
jac = kinet.jacobian(temp, conc, cvol, rate, rc_ddC, rc_ddT=None)

# Explicit forward rate constants (required for zero-abundance condensates)
rate, rc_ddC, rc_ddT, kf = kinet.forward_with_rate_constants(temp, pres, conc)
jac = kinet.jacobian(temp, conc, cvol, rate, rc_ddC, rc_ddT,
                     rate_constants=kf)

# Introspection
kinet.options                   # KineticsOptions (read-only)
kinet.stoich                    # Tensor (nspecies, nreaction)
kinet.module("arrhenius")       # named sub-module
kinet.buffer("photolysis.wavelength")  # named buffer
```

**Parameters:**

| Argument | Shape | Units | Description |
|----------|-------|-------|-------------|
| `temp` | `(...)` | K | Temperature |
| `pres` | `(...)` | Pa | Pressure |
| `conc` | `(..., nspecies)` | mol/m³ | Species concentrations |
| `extra` | dict | — | Optional; use `{"actinic_flux": flux}` for photolysis, where `flux` is sampled on `photolysis.wavelength` |

**Returns from `forward`:**

| Output | Shape | Description |
|--------|-------|-------------|
| `rate` | `(..., nreaction)` | Reaction rates |
| `rc_ddC` | `(..., nreaction, nspecies)` | d(rate)/d(conc) |
| `rc_ddT` | `(..., nreaction)` or `None` | d(rate)/d(T), present when temperature evolution is enabled |

---

## Rate Modules

Thermal rate modules share a common `forward` signature:

```cpp
// C++
Tensor forward(Tensor T, Tensor P, Tensor C,
               std::map<std::string, Tensor> const& other);
```

```python
# Python
result = module.forward(temp, pres, conc, other)   # other: Dict[str, Tensor]
```

`Photolysis` is separate and uses:

```python
photolysis.update_xs_diss_stacked(temp)
rate = photolysis.forward(temp, actinic_flux)
```

### Arrhenius

Rate law: `k = A · T^b · exp(-Ea_R / T)`

| Option | Type | Description |
|--------|------|-------------|
| `Tref` | `float` | Reference temperature [K] (default 300) |
| `reactions` | `List[Reaction]` | Reaction definitions |
| `A` | `List[float]` | Pre-exponential factors |
| `b` | `List[float]` | Temperature exponents |
| `Ea_R` | `List[float]` | Activation energy / R [K] |
| `E4_R` | `List[float]` | Optional 4th parameter |

### Three-Body

Rate law: `k = k0 · [M]_eff`

| Option | Type | Description |
|--------|------|-------------|
| `Tref` | `float` | Reference temperature [K] |
| `units` | `str` | Unit system (default `"molecule,cm,s"`) |
| `reactions` | `List[Reaction]` | Reaction definitions |
| `k0_A`, `k0_b`, `k0_Ea_R` | `List[float]` | Low-pressure Arrhenius parameters |
| `efficiencies` | `List[Composition]` | Third-body efficiencies per reaction |

Extra method: `pretty_print(ostream&)` / `.pretty_print()` in Python.

### Lindemann Falloff

Rate law: `k = k0·[M]_eff / (1 + Pr)` where `Pr = k0·[M] / k∞`

| Option | Type | Description |
|--------|------|-------------|
| `Tref`, `units` | — | Same as Three-Body |
| `k0_A`, `k0_b`, `k0_Ea_R` | `List[float]` | Low-pressure parameters |
| `kinf_A`, `kinf_b`, `kinf_Ea_R` | `List[float]` | High-pressure parameters |
| `efficiencies` | `List[Composition]` | Third-body efficiencies |

### Troe Falloff

Lindemann × Troe broadening factor `F_cent`.

Additional options beyond Lindemann:

| Option | Type | Description |
|--------|------|-------------|
| `troe_A` | `List[float]` | Troe A parameter |
| `troe_T3` | `List[float]` | Troe T*** parameter |
| `troe_T1` | `List[float]` | Troe T* parameter |
| `troe_T2` | `List[float]` | Troe T** parameter |

### SRI Falloff

Lindemann × SRI broadening factor.

Additional options beyond Lindemann:

| Option | Type | Description |
|--------|------|-------------|
| `sri_A`, `sri_B`, `sri_C` | `List[float]` | SRI primary parameters |
| `sri_D`, `sri_E` | `List[float]` | SRI secondary parameters |

### Photolysis

Photolysis rate constants from cross-section data and actinic flux.

| Option | Type | Description |
|--------|------|-------------|
| `reactions` | `List[Reaction]` | Photolysis reactions |
| `wavelength` | `List[float]` | Wavelength grid [nm] |
| `temperature` | `List[float]` | Temperature grid [K] |
| `cross_section` | `List[float]` | Absorption cross-sections [cm² molecule⁻¹] |
| `branches` | `List[List[Composition]]` | Product branching ratios per reaction per wavelength |
| `branch_names` | `List[List[str]]` | Branch labels |

Extra methods:

```python
# Refresh the cached dissociative cross-sections for the current temperature
photolysis.update_xs_diss_stacked(temp)

# Evaluate rates on the module wavelength grid
rate = photolysis.forward(temp, actinic_flux)

# Interpolate cross-section onto a wavelength/temperature grid
sigma = photolysis.interp_cross_section(rxn_idx, wave, temp)

# Effective stoichiometry weighted by actinic flux and cross-section
stoich = photolysis.get_effective_stoich(rxn_idx, wave, aflux, temp)
```

### Evaporation

Diffusion-limited evaporation. Inherits from `NucleationOptions`.

| Option | Type | Description |
|--------|------|-------------|
| `Tref` | `float` | Reference temperature [K] |
| `Pref` | `float` | Reference pressure [Pa] |
| `diff_c`, `diff_T`, `diff_P` | `List[float]` | Diffusivity parameters |
| `vm` | `List[float]` | Molar volume [m³/mol] |
| `diameter` | `List[float]` | Particle diameter [m] |

Evaporation supports an irreversible reaction with one condensed reactant
(coefficient 1), optional gas reactants, and any number of gas products with
positive coefficients. For example:

```yaml
species:
  - {name: H2, phase: gas, composition: {H: 2}, cv_R: 2.5}
  - {name: Na, phase: gas, composition: {Na: 1}, cv_R: 1.5}
  - {name: H2S, phase: gas, composition: {H: 2, S: 1}, cv_R: 2.5}
  - {name: Na2S(s), phase: solid, composition: {Na: 2, S: 1}, cv_R: 18, u0_R: -60000}
reactions:
  - equation: Na2S(s) + H2 => 2 Na + H2S
    type: evaporation
    rate-constant: {formula: na_h2s_visscher, diff_c: 2.e-5,
                    vm: 4.22e-5, diameter: 1.e-4}
```

These material parameters are illustrative. Mixed-reactant YAML requires explicit
phases on every participating species. Legacy single-reactant cards retain their
phase inference. Condensed products, multiple condensed reactants, reversible
arrows, and custom kinetic orders are not supported by this rate law.

For gas coefficients `nu` positive on the evaporation product side, the surface
extent `x` satisfies
`sum(nu * log(C + nu*x)) = log(Qsat) - sum(nu)*log(R*T)`.
`Qsat` is the existing saturation quotient using numerical partial pressures in
Pa, including consumed gases in its denominator. The rate in mol/(m³ s) is
`12*D*vm/diameter**2 * Csolid * x`, where
`D = diff_c*(T/Tref)**diff_T*(P/Pref)**diff_P`.
Gas consumption enters the surface solve, not an extra mass-action multiplier.
The model assumes one effective diffusivity for all gases, pure condensed-phase
activity, and a fixed particle diameter; it is not a multicomponent transport
model. Saturated/supersaturated or gas-reactant-starved states do not evaporate.

General surface extents use a bracketed solve and implicit first derivatives.
Inputs to this path must have finite concentrations and a finite
logarithmic saturation quotient. Legacy one-product and unit-coefficient
two-product reactions retain their analytic rate formulas.

`Kinetics.forward` supplies gas classification automatically. Direct calls to
`Evaporation.forward(T, P, C, other)` require `other["stoich"]` and, for mixed
reactants, `other["gas_mask"]`: a boolean `(nspecies,)` tensor in the same row
order as `C` and `stoich`. Unexpanded `C` has shape `(..., nspecies)`; expanded
`C` has shape `(..., nspecies, nreaction)`. Classification is not taken from a
mutable global registry during rate evaluation.

Negative concentrations are treated as zero during the general surface solve;
its derivative with respect to a negative concentration is zero. Condensed
abundance is also clamped at zero, so negative solver iterates never generate
negative evaporation rates. Non-finite inputs still raise an error. The surface
solve retains its 128-iteration convergence guard; on CUDA it checks convergence
every eight iterations to reduce host synchronization.

Chemical stoichiometry is unchanged; only evaporation's kinetic prefactor is
first order in the solid. `jacobian` is stateless: it uses only its arguments.
Use `forward_with_rate_constants` (the same arguments as `forward`) to obtain
`(rate, rc_ddC, rc_ddT, kf)`, and pass `rate_constants=kf` to `jacobian`. `kf` has
shape `(..., nreaction_orig)` and includes any `kf_override` values, before
mass-action multiplication or reversible augmentation. The returned coefficient
is detached, just like `rate`; differentiating the assembled Jacobian treats
these supplied tensors and derivatives as independent inputs, not a full
second derivative through the surface solver.

The existing three-result `forward` remains available. Without `rate_constants`,
the Jacobian recovers evaporation coefficients from rate/solid abundance where
the solid is positive. At exactly zero solid abundance, rate alone cannot recover
the coefficient: the call raises an explicit error instead of consulting hidden
state. Evaporation callers in snapy/paddle that can encounter empty condensates
should migrate to the explicit form above. Results remain valid across
interleaved forwards with different states and batch shapes. `evolve_implicit` remains a linearized step
without a positivity guarantee. The standalone
`examples/example_na2s_evaporation.py` demonstrates timestep rejection and
substepping for a conserved, isothermal box without clipping species inventories.

### Coagulation

Inherits from `ArrheniusOptions` with no additional parameters.

---

## Actinic Flux

### `ActinicFluxOptions`

| Option | Type | Description |
|--------|------|-------------|
| `wavelength` | `List[float]` | Wavelength grid [nm] |
| `default_flux` | `List[float]` | Default flux values |
| `wave_min` | `float` | Minimum wavelength [nm] |
| `wave_max` | `float` | Maximum wavelength [nm] |

### Factory Functions

```python
# From options onto a target wavelength grid
aflux = kt.create_actinic_flux(options, wavelength)

# Uniform flux on an existing wavelength grid
aflux = kt.create_uniform_flux(wavelength, flux_value)

# Simplified solar spectrum on an existing wavelength grid
aflux = kt.create_solar_flux(wavelength, peak_flux=1e14)

# Interpolate a flux field to a new wavelength grid
aflux_new = kt.interpolate_actinic_flux(src_wavelength, src_flux,
                                        new_wavelength)
```

---

## Time Stepping

### `evolve_implicit`

Backward Euler: solves `(I/dt - S·J) · δ = S · rate`.

```python
delta = kt.evolve_implicit(rate, stoich, jacobian, dt)
```

| Argument | Shape | Description |
|----------|-------|-------------|
| `rate` | `(..., nreaction)` | Reaction rates |
| `stoich` | `(nspecies, nreaction)` | Stoichiometric matrix |
| `jacobian` | `(..., nreaction, nspecies)` | Reaction-space Jacobian |
| `dt` | `float` | Time step [s] |
| **Returns** | `(..., nspecies)` | Concentration increment δ |

### `evolve_ros2`

Second-order Rosenbrock (Ros2) scheme.

```python
delta, error = kt.evolve_ros2(rate1, rate2, stoich, jacobian, dt)
```

| Argument | Description |
|----------|-------------|
| `rate1` | Rates evaluated at current state C |
| `rate2` | Rates evaluated at C + (1/γ)·k1 |
| **Returns** | `(delta, error)` — increment and embedded error estimate |

### `ros2_k1`

First-stage intermediate for Ros2 (use to compute the state for `rate2`).

```python
k1 = kt.ros2_k1(rate1, stoich, jacobian, dt)
```

---

## Global Species State

Module-level getters and setters for the global species registry:

```python
kt.species_names()                    # -> List[str]
kt.set_species_names(["O2", "O3"])    # -> List[str]

kt.species_weights()                  # -> List[float]
kt.set_species_weights([32.0, 48.0])

kt.species_cref_R()                   # -> List[float]   (Cp_ref / R)
kt.set_species_cref_R([...])

kt.species_uref_R()                   # -> List[float]   (U_ref / R)
kt.set_species_uref_R([...])

kt.species_sref_R()                   # -> List[float]   (S_ref / R)
kt.set_species_sref_R([...])
```

---

## Utility Functions

```python
kt.set_search_paths("/path/to/data")              # set YAML/data search paths
kt.get_search_paths()                              # -> List[str]
kt.add_resource_directory("/another/path", prepend=True)
kt.find_resource("mechanism.yaml")                 # -> resolved absolute path
```

---

## Typical Usage (Python)

```python
import torch
import kintera as kt

# 1. Load mechanism from YAML
opts = kt.KineticsOptions.from_yaml("mechanism.yaml")
kinet = kt.Kinetics(opts)

# 2. Set up state
T    = torch.tensor([1500.0])        # [K]
P    = torch.tensor([101325.0])      # [Pa]
conc = torch.tensor([[1e18, 1e16]])  # [mol/m³], shape (1, nspecies)
cvol = torch.tensor([1.0])           # volume per layer

# 3. (Optional) Actinic flux for photolysis
phot = kinet.module("photolysis")
aflux = kt.create_solar_flux(phot.wavelength)
phot.update_xs_diss_stacked(T)
extra = {"actinic_flux": aflux}

# 4. Forward pass
rate, rc_ddC, rc_ddT = kinet.forward(T, P, conc, extra)

# 5. Jacobian
jac = kinet.jacobian(T, conc, cvol, rate, rc_ddC, rc_ddT)

# 6. Time step (implicit Euler)
dt = 1.0  # seconds
delta = kt.evolve_implicit(rate, kinet.stoich, jac, dt)
conc = torch.clamp(conc + delta, min=0.0)

# Or Ros2 for second-order accuracy:
k1 = kt.ros2_k1(rate, kinet.stoich, jac, dt)
# ... evaluate rate2 at updated state ...
# delta, err = kt.evolve_ros2(rate, rate2, kinet.stoich, jac, dt)
```
