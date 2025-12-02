#pragma once

// C/C++
#include <initializer_list>
#include <iomanip>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include <kintera/constants.h>

#include <kintera/species.hpp>

#include "eval_uhs.hpp"
#include "nucleation.hpp"

// arg
#include <kintera/add_arg.h>

namespace YAML {
class Node;
}  // namespace YAML

namespace kintera {

template <typename T>
inline std::vector<T> insert_first(T value, std::vector<T> const& input) {
  std::vector<T> result;
  result.reserve(input.size() + 1);
  result.push_back(value);
  result.insert(result.end(), input.begin(), input.end());
  return result;
}

struct ThermoOptionsImpl final : public SpeciesThermoImpl {
  static std::shared_ptr<ThermoOptionsImpl> create() {
    auto op = std::make_shared<ThermoOptionsImpl>();
    op->nucleation() = NucleationOptionsImpl::create();
    return op;
  }

  //! \brief Create a `ThermoOptions` object from a YAML file
  /*!
   * This function reads a YAML file and attempts to create a `ThermoOptions`
   * object from it. If the YAML file contains "reference-state", a valid
   * `ThermoOptions` object will be created. Otherwise, a nullptr
   * is returned.
   */
  static std::shared_ptr<ThermoOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);
  static std::shared_ptr<ThermoOptionsImpl> from_yaml(YAML::Node const& config,
                                                      bool verbose = false);

  void report(std::ostream& os) const {
    os << "* Tref = " << Tref() << " K\n"
       << "* Pref = " << Pref() << " Pa\n"
       << "* max_iter = " << max_iter() << "\n"
       << "* ftol = " << ftol() << "\n"
       << "* gas_floor = " << gas_floor() << "\n"
       << "* offset_zero = " << (offset_zero() ? "true" : "false") << "\n"
       << "* verbose = " << (verbose() ? "true" : "false") << "\n";
  }

  std::vector<Reaction> reactions() const;

  ADD_ARG(double, Tref) = 300.0;
  ADD_ARG(double, Pref) = 1.e5;

  ADD_ARG(int, max_iter) = 10;
  ADD_ARG(double, ftol) = 1e-6;
  ADD_ARG(double, gas_floor) = 1.e-20;
  ADD_ARG(bool, verbose) = false;
  ADD_ARG(bool, offset_zero) = false;

  ADD_ARG(NucleationOptions, nucleation) = nullptr;
};
using ThermoOptions = std::shared_ptr<ThermoOptionsImpl>;

//! Mass Thermodynamics
class ThermoYImpl : public torch::nn::Cloneable<ThermoYImpl> {
 public:
  //! Create and register an ThermoY module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts  options for creating the ThermoY module
   * \param[in] p     parent module for registering the created module
   * \return          created ThermoY module
   */
  static std::shared_ptr<ThermoYImpl> create(
      ThermoOptions const& opts, torch::nn::Module* p,
      std::string const& name = "thermo");

  //! 1. / mu. [mol/kg]
  torch::Tensor inv_mu;

  //! constant part of heat capacity at constant volume [J/(kg K)]
  torch::Tensor cv0;

  //! internal energy offset at T = 0 [J/kg]
  torch::Tensor u0;

  //! stoichiometry matrix (nspecies, nreaction)
  torch::Tensor stoich;

  //! kkt warm start active set
  torch::Tensor reaction_set, nactive;

  //! options with which this `ThermoY` was constructed
  ThermoOptions options;

  ThermoYImpl() : options(ThermoOptionsImpl::create()) {}
  explicit ThermoYImpl(const ThermoOptions& options_);
  ThermoYImpl(const ThermoOptions& options1, const SpeciesThermo& options2);
  void reset() override;
  void pretty_print(std::ostream& os) const override;

  //! \brief perform conversions
  torch::Tensor compute(std::string ab, std::vector<torch::Tensor> const& args);

  //! \brief Perform saturation adjustment
  /*!
   * This function adjusts the mass fraction to account for saturation
   * conditions. It modifies the input mass fraction tensor in place.
   * After calling this function, equilibrium temperature (T) and pressure (P)
   * will be cached and can be accessed later using `buffer("T")` and
   * `buffer("P")`.
   *
   * \param[in] rho density
   * \param[in] intEng total internal energy [J/m^3]
   * \param[in,out] yfrac mass fraction, (ny, ...)
   * \param[in] warm_start if true, use previous active set as warm start
   * \param[out] diag optional diagnostic output, (..., ndiag)
   * \return gain matrix, (..., nreaction, nreaction)
   */
  torch::Tensor forward(torch::Tensor rho, torch::Tensor intEng,
                        torch::Tensor& yfrac, bool warm_start = false,
                        torch::optional<torch::Tensor> diag = torch::nullopt);

 private:
  //! \brief specific volume (m^3/kg) to mass fraction
  /*
   * \param[in] ivol inverse specific volume, kg/m^3, (..., 1 + ny)
   * \param[out] out mass fraction, (ny, ...)
   */
  void _ivol_to_yfrac(torch::Tensor ivol, torch::Tensor& out) const;

  //! \brief Calculate mole fraction from mass fraction
  /*!
   * \param[in] yfrac mass fraction, (ny, ...)
   * \param[out] out mole fraction, (..., 1 + ny)
   */
  void _yfrac_to_xfrac(torch::Tensor yfrac, torch::Tensor& out) const;

  //! \brief mass fraction to specific volume (m^3/kg)
  /*!
   * \param[in] rho total density, kg/m^3
   * \param[in] yfrac mass fraction, (ny, ...)
   * \param[out] out inverse specific volume, kg/m^3/, (..., 1 + ny)
   */
  void _yfrac_to_ivol(torch::Tensor rho, torch::Tensor yfrac,
                      torch::Tensor& out) const;

  //! \brief Calculate temperature (K)
  /*`
   * \param[in] pres pressure, pa
   * \param[in] ivol inverse of specific volume, kg/m^3, (..., 1 + ny)
   * \param[out] temperature, K, (...)
   */
  void _pres_to_temp(torch::Tensor pres, torch::Tensor ivol,
                     torch::Tensor& out) const;

  //! \brief calculate volumetric heat capacity (J/(m^3 K))
  /*!
   * \param[in] ivol inverse of specific volume, kg/m^3, (..., 1 + ny)
   * \param[in] temp temperature, K
   * \param[out] out volumetric heat capacity, J/(m^3 K), (...)
   */
  void _cv_vol(torch::Tensor ivol, torch::Tensor temp,
               torch::Tensor& out) const;

  //! \brief calculate volumetric internal energy (J/m^3)
  /*!
   * \param[in] ivol inverse of specific volume, kg/m^3, (..., 1 + ny)
   * \param[in] temp temperature, K
   * \param[out] out volumetric internal energy, J/m^3, (...)
   */
  void _intEng_vol(torch::Tensor ivol, torch::Tensor temp,
                   torch::Tensor& out) const {
    // kg/m^3 -> mol/m^3
    auto conc = ivol * inv_mu;
    auto ui = eval_intEng_R(temp, conc, options) * constants::Rgas;
    out.set_((ui * conc).sum(-1));
  }

  //! \brief calculate temperature (K) from internal energy
  /*!
   * \param[in] ivol inverse of specific volume, kg/m^3, (..., 1 + ny)
   * \param[in] intEng volumetric internal energy, J/m^3, (...)
   * \param[out] out temperature, K, (...)
   */
  void _intEng_to_temp(torch::Tensor ivol, torch::Tensor intEng,
                       torch::Tensor& out) const;

  //! \brief calculate pressure (Pa)
  /*!
   * \param[in] ivol inverse of specific volume, kg/m^3, (..., 1 + ny)
   * \param[in] temp temperature, K
   * \param[out] out pressure, Pa, (...)
   */
  void _temp_to_pres(torch::Tensor ivol, torch::Tensor temp,
                     torch::Tensor& out) const;

  //! \brief calculate volumetric entropy (J/(m^3 K))
  /*
   * \param[in] pres pressure, pa
   * \param[in] ivol inverse of specific volume, kg/m^3, (..., 1 + ny)
   * \param[in] temp temperature, K
   * \param[out] out volumetric entropy, J/(m^3 K), (...)
   */
  void _entropy_vol(torch::Tensor pres, torch::Tensor ivol, torch::Tensor temp,
                    torch::Tensor& out) const {
    // kg/m^3 -> mol/m^3
    auto conc = ivol * inv_mu;
    auto si =
        eval_entropy_R(temp, pres, conc, stoich, options) * constants::Rgas;
    out.set_((si * conc).sum(-1));
  }
};
TORCH_MODULE(ThermoY);

//! Molar thermodynamics
class ThermoXImpl : public torch::nn::Cloneable<ThermoXImpl> {
 public:
  //! Create and register an ThermoX module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts  options for creating the ThermoX module
   * \param[in] p     parent module for registering the created module
   * \return          created ThermoX module
   */
  static std::shared_ptr<ThermoXImpl> create(
      ThermoOptions const& opts, torch::nn::Module* p,
      std::string const& name = "thermo");

  //! mu.
  torch::Tensor mu;

  //! stoichiometry matrix
  torch::Tensor stoich;

  //! kkt warm start active set
  torch::Tensor reaction_set, nactive;

  //! options with which this `ThermoX` was constructed
  ThermoOptions options;

  ThermoXImpl() : options(ThermoOptionsImpl::create()) {}
  explicit ThermoXImpl(const ThermoOptions& options_);
  ThermoXImpl(const ThermoOptions& options1, const SpeciesThermo& options2);
  void reset() override;
  void pretty_print(std::ostream& os) const override;

  //! \brief perform conversions
  /*!
   * \param ab name of the conversion to perform
   * \param args arguments to the conversion function
   * \return result of the conversion
   */
  torch::Tensor compute(std::string ab, std::vector<torch::Tensor> const& args);

  //! \brief Calculate effective heat capacity at constant pressure
  /*!
   *
   * \param temp Temperature tensor (K)
   * \param pres Pressure tensor (Pa)
   * \param xfrac Mole fraction tensor
   * \param gain Gain tensor
   * \param conc Optional concentration tensor, if not provided it will be
   * computed
   * \return Equivalent heat capacity at constant pressure (Cp) tensor [J/(mol
   * K)]
   */
  torch::Tensor effective_cp(
      torch::Tensor temp, torch::Tensor pres, torch::Tensor xfrac,
      torch::Tensor gain, torch::optional<torch::Tensor> conc = torch::nullopt);

  //! \brief Extrapolate state TPX to a new pressure along an adiabat
  /*!
   * Extrapolates the state variables (temperature, pressure, and mole
   * fractions)
   *
   * \param[in,out] temp Temperature tensor (K)
   * \param[in,out] pres Pressure tensor (Pa)
   * \param[in,out] xfrac Mole fraction tensor
   * \param[in] dlnp Logarithmic change in pressure (dlnp = ln(p_new / p_old))
   * \param[in] verbose If true, print debug information
   */
  void extrapolate_ad(torch::Tensor temp, torch::Tensor pres,
                      torch::Tensor xfrac, double dlnp, bool verbose = false);

  /*! \brief Extrapolate state TPX to a new height along an adiabat
   * Extrapolates the state variables (temperature, pressure, and mole
   * fractions)
   *
   * \param[in,out] temp Temperature tensor (K)
   * \param[in,out] pres Pressure tensor (Pa)
   * \param[in,out] xfrac Mole fraction tensor
   * \param[in] grav Gravitational acceleration (m/s^2)
   * \param[in] dz Height change (m)
   * \param[in] verbose If true, print debug information
   */
  void extrapolate_ad(torch::Tensor temp, torch::Tensor pres,
                      torch::Tensor xfrac, double grav, double dz,
                      bool verbose = false);

  //! \brief Calculate the equilibrium state given temperature and pressure
  /*!
   * \param[in] temp temperature, K
   * \param[in] pres pressure, Pa
   * \param[in,out] xfrac mole fraction, (..., 1 + ny)
   * \param[in] warm_start if true, use previous active set as warm start
   * \param[out] diag optional diagnostic output, (..., ndiag)
   * \return gain matrix, (..., nreaction, nreaction)
   */
  torch::Tensor forward(torch::Tensor temp, torch::Tensor pres,
                        torch::Tensor& xfrac, bool warm_start = false,
                        torch::optional<torch::Tensor> diag = torch::nullopt);

 private:
  //! \brief Calculate mass fraction from mole fraction
  /*!
   * \param[in] xfrac mole fraction, (..., 1 + ny)
   * \param[out] out mass fraction, (ny, ...)
   */
  void _xfrac_to_yfrac(torch::Tensor xfrac, torch::Tensor& out) const;

  //! \brief Calculate total density
  /*!
   * \param[in] conc mole concentration, (..., 1 + ny)
   * \param[out] out total density, kg/m^3, (...)
   */
  void _conc_to_dens(torch::Tensor conc, torch::Tensor& out) const {
    out.set_((conc * mu).sum(-1));
  }

  //! \brief calculatec volumetric heat capacity
  /*
   * \param[in] temp temperature, K
   * \param[in] conc mole concentration, (..., 1 + ny)
   * \param[out] out volumetric heat capacity, J/(m^3 K), (...)
   */
  void _cp_vol(torch::Tensor temp, torch::Tensor conc,
               torch::Tensor& out) const {
    auto cp = eval_cp_R(temp, conc, options) * constants::Rgas;
    out.set_((conc * cp).sum(-1));
  }

  //! \brief calculate volumetric enthalpy
  /*!
   * \param[in] temp temperature, K
   * \param[in] conc mole concentration, (..., 1 + ny)
   * \param[out] out volumetric enthalpy, J/m^3, (...)
   */
  void _enthalpy_vol(torch::Tensor temp, torch::Tensor conc,
                     torch::Tensor& out) const {
    auto hi = eval_enthalpy_R(temp, conc, options) * constants::Rgas;
    out.set_((conc * hi).sum(-1));
  }

  //! \brief calculate volumetric entropy
  /*!
   * \param[in] temp temperature, K, (...)
   * \param[in] pres pressure, Pa, (...)
   * \param[in] conc mole concentration, (..., 1 + ny)
   * \param[out] out volumetric entropy, J/(m^3 K), (...)
   */
  void _entropy_vol(torch::Tensor temp, torch::Tensor pres, torch::Tensor conc,
                    torch::Tensor& out) const {
    auto si =
        eval_entropy_R(temp, pres, conc, stoich, options) * constants::Rgas;
    out.set_((conc * si).sum(-1));
  }

  //! \brief Calculate temperature from pressure and entropy
  /*!
   * \param[in] pres pressure, Pa, (...)
   * \param[in] xfrac mole fractions, (..., 1 + ny)
   * \param[in] entropy volumetric entropy, J/(m^3 K), (...)
   * \param[out] out temperature, K, (...)
   */
  void _entropy_to_temp(torch::Tensor pres, torch::Tensor xfrac,
                        torch::Tensor entropy, torch::Tensor& out);

  //! \brief Calculate concentration from mole fraction
  /*
   * \param[in] temp temperature, K
   * \param[in] pres pressure, Pa
   * \param[in] xfrac mole fractions, (..., 1 + ny)
   * \param[out] out mole concentration, mol/m^3, (..., 1 + ny)
   */
  void _xfrac_to_conc(torch::Tensor temp, torch::Tensor pres,
                      torch::Tensor xfrac, torch::Tensor& out) const;
};
TORCH_MODULE(ThermoX);

}  // namespace kintera

#undef ADD_ARG
