#pragma once

// C/C++
#include <initializer_list>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include <kintera/utils/func1.hpp>

#include "thermo_reactions.hpp"

// arg
#include <kintera/add_arg.h>

namespace kintera {

//! names of all species
extern std::vector<std::string> species_names;

//! molecular weights of all species [kg/mol]
extern std::vector<double> species_weights;

template <typename T>
inline std::vector<T> insert_first(T value, std::vector<T> const& input) {
  std::vector<T> result;
  result.reserve(input.size() + 1);
  result.push_back(value);
  result.insert(result.end(), input.begin(), input.end());
  return result;
}

struct ThermoOptions {
  //! \brief Create a `ThermoOptions` object from a YAML file
  /*!
   * This function reads a YAML file and creates a `ThermoOptions`
   * object from it. The YAML file must contain the following fields:
   *  - "species", list of species names and their composition
   *  - "vapor": list of vapor species
   *  - "cloud": list of cloud species
   */
  static ThermoOptions from_yaml(std::string const& filename);

  ThermoOptions() = default;

  ADD_ARG(double, gammad) = 1.4;
  ADD_ARG(double, Rd) = 287.0;
  ADD_ARG(double, Tref) = 300.0;
  ADD_ARG(double, Pref) = 1.e5;

  ADD_ARG(std::vector<int>, vapor_ids);
  ADD_ARG(std::vector<int>, cloud_ids);

  ADD_ARG(std::vector<double>, mu_ratio);
  ADD_ARG(std::vector<double>, cref_R);
  ADD_ARG(std::vector<double>, uref_R);

  ADD_ARG(std::vector<user_func1>, intEng_extra);
  ADD_ARG(std::vector<user_func1>, cv_extra);

  ADD_ARG(std::vector<Nucleation>, react);
  ADD_ARG(std::vector<std::string>, species);

  ADD_ARG(int, max_iter) = 5;
  ADD_ARG(double, ftol) = 1e-6;
};

//! Mass Thermodynamics
class ThermoYImpl : public torch::nn::Cloneable<ThermoYImpl> {
 public:
  //! mud / mu - 1.
  torch::Tensor mu_ratio_m1;

  //! cv/cvd - 1.
  torch::Tensor cv_ratio_m1;

  //! dimensionless internal energy offset at T = 0
  torch::Tensor u0_R;

  //! stoichiometry matrix
  torch::Tensor stoich;

  //! options with which this `ThermoY` was constructed
  ThermoOptions options;

  ThermoYImpl() = default;
  explicit ThermoYImpl(const ThermoOptions& options_);
  void reset() override;

  //! \brief multi-component density correction
  /*!
   * Eq.16 in Li2019
   * $ f_{\epsilon} = 1 + \sum_{i \in V \cup C} y_i(1./\epsilon_{i} - 1) $
   *
   * \param yfrac mass fraction
   * \param n starting index
   */
  torch::Tensor f_eps(torch::Tensor yfrac, int n = 0) const;

  //! \brief multi-component cv correction
  /*!
   * Eq.62 in Li2019
   * $ f_{\sigma} = 1 + \sum_{i \in V \cup C} y_i(\sigma_{v,i} - 1) $
   *
   * \param yfrac mass fraction
   * \param n starting index
   */
  torch::Tensor f_sig(torch::Tensor yfrac, int n = 0) const;

  //! \brief perform conversions
  torch::Tensor compute(
      std::string ab, std::initializer_list<torch::Tensor> args,
      torch::optional<torch::Tensor> out = torch::nullopt) const;

  //! \brief Perform saturation adjustment
  /*!
   * \param rho density
   *
   * \param yfrac mass fraction
   *
   * \param intEng zero-offset total internal energy [J/m^3]
   * $ U = \rho U_0 f_{sigma} $
   *
   * \return adjusted density
   */
  torch::Tensor forward(torch::Tensor rho, torch::Tensor intEng,
                        torch::Tensor yfrac);

 private:
  //! \brief pressure (pa) -> temperature (K)
  torch::Tensor _pres_to_temp(torch::Tensor rho, torch::Tensor pres,
                              torch::Tensor yfrac) const {
    return pres / (rho * options.Rd() * f_eps(yfrac));
  }

  //! \brief temperature (K) -> pressure (pa)
  torch::Tensor _temp_to_pres(torch::Tensor rho, torch::Tensor temp,
                              torch::Tensor yfrac) const {
    return rho * temp * options.Rd() * f_eps(yfrac);
  }

  //! \brief Calculate mole fraction from mass fraction
  /*!
   * Eq.77 in Li2019
   * $ x_i = \frac{y_i/\epsilon_i}{1. + \sum_{i \in V \cup C} y_i(1./\epsilon_i
   * - 1.)} $
   *
   * \param yfrac mass fraction, (ny, ...)
   * \return mole fraction, (..., 1 + ny)
   */
  torch::Tensor _yfrac_to_xfrac(torch::Tensor yfrac) const;

  //! \brief pressure (pa) -> internal energy (J/m^3)
  torch::Tensor _pres_to_intEng(torch::Tensor rho, torch::Tensor pres,
                                torch::Tensor yfrac) const;

  //! \brief internal energy (J/m^3) -> pressure (pa)
  torch::Tensor _intEng_to_pres(torch::Tensor rho, torch::Tensor intEng,
                                torch::Tensor yfrac) const;

  //! \brief mass fraction to mole concentration (mol/m^3)
  /*!
   * \param rho total density, kg/m^3
   * \param yfrac mass fraction, (ny, ...)
   * \return mole concentration, mol/m^3, (..., 1 + ny)
   */
  torch::Tensor _yfrac_to_conc(torch::Tensor rho, torch::Tensor yfrac) const;

  //! \brief mole concentration (mol/m^3) to mass fraction
  torch::Tensor _conc_to_yfrac(
      torch::Tensor conc,
      torch::optional<torch::Tensor> out = torch::nullopt) const;
};
TORCH_MODULE(ThermoY);

//! Molar thermodynamics
class ThermoXImpl : public torch::nn::Cloneable<ThermoXImpl> {
 public:
  //! mu / mud - 1.
  torch::Tensor mu_ratio_m1;

  //! cp/cpd - 1.
  torch::Tensor cp_ratio_m1;

  //! dimensionless enthalpy offset at T = 0
  torch::Tensor h0_R;

  //! stoichiometry matrix
  torch::Tensor stoich;

  //! options with which this `ThermoX` was constructed
  ThermoOptions options;

  ThermoXImpl() = default;
  explicit ThermoXImpl(const ThermoOptions& options_);
  void reset() override;

  //! \brief multi-component cp correction
  /*!
   * Eq.71 in Li2019
   * $ f_{\psi} = 1 + \sum_{i \in V \cup C} x_i(\sigma_{p,i} - 1) $
   *
   * \param xfrac mole fraction
   */
  torch::Tensor f_psi(torch::Tensor xfrac) const;

  //! \brief perform conversions
  torch::Tensor compute(
      std::string ab, std::initializer_list<torch::Tensor> args,
      torch::optional<torch::Tensor> out = torch::nullopt) const;

  //! \brief Calculate the equilibrium state given temperature and pressure
  torch::Tensor forward(torch::Tensor temp, torch::Tensor pres,
                        torch::Tensor xfrac);

 private:
  //! \brief Calculate mass fraction from mole fraction
  /*!
   * Eq.76 in Li2019
   * $ y_i = \frac{x_i \epsilon_i}{1. + \sum_{i \in V \cup C} x_i(\epsilon_i
   * - 1.)} $
   *
   * \param xfrac mole fraction, (..., 1 + ny)
   * \return mass fraction, (ny, ...)
   */
  torch::Tensor _xfrac_to_yfrac(torch::Tensor xfrac) const;

  //! \brief Calculate density from temperature, pressure and mole fraction
  /*!
   * Eq.94 in Li2019
   * $ \rho = \frac{P}{R_d T_v} $
   *
   * \param temp temperature, K
   * \param pres pressure, pa
   * \param xfrac mole fraction, (..., 1 + ny)
   */
  torch::Tensor _temp_to_dens(torch::Tensor temp, torch::Tensor pres,
                              torch::Tensor xfrac) const;
};
TORCH_MODULE(ThermoX);

}  // namespace kintera

#undef ADD_ARG
