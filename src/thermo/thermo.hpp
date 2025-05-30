#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// arg
#include <kintera/add_arg.h>

namespace kintera {

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
  ADD_ARG(std::vector<double>, cv_R);
  ADD_ARG(std::vector<double>, cp_R);
  ADD_ARG(std::vector<double>, u0_R);

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

  //! internal energy offset at T = Tref [J/kg]
  torch::Tensor u0;

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
   */
  torch::Tensor f_eps(torch::Tensor yfrac) const;

  //! \brief multi-component cv correction
  /*!
   * Eq.62 in Li2019
   * $ f_{\sigma} = 1 + \sum_{i \in V \cup C} y_i(\sigma_{v,i} - 1) $
   *
   * \param yfrac mass fraction
   */
  torch::Tensor f_sig(torch::Tensor yfrac) const;

  //! \brief Calculate the internal energy, J/m^3
  torch::Tensor get_intEng(torch::Tensor rho, torch::Tensor pres,
                           torch::Tensor yfrac) const;

  //! \brief Calculate the pressure, pa
  torch::Tensor get_pres(torch::Tensor rho, torch::Tensor intEng,
                         torch::Tensor yfrac) const;

  //! \brief Calculate the temperature, K
  torch::Tensor get_temp(torch::Tensor rho, torch::Tensor pres,
                         torch::Tensor yfrac) const {
    return pres / (rho * options.Rd() * f_eps(yfrac));
  }

  // torch::Tensor saturation_surplus(torch::Tensor var, int type = kTPMole)
  // const;

  //! \brief Calculate mole fraction from mass fraction
  /*!
   * Eq.77 in Li2019
   * $ x_i = \frac{y_i/\epsilon_i}{1. + \sum_{i \in V \cup C} y_i(1./\epsilon_i
   * - 1.)} $
   *
   * \param yfrac mass fraction, (nspecies, ...)
   * \return mole fraction, (..., 1 + nspecies)
   */
  torch::Tensor get_mole_fraction(torch::Tensor yfrac) const;

  //! \brief Calculate molar concentration (mol/m^3) fron mass fraction
  /*!
   * \param rho total density, kg/m^3
   * \param yfrac mass fraction, (nspecies, ...)
   * \return mole concentration, mol/m^3, (..., 1 + nspecies)
   */
  torch::Tensor get_concentration(torch::Tensor rho, torch::Tensor yfrac) const;

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
};
TORCH_MODULE(ThermoY);

//! Molar thermodynamics
class ThermoXImpl : public torch::nn::Cloneable<ThermoXImpl> {
 public:
  //! mu / mud - 1.
  torch::Tensor mu_ratio_m1;

  //! cp/cpd - 1.
  torch::Tensor cp_ratio_m1;

  //! enthalpy offset at T = Tref [J/mol]
  torch::Tensor h0;

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

  //! \brief Calculate mass fraction from mole fraction
  /*!
   * Eq.76 in Li2019
   * $ y_i = \frac{x_i \epsilon_i}{1. + \sum_{i \in V \cup C} x_i(\epsilon_i
   * - 1.)} $
   *
   * \param xfrac mole fraction, (..., 1 + nspecies)
   * \return mass fraction, (nspecies, ...)
   */
  torch::Tensor get_mass_fraction(torch::Tensor xfrac) const;

  //! \brief Calculate density from temperature, pressure and mole fraction
  /*!
   * Eq.94 in Li2019
   * $ \rho = \frac{P}{R_d T_v} $
   *
   * \param temp temperature, K
   * \param pres pressure, pa
   * \param xfrac mole fraction, (..., 1 + nspecies)
   */
  torch::Tensor get_density(torch::Tensor temp, torch::Tensor pres,
                            torch::Tensor xfrac) const;

  //! \brief Calculate the equilibrium state given temperature and pressure
  torch::Tensor forward(torch::Tensor temp, torch::Tensor pres,
                        torch::Tensor xfrac);
};
TORCH_MODULE(ThermoX);

}  // namespace kintera

#undef ADD_ARG
