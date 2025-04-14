#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include <kintera/add_arg.h>

#include <kintera/eos/equation_of_state.hpp>

#include "condensation.hpp"

namespace kintera {

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

  ADD_ARG(std::vector<double>, mu_ratio_m1);
  ADD_ARG(std::vector<double>, cv_ratio_m1);
  ADD_ARG(std::vector<double>, cp_ratio_m1);
  ADD_ARG(std::vector<double>, h0_R);

  ADD_ARG(int, max_iter) = 5;
  ADD_ARG(double, ftol) = 1e-2;
  ADD_ARG(double, rtol) = 1e-4;
  ADD_ARG(double, boost) = 256;

  ADD_ARG(EquationOfStateOptions, eos);
  ADD_ARG(CondensationOptions, cond);
};

class ThermoYImpl : public torch::nn::Cloneable<ThermoYImpl> {
 public:
  //! mud / mu - 1.
  torch::Tensor mu_ratio_m1;

  //! cv/cvd - 1.
  torch::Tensor cv_ratio_m1;

  //! cp/cpd - 1.
  torch::Tensor cp_ratio_m1;

  //! enthalpy offset at T = Tref
  torch::Tensor h0_R;

  //! submodules
  Condensation pcond = nullptr;
  EquationOfState peos = nullptr;

  //! options with which this `ThermoY` was constructed
  ThermoOptions options;

  ThermoYImpl() = default;
  explicit ThermoYImpl(const ThermoOptions& options_);
  void reset() override;

  int nspecies() const {
    return static_cast<int>(options.vapor_ids().size() +
                            options.cloud_ids().size());
  }

  //! \brief Inverse of the mean molecular weight
  /*!
   * Eq.16 in Li2019
   * $ \frac{R}{R_d} = \frac{\mu_d}{\mu}$
   *
   * \param yfrac mass fraction
   * \return $1/\mu$
   */
  torch::Tensor f_eps(torch::Tensor yfrac) const;
  torch::Tensor f_sig(torch::Tensor yfrac) const;
  torch::Tensor f_psi(torch::Tensor yfrac) const;

  //! \brief Calculate the molecular weights
  /*!
   * \return a vector of molecular weights
   */
  torch::Tensor get_mu() const;

  //! \brief Calculate the specific heat at constant volume
  /*!
   * \return a vector of specific heats
   */
  torch::Tensor get_cv() const;

  //! \brief Calculate the specific heat at constant pressure
  /*!
   * \return a vector of specific heats
   */
  torch::Tensor get_cp() const;

  // torch::Tensor saturation_surplus(torch::Tensor var, int type = kTPMole)
  // const;

  //! \brief Calculate mole fraction from mass fraction
  /*!
   * \param yfrac mass fraction, (nmass, ...)
   * \return mole fraction, (..., 1 + nmass)
   */
  torch::Tensor get_mole_fraction(torch::Tensor yfrac) const;

  //! \brief Calculate molar concentration fron mass fraction
  /*!
   * \param rho total density, kg/m^3
   * \param yfrac mass fraction, (nmass, ...)
   * \return mole concentration, (..., 1 + nmass)
   */
  torch::Tensor get_concentration(torch::Tensor rho, torch::Tensor yfrac) const;

  //! \brief Perform saturation adjustment
  /*!
   * \param rho density
   * \param yfrac mass fraction
   * \param intEng internal energy
   * \return adjusted density
   */
  torch::Tensor forward(torch::Tensor rho, torch::Tensor intEng,
                        torch::Tensor yfrac);
};
TORCH_MODULE(ThermoY);

class ThermoXImpl : public torch::nn::Cloneable<ThermoXImpl> {
 public:
  //! mud / mu - 1.
  torch::Tensor mu_ratio_m1;

  //! cv/cvd - 1.
  torch::Tensor cv_ratio_m1;

  //! cp/cpd - 1.
  torch::Tensor cp_ratio_m1;

  //! enthalpy offset at T = Tref
  torch::Tensor h0_R;

  //! submodules
  Condensation pcond = nullptr;
  EquationOfState peos = nullptr;

  //! options with which this `ThermoX` was constructed
  ThermoOptions options;

  ThermoXImpl() = default;
  explicit ThermoXImpl(const ThermoOptions& options_);
  void reset() override;

  //! \brief Calculate the molecular weights
  /*!
   * \return a vector of molecular weights
   */
  torch::Tensor get_mu() const;

  //! \brief Calculate the specific heat at constant volume
  /*!
   * \return a vector of specific heats
   */
  torch::Tensor get_cv() const;

  //! \brief Calculate the specific heat at constant pressure
  /*!
   * \return a vector of specific heats
   */
  torch::Tensor get_cp() const;

  //! \brief Calculate mass fraction from mole fraction
  /*!
   * \param xfrac mole fraction, (..., 1 + nmass)
   * \return mass fraction, (nmass, ...)
   */
  torch::Tensor get_mass_fraction(torch::Tensor xfrac) const;

  //! \brief Calculate mass density fron mole fraction
  /*!
   * \param conc total concentration, mole/m^3
   * \param xfrac mole fraction, (..., 1 + nmass)
   * \return mass density, (1 + nmass, ...)
   */
  torch::Tensor get_density(torch::Tensor conc, torch::Tensor xfrac) const;

  //! \brief Calculate the equilibrium state given temperature and pressure
  torch::Tensor forward(torch::Tensor temp, torch::Tensor pres,
                        torch::Tensor xfrac);
};
TORCH_MODULE(ThermoX);

}  // namespace kintera
