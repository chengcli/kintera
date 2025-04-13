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

struct ThermodynamicsOptions {
  ThermodynamicsOptions() = default;

  ADD_ARG(double, gammad) = 1.4;
  ADD_ARG(double, Rd) = 287.0;
  ADD_ARG(double, Tref) = 300.0;

  ADD_ARG(std::vector<int>, vapor_ids);
  ADD_ARG(std::vector<int>, cloud_ids);

  ADD_ARG(std::vector<double>, mu_ratio_m1);
  ADD_ARG(std::vector<double>, cv_ratio_m1);
  ADD_ARG(std::vector<double>, cp_ratio_m1);
  ADD_ARG(std::vector<double>, h0);

  ADD_ARG(int, max_iter) = 5;
  ADD_ARG(double, ftol) = 1e-2;
  ADD_ARG(double, rtol) = 1e-4;
  ADD_ARG(double, boost) = 256;

  ADD_ARG(EquationOfStateOptions, eos);
  ADD_ARG(CondensationOptions, cond);
};

class ThermodynamicsImpl : public torch::nn::Cloneable<ThermodynamicsImpl> {
 public:
  //! mud / mu - 1.
  torch::Tensor mu_ratio_m1;

  //! cv/cvd - 1.
  torch::Tensor cv_ratio_m1;

  //! cp/cpd - 1.
  torch::Tensor cp_ratio_m1;

  //! enthalpy offset at T = Tref
  torch::Tensor h0;

  //! submodules
  Condensation pcond = nullptr;
  EquationOfState peos = nullptr;

  //! options with which this `Thermodynamics` was constructed
  ThermodynamicsOptions options;

  ThermodynamicsImpl() = default;
  explicit ThermodynamicsImpl(const ThermodynamicsOptions& options_);
  void reset() override;

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

  //! \brief Calculate molar concentration fron conserved variables
  /*!
   * \param yfrac mass fraction, (nmass, ...)
   * \return molar concentration, (..., 1 + nmass)
   */
  torch::Tensor get_mole_concentration(torch::Tensor rho, torch::Tensor yfrac) const;

  //! \brief Perform saturation adjustment
  /*!
   * \param rho density
   * \param yfrac mass fraction
   * \param intEng internal energy
   * \return adjusted density
   */
  torch::Tensor forward(torch::Tensor rho, torch::Tensor yfrac,
                        torch::Tensor intEng);

  //! \brief Calculate mass fraction from mole fraction
  /*!
   * \param xfrac mole fraction, (..., 1 + nmass)
   * \return mass fraction, (nmass, ...)
   */
  torch::Tensor get_mass_fraction(torch::Tensor xfrac) const;

  //! \brief Calculate the equilibrium state given temperature and pressure
  torch::Tensor equilibrate_tp(torch::Tensor temp, torch::Tensor pres,
                               torch::Tensor yfrac) const;
};
TORCH_MODULE(Thermodynamics);

}  // namespace kintera
