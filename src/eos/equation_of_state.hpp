#pragma once

// torch
#include <torch/torch.h>

// kintera
#include <kintera/constants.h>

// arg
#include <kintera/add_arg.h>

namespace kintera {

struct EquationOfStateOptions {
  EquationOfStateOptions() = default;

  ADD_ARG(std::string, file) = "";
  ADD_ARG(std::string, type) = "ideal_gas";
  ADD_ARG(double, mu) = 2.0e-3;  // default to hydrogen
  ADD_ARG(double, gamma_ref) = 1.4;
};

class EquationOfStateImpl {
 public:
  //! options with which this `EquationOfStateImpl` was constructed
  EquationOfStateOptions options;

  EquationOfStateImpl() = default;
  explicit EquationOfStateImpl(const EquationOfStateOptions& options_);

  virtual torch::Tensor get_gamma(torch::Tensor rho, torch::Tensor intEng);

  virtual torch::Tensor get_sound_speed(torch::Tensor rho, torch::Tensor pres);

  virtual torch::Tensor get_enthalpy(torch::Tensor rho, torch::Tensor pres) {
    return get_intEng(rho, pres) + pres / rho;
  }

  //! \brief Calculate internal energy from density and pressure
  virtual torch::Tensor get_intEng(torch::Tensor rho, torch::Tensor pres) {
    return pres / (options.gamma_ref() - 1.) / rho;
  }

  //! \brief Calculate pressure from density and internal energy
  virtual torch::Tensor get_pres(torch::Tensor rho, torch::Tensor intEng) {
    return (options.gamma_ref() - 1.) * intEng * rho;
  }

  //! \brief Calculate temperature from density and pressure
  virtual torch::Tensor get_temp(torch::Tensor rho, torch::Tensor pres) {
    return (pres * options.mu()) / (rho * constants::Rgas);
  }
};

using EquationOfState = std::shared_ptr<EquationOfStateImpl>;

}  // namespace kintera

#undef ADD_ARG
