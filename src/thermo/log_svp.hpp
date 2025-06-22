#pragma once

// torch
#include <torch/torch.h>

// kintera
#include "nucleation.hpp"

namespace kintera {

class LogSVPFunc : public torch::autograd::Function<LogSVPFunc> {
 public:
  static constexpr bool is_traceable = true;

  static void init(NucleationOptions const& op) {
    _logsvp = op.logsvp();
    _logsvp_ddT = op.logsvp_ddT();
  }

  //! \brief Computes the gradient of logarithm of the saturation vapor pressure
  /*!
   * \param temp          Temperature tensor
   * \param expanded      If true, the input temperature is already expanded
   */
  static torch::Tensor grad(torch::Tensor const& temp, bool expanded = false);

  //! \brief Computes the logarithm of the saturation vapor pressure
  /*!
   * \param temp          Temperature tensor
   * \param expanded      If true, the input temperature is already expanded
   */
  static torch::Tensor call(torch::Tensor const& temp, bool expanded = false);

  //! \brief Computes the logarithm of the saturation vapor pressure
  /*!
   * This function is not to be used directly, but rather through the
   * 'apply' method of the autograd function.
   *
   * Example usage:
   *  \code{.cpp}
   *    torch::Tensor temp = ...; // Temperature tensor
   *    torch::Tensor logsvp = LogSVPFunc::apply(temp);
   *  \endcode
   *
   * \param ctx           Autograd context for storing state
   * \param temp          Temperature tensor (expanded)
   */
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor const& temp);

  //! \brief Computes the gradient of the logarithm of the saturation vapor
  //! pressure
  /*!
   * This function is not to be used directly, but rather through the
   * 'backward' method of the autograd function.
   *
   * Example usage:
   *  \code{.cpp}
   *    torch::Tensor temp = ...; // Temperature tensor
   *    temp.requires_grad_(); // Ensure temp requires gradient
   *    torch::Tensor logsvp = LogSVPFunc::apply(temp);
   *    logsvp.backward(torch::ones_like(logsvp)); // Backward pass
   *    std::cout << "Gradient: " << temp.grad() << std::endl;
   *  \endcode
   *
   *  \param ctx            Autograd context for storing state
   *  \param grad_outputs   Gradient of the output tensor
   */
  static std::vector<torch::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      std::vector<torch::Tensor> grad_outputs);

 private:
  static std::vector<user_func1> _logsvp;
  static std::vector<user_func1> _logsvp_ddT;
};

}  // namespace kintera
