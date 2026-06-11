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
    _logsvp = op->logsvp();
    _svp_params = op->svp_params();

    // Classify each column: 0 = named func-table formula, 1 = inline 'ideal',
    // 2 = inline 'antoine'. For inline columns, swap in a valid sentinel name so
    // the func-table dispatch does not fail; the column is overwritten with the
    // analytic torch-op result afterwards.
    _formula_kind.assign(_logsvp.size(), 0);
    for (size_t i = 0; i < _logsvp.size(); ++i) {
      if (_logsvp[i] == "ideal") {
        _formula_kind[i] = 1;
        _logsvp[i] = "h2o_ideal";
      } else if (_logsvp[i] == "antoine") {
        _formula_kind[i] = 2;
        _logsvp[i] = "h2o_ideal";
      }
    }

    _logsvp_ddT = _logsvp;
    for (auto& name : _logsvp_ddT) name += "_ddT";
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
  //! Overwrite the inline-parametrized columns of \p out (and its temperature
  //! derivative when \p deriv is true) with the analytic 'ideal'/'antoine' form
  //! evaluated from _svp_params. Named columns are left untouched.
  static void apply_inline(torch::Tensor& out, torch::Tensor const& temp,
                           bool expanded, bool deriv);

  static std::vector<std::string> _logsvp;
  static std::vector<std::string> _logsvp_ddT;
  //! per-column formula kind: 0 named, 1 'ideal', 2 'antoine'
  static std::vector<int> _formula_kind;
  //! per-column inline parameters (empty for named columns)
  static std::vector<std::vector<double>> _svp_params;
};

}  // namespace kintera
