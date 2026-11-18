#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <torch/torch.h>

// kintera
#include "nucleation.hpp"

namespace kintera {

class LogSVPFunc : public torch::autograd::Function<LogSVPFunc> {
 public:
  static constexpr bool is_traceable = true;

  //! the svp columns of op_ (null: the empty default) for the next calls,
  //! which read them from a ThermoTables (its svp spec and FuncTables; an
  //! inline column has a valid name there, overwritten by apply_inline)
  static void init(NucleationOptions const& op_);

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

 public:
  //! \brief Fill the svp part of a table (op null: the empty default), on
  //! `device`: the inline-SVP spec for the equilibrate kernels and the
  //! logsvp, logsvp_ddT function tables (an inline column has a valid name).
  //!
  //! svp_kind   is int32   [nreaction]   (0 named, 1 'ideal', 2 'antoine');
  //! svp_params is float64 [nreaction, KSVP_NPARAM] (zero-padded; parameters
  //! are unused for named rows).
  static void make_svp_spec(NucleationOptions const& op, ThermoTables& t,
                            torch::Device device);

 private:
  //! Inline formulas need their YAML parameters: 6 for 'ideal', 3 for 'antoine'
  static void check_inline(NucleationOptions const& op, size_t j, int kind) {
    size_t need = kind == 1 ? 6 : 3;
    TORCH_CHECK(
        j < op->svp_params().size() && op->svp_params()[j].size() == need,
        "inline svp formula '", op->logsvp()[j], "' (reaction ", j,
        ") lacks its ", need,
        " parameters; inline 'ideal'/'antoine' formulas "
        "must be defined in YAML");
  }

  static torch::Tensor evaluate(torch::Tensor const& temp, bool expanded,
                                FuncTable const& funcs,
                                torch::Tensor const& svp_kind,
                                torch::Tensor const& svp_params,
                                bool has_inline, bool deriv);

  //! Overwrite inline-parametrized columns in iter with eval_logsvp (or
  //! eval_logsvp_ddT when deriv is true). Named columns are left untouched.
  static void apply_inline(at::TensorIterator& iter,
                           torch::Tensor const& svp_kind,
                           torch::Tensor const& svp_params, bool has_inline,
                           bool deriv);

  //! the store's table for temp's device and dtype (built once)
  static ThermoTables const& tables(torch::Tensor const& temp);
  static SpeciesThermo _op;  // a ThermoOptions holding init()'s op only
};

}  // namespace kintera
