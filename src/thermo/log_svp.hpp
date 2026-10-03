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

  static void init(NucleationOptions const& op_) {
    // a null NucleationOptions (e.g. nucleation(None)) is the empty default
    auto const& op = op_ ? op_ : NucleationOptionsImpl::create();
    _logsvp = op->logsvp();

    auto spec = make_svp_spec(op, torch::kCPU);
    _svp_kind = spec.first;
    _svp_params = spec.second;

    // Inline columns use eval_logsvp below. Give func-table dispatch a valid
    // sentinel name; apply_inline overwrites those columns afterwards.
    _has_inline = false;
    auto kind = _svp_kind.accessor<int, 1>();
    for (size_t i = 0; i < _logsvp.size(); ++i) {
      if (kind[i] != 0) {
        _has_inline = true;
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

 public:
  //! \brief Build inline-SVP spec tensors {kind, params} for the equilibrate
  //! kernels from a nucleation option set, on the given device.
  //!
  //! kind   is int32   [nreaction]   (0 named, 1 'ideal', 2 'antoine');
  //! params is float64 [nreaction, KSVP_NPARAM] (zero-padded; named rows are
  //! zero).
  static std::pair<torch::Tensor, torch::Tensor> make_svp_spec(
      NucleationOptions const& op, torch::Device device);

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
                                std::vector<std::string> const& names,
                                torch::Tensor const& svp_kind,
                                torch::Tensor const& svp_params,
                                bool has_inline, bool deriv);

  //! Overwrite inline-parametrized columns in iter with eval_logsvp (or
  //! eval_logsvp_ddT when deriv is true). Named columns are left untouched.
  static void apply_inline(at::TensorIterator& iter,
                           torch::Tensor const& svp_kind,
                           torch::Tensor const& svp_params, bool has_inline,
                           bool deriv);

  static std::vector<std::string> _logsvp;
  static std::vector<std::string> _logsvp_ddT;
  static torch::Tensor _svp_kind;
  static torch::Tensor _svp_params;
  static bool _has_inline;
};

}  // namespace kintera
