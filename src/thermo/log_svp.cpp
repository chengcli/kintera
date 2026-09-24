// torch
#include <torch/torch.h>

// kintera
#include <kintera/utils/utils_dispatch.hpp>

#include "log_svp.hpp"
#include "svp_eval.h"
#include "thermo_dispatch.hpp"

namespace kintera {

std::pair<torch::Tensor, torch::Tensor> LogSVPFunc::make_svp_spec(
    NucleationOptions const& op, torch::Device device) {
  auto const& names = op->logsvp();
  auto const& params = op->svp_params();
  int n = static_cast<int>(names.size());

  std::vector<int> kind(n, 0);
  std::vector<double> flat(static_cast<size_t>(n) * KSVP_NPARAM, 0.0);
  for (int j = 0; j < n; ++j) {
    if (names[j] == "ideal") {
      kind[j] = 1;
    } else if (names[j] == "antoine") {
      kind[j] = 2;
    }
    if (kind[j] != 0) check_inline(op, j, kind[j]);
    if (j < static_cast<int>(params.size())) {
      auto const& pj = params[j];
      int m = std::min(static_cast<int>(pj.size()), KSVP_NPARAM);
      for (int k = 0; k < m; ++k)
        flat[static_cast<size_t>(j) * KSVP_NPARAM + k] = pj[k];
    }
  }

  auto kind_t = torch::tensor(kind, torch::dtype(torch::kInt32)).to(device);
  auto par_t = torch::tensor(flat, torch::dtype(torch::kFloat64))
                   .to(device)
                   .view({n, KSVP_NPARAM});
  return {kind_t, par_t};
}

std::vector<std::string> LogSVPFunc::_logsvp = {};
std::vector<std::string> LogSVPFunc::_logsvp_ddT = {};
std::vector<int> LogSVPFunc::_formula_kind = {};
std::vector<std::vector<double>> LogSVPFunc::_svp_params = {};

void LogSVPFunc::apply_inline(at::TensorIterator& iter, bool deriv) {
  auto& out = iter.output();
  int64_t ncol = out.size(-1);
  std::vector<int> kind(ncol, 0);
  std::vector<double> flat(static_cast<size_t>(ncol) * KSVP_NPARAM, 0.0);
  bool any = false;
  for (int64_t j = 0; j < ncol; ++j) {
    if (j >= static_cast<int64_t>(_formula_kind.size())) break;
    if (_formula_kind[j] == 0) continue;  // named func-table formula
    if (j >= static_cast<int64_t>(_svp_params.size()) || _svp_params[j].empty())
      continue;
    kind[j] = _formula_kind[j];
    auto const& p = _svp_params[j];
    int m = std::min(static_cast<int>(p.size()), KSVP_NPARAM);
    for (int k = 0; k < m; ++k) flat[j * KSVP_NPARAM + k] = p[k];
    any = true;
  }
  if (!any) return;

  // Evaluate through the same scalar eval_logsvp / eval_logsvp_ddT as the
  // equilibrate kernels, element by element like the func-table dispatch, so
  // an inline curve with built-in constants is bit-for-bit its named twin
  // regardless of the compiler's floating-point contraction.
  auto kind_t =
      torch::tensor(kind, torch::dtype(torch::kInt32)).to(out.device());
  auto par_t =
      torch::tensor(flat, torch::dtype(torch::kFloat64)).to(out.device());
  at::native::call_logsvp_inline(out.device().type(), iter, kind_t, par_t,
                                 deriv);
}

torch::Tensor LogSVPFunc::grad(torch::Tensor const& temp, bool expanded) {
  auto vec = temp.sizes().vec();
  if (!expanded) {
    vec.push_back(_logsvp_ddT.size());
  }

  auto logsvp_ddT = torch::zeros(vec, temp.options());

  at::TensorIteratorConfig iter_config;
  iter_config.resize_outputs(false)
      .check_all_same_dtype(true)
      .declare_static_shape(logsvp_ddT.sizes(),
                            /*squash_dim=*/{logsvp_ddT.dim() - 1})
      .add_output(logsvp_ddT);

  if (expanded) {
    iter_config.add_input(temp);
  } else {
    iter_config.add_owned_input(temp.unsqueeze(-1));
  }

  auto iter = iter_config.build();
  at::native::call_func1(logsvp_ddT.device().type(), iter, _logsvp_ddT);

  apply_inline(iter, /*deriv=*/true);

  return logsvp_ddT;
}

torch::Tensor LogSVPFunc::call(torch::Tensor const& temp, bool expanded) {
  auto vec = temp.sizes().vec();
  if (!expanded) {
    vec.push_back(_logsvp.size());
  }

  auto logsvp = torch::zeros(vec, temp.options());

  at::TensorIteratorConfig iter_config;
  iter_config.resize_outputs(false)
      .check_all_same_dtype(true)
      .declare_static_shape(logsvp.sizes(),
                            /*squash_dim=*/{logsvp.dim() - 1})
      .add_output(logsvp);

  if (expanded) {
    iter_config.add_input(temp);
  } else {
    iter_config.add_owned_input(temp.unsqueeze(-1));
  }

  auto iter = iter_config.build();
  at::native::call_func1(logsvp.device().type(), iter, _logsvp);

  apply_inline(iter, /*deriv=*/false);

  return logsvp;
}

torch::Tensor LogSVPFunc::forward(torch::autograd::AutogradContext* ctx,
                                  torch::Tensor const& temp) {
  ctx->save_for_backward({temp});
  return call(temp, true);
}

std::vector<torch::Tensor> LogSVPFunc::backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<torch::Tensor> grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto logsvp_ddT = grad(/*temp=*/saved[0], true);
  return {grad_outputs[0] * logsvp_ddT};
}

}  // namespace kintera
