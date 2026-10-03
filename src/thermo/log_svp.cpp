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
torch::Tensor LogSVPFunc::_svp_kind =
    torch::empty({0}, torch::dtype(torch::kInt32));
torch::Tensor LogSVPFunc::_svp_params =
    torch::empty({0, KSVP_NPARAM}, torch::dtype(torch::kFloat64));
bool LogSVPFunc::_has_inline = false;

void LogSVPFunc::apply_inline(at::TensorIterator& iter,
                              torch::Tensor const& svp_kind,
                              torch::Tensor const& svp_params, bool has_inline,
                              bool deriv) {
  if (!has_inline) return;

  // Evaluate through the same scalar eval_logsvp / eval_logsvp_ddT as the
  // equilibrate kernels, element by element like the func-table dispatch, so
  // an inline curve with built-in constants is bit-for-bit its named twin
  // regardless of the compiler's floating-point contraction.
  at::native::call_logsvp_inline(iter.output().device().type(), iter, svp_kind,
                                 svp_params, deriv);
}

torch::Tensor LogSVPFunc::evaluate(torch::Tensor const& temp, bool expanded,
                                   std::vector<std::string> const& names,
                                   torch::Tensor const& svp_kind,
                                   torch::Tensor const& svp_params,
                                   bool has_inline, bool deriv) {
  auto vec = temp.sizes().vec();
  if (!expanded) {
    vec.push_back(names.size());
  }

  auto result = torch::zeros(vec, temp.options());

  at::TensorIteratorConfig iter_config;
  iter_config.resize_outputs(false)
      .check_all_same_dtype(true)
      .declare_static_shape(result.sizes(),
                            /*squash_dim=*/{result.dim() - 1})
      .add_output(result);

  if (expanded) {
    iter_config.add_input(temp);
  } else {
    iter_config.add_owned_input(temp.unsqueeze(-1));
  }

  auto iter = iter_config.build();
  at::native::call_func1(result.device().type(), iter, names);
  apply_inline(iter, svp_kind, svp_params, has_inline, deriv);
  return result;
}

torch::Tensor LogSVPFunc::grad(torch::Tensor const& temp, bool expanded) {
  torch::Tensor svp_kind;
  torch::Tensor svp_params;
  if (_has_inline) {
    svp_kind = _svp_kind.to(temp.device());
    svp_params = _svp_params.to(temp.device());
  }
  return evaluate(temp, expanded, _logsvp_ddT, svp_kind, svp_params,
                  _has_inline, /*deriv=*/true);
}

torch::Tensor LogSVPFunc::call(torch::Tensor const& temp, bool expanded) {
  torch::Tensor svp_kind;
  torch::Tensor svp_params;
  if (_has_inline) {
    svp_kind = _svp_kind.to(temp.device());
    svp_params = _svp_params.to(temp.device());
  }
  return evaluate(temp, expanded, _logsvp, svp_kind, svp_params, _has_inline,
                  /*deriv=*/false);
}

torch::Tensor LogSVPFunc::forward(torch::autograd::AutogradContext* ctx,
                                  torch::Tensor const& temp) {
  auto names = _logsvp;
  auto names_ddT = _logsvp_ddT;
  bool has_inline = _has_inline;
  torch::Tensor svp_kind;
  torch::Tensor svp_params;
  if (has_inline) {
    svp_kind = _svp_kind.to(temp.device()).clone();
    svp_params = _svp_params.to(temp.device()).clone();
    ctx->save_for_backward({temp, svp_kind, svp_params});
  } else {
    ctx->save_for_backward({temp});
  }

  ctx->saved_data["logsvp_ddT"] = names_ddT;
  ctx->saved_data["has_inline"] = has_inline;
  return evaluate(temp, true, names, svp_kind, svp_params, has_inline,
                  /*deriv=*/false);
}

std::vector<torch::Tensor> LogSVPFunc::backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<torch::Tensor> grad_outputs) {
  auto saved = ctx->get_saved_variables();
  std::vector<std::string> names_ddT;
  for (auto const& name : ctx->saved_data.at("logsvp_ddT").toListRef()) {
    names_ddT.push_back(name.toStringRef());
  }
  bool has_inline = ctx->saved_data.at("has_inline").toBool();
  torch::Tensor svp_kind;
  torch::Tensor svp_params;
  if (has_inline) {
    svp_kind = saved[1];
    svp_params = saved[2];
  }
  auto logsvp_ddT = evaluate(/*temp=*/saved[0], /*expanded=*/true, names_ddT,
                             svp_kind, svp_params, has_inline, /*deriv=*/true);
  return {grad_outputs[0] * logsvp_ddT};
}

}  // namespace kintera
