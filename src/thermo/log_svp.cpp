// torch
#include <torch/torch.h>

// kintera
#include <kintera/utils/utils_dispatch.hpp>

#include "log_svp.hpp"
#include "svp_eval.h"

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

void LogSVPFunc::apply_inline(torch::Tensor& out, torch::Tensor const& temp,
                              bool expanded, bool deriv) {
  int64_t ncol = out.size(-1);
  for (int64_t j = 0; j < ncol; ++j) {
    if (j >= static_cast<int64_t>(_formula_kind.size())) break;
    int kind = _formula_kind[j];
    if (kind == 0) continue;  // named func-table formula, leave as computed
    if (j >= static_cast<int64_t>(_svp_params.size()) || _svp_params[j].empty())
      continue;
    auto const& p = _svp_params[j];

    // Temperature for this column; the func-table dispatch broadcasts the same
    // temperature across every column, so we reproduce that here. In the
    // expanded path the temperature already carries the column dimension.
    auto t = expanded ? temp.select(-1, j) : temp;

    torch::Tensor val;
    if (kind == 1) {  // 'ideal': {T3, P3, beta, gamma, betas, gammas}
      double T3 = p[0], P3 = p[1], betal = p[2], gammal = p[3], betas = p[4],
             gammas = p[5];
      auto liquid = t > T3;  // matches 'T > tr' in the named func-table forms
      if (!deriv) {
        auto logt = torch::log(t / T3);
        auto vl = (1.0 - T3 / t) * betal - gammal * logt + std::log(P3);
        auto vs = (1.0 - T3 / t) * betas - gammas * logt + std::log(P3);
        val = torch::where(liquid, vl, vs);
      } else {
        auto t2 = t * t;
        auto dl = betal * T3 / t2 - gammal / t;
        auto ds = betas * T3 / t2 - gammas / t;
        val = torch::where(liquid, dl, ds);
      }
    } else {  // kind == 2, 'antoine': {A, B, C}
      double A = p[0], B = p[1], C = p[2];
      if (!deriv) {
        val = std::log(1.0e5) + (A - B / (t + C)) * std::log(10.0);
      } else {
        auto tc = t + C;
        val = B * std::log(10.0) / (tc * tc);
      }
    }

    out.select(-1, j).copy_(val);
  }
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

  apply_inline(logsvp_ddT, temp, expanded, /*deriv=*/true);

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

  apply_inline(logsvp, temp, expanded, /*deriv=*/false);

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
