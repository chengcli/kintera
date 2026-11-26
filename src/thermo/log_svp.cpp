// torch
#include <torch/torch.h>

// kintera
#include <kintera/utils/utils_dispatch.hpp>

#include "log_svp.hpp"
#include "svp_eval.h"
#include "thermo.hpp"
#include "thermo_dispatch.hpp"

namespace kintera {

void LogSVPFunc::make_svp_spec(NucleationOptions const& op_, ThermoTables& t,
                               torch::Device device) {
  auto const& op = op_ ? op_ : NucleationOptionsImpl::create();
  auto names = op->logsvp();  // inline columns get a valid name below
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
    if (kind[j] != 0) {
      check_inline(op, j, kind[j]);
      names[j] = "h2o_ideal", t.svp_inline = true;
    }
    if (j < static_cast<int>(params.size())) {
      auto const& pj = params[j];
      int m = std::min(static_cast<int>(pj.size()), KSVP_NPARAM);
      for (int k = 0; k < m; ++k)
        flat[static_cast<size_t>(j) * KSVP_NPARAM + k] = pj[k];
    }
  }

  t.svp_kind = torch::tensor(kind, torch::dtype(torch::kInt32)).to(device);
  t.svp_params = torch::tensor(flat, torch::dtype(torch::kFloat64))
                     .to(device)
                     .view({n, KSVP_NPARAM});
  auto names_ddT = names;  // every name, as the solvers and grad() had it
  for (auto& name : names_ddT) name += "_ddT";
  t.logsvp = at::native::fill_func_table(device.type(), names, 1, device);
  t.logsvp_ddT =
      at::native::fill_func_table(device.type(), names_ddT, 1, device);
}

ThermoTables const& LogSVPFunc::tables(torch::Tensor const& temp,
                                       NucleationOptions const& op_) {
  auto op = ThermoOptionsImpl::create();  // no species: the svp columns only
  op->nucleation() = op_ ? op_ : NucleationOptionsImpl::create();
  return *tables_for(op, temp);  // the store keeps it
}

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
                                   FuncTable const& funcs,
                                   torch::Tensor const& svp_kind,
                                   torch::Tensor const& svp_params,
                                   bool has_inline, bool deriv) {
  auto vec = temp.sizes().vec();
  if (!expanded) {
    vec.push_back(funcs.size(0));
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
  at::native::call_func1(result.device().type(), iter, funcs);
  apply_inline(iter, svp_kind, svp_params, has_inline, deriv);
  return result;
}

torch::Tensor LogSVPFunc::grad(torch::Tensor const& temp,
                               NucleationOptions const& op, bool expanded) {
  auto const& t = tables(temp, op);
  return evaluate(temp, expanded, t.logsvp_ddT, t.svp_kind, t.svp_params,
                  t.svp_inline, /*deriv=*/true);
}

torch::Tensor LogSVPFunc::call(torch::Tensor const& temp,
                               NucleationOptions const& op, bool expanded) {
  auto const& t = tables(temp, op);
  return evaluate(temp, expanded, t.logsvp, t.svp_kind, t.svp_params,
                  t.svp_inline, /*deriv=*/false);
}

torch::Tensor LogSVPFunc::forward(torch::autograd::AutogradContext* ctx,
                                  torch::Tensor const& temp,
                                  NucleationOptions const& op) {
  auto const& t = tables(temp, op);
  ctx->save_for_backward({temp, t.logsvp_ddT, t.svp_kind, t.svp_params});
  ctx->saved_data["has_inline"] = t.svp_inline;
  return evaluate(temp, true, t.logsvp, t.svp_kind, t.svp_params, t.svp_inline,
                  /*deriv=*/false);
}

std::vector<torch::Tensor> LogSVPFunc::backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<torch::Tensor> grad_outputs) {
  auto saved = ctx->get_saved_variables();
  bool has_inline = ctx->saved_data.at("has_inline").toBool();
  auto logsvp_ddT = evaluate(/*temp=*/saved[0], /*expanded=*/true, saved[1],
                             saved[2], saved[3], has_inline, /*deriv=*/true);
  return {grad_outputs[0] * logsvp_ddT, torch::Tensor()};  // none for op
}

}  // namespace kintera
