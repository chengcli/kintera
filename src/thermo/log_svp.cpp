// torch
#include <torch/torch.h>

// kintera
#include <kintera/utils/utils_dispatch.hpp>

#include "log_svp.hpp"

namespace kintera {

std::vector<user_func1> LogSVPFunc::_logsvp = {};
std::vector<user_func1> LogSVPFunc::_logsvp_ddT = {};

torch::Tensor LogSVPFunc::grad(torch::Tensor const &temp, bool expanded) {
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
  at::native::call_func1(logsvp_ddT.device().type(), iter, _logsvp_ddT.data());

  return logsvp_ddT;
}

torch::Tensor LogSVPFunc::call(torch::Tensor const &temp, bool expanded) {
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
  at::native::call_func1(logsvp.device().type(), iter, _logsvp.data());

  return logsvp;
}

torch::Tensor LogSVPFunc::forward(torch::autograd::AutogradContext *ctx,
                                  torch::Tensor const &temp) {
  ctx->save_for_backward({temp});
  return call(temp, true);
}

std::vector<torch::Tensor> LogSVPFunc::backward(
    torch::autograd::AutogradContext *ctx,
    std::vector<torch::Tensor> grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto logsvp_ddT = grad(/*temp=*/saved[0], true);
  return {grad_outputs[0] * logsvp_ddT};
}

}  // namespace kintera
