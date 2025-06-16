// torch
#include <torch/torch.h>

// kintera
#include <kintera/utils/utils_dispatch.hpp>

#include "log_svp.hpp"

namespace kintera {

std::vector<Nucleation> LogSVPFunc::_react = {};

torch::Tensor LogSVPFunc::grad(torch::Tensor const &temp) {
  auto vec = temp.sizes().vec();
  vec.push_back(_react.size());

  auto logsvp_ddT = torch::zeros(vec, temp.options());
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(logsvp_ddT.sizes(),
                                        /*squash_dim=*/{logsvp_ddT.dim() - 1})
                  .add_output(logsvp_ddT)
                  .add_owned_input(temp.unsqueeze(-1))
                  .build();

  user_func1 *logsvp_func_ddT = new user_func1[_react.size()];
  for (int i = 0; i < _react.size(); ++i) {
    logsvp_func_ddT[i] = _react[i].func_ddT();
  }

  at::native::call_func1(logsvp_ddT.device().type(), iter, logsvp_func_ddT);
  delete[] logsvp_func_ddT;

  return logsvp_ddT;
}

torch::Tensor LogSVPFunc::forward(torch::autograd::AutogradContext *ctx,
                                  torch::Tensor const &temp) {
  auto vec = temp.sizes().vec();
  vec.push_back(_react.size());

  auto logsvp = torch::zeros(vec, temp.options());
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(logsvp.sizes(),
                                        /*squash_dim=*/{logsvp.dim() - 1})
                  .add_output(logsvp)
                  .add_owned_input(temp.unsqueeze(-1))
                  .build();

  user_func1 *logsvp_func = new user_func1[_react.size()];
  for (int i = 0; i < _react.size(); ++i) {
    logsvp_func[i] = _react[i].func();
  }

  at::native::call_func1(logsvp.device().type(), iter, logsvp_func);
  delete[] logsvp_func;

  ctx->save_for_backward({temp});
  return logsvp;
}

std::vector<torch::Tensor> LogSVPFunc::backward(
    torch::autograd::AutogradContext *ctx,
    std::vector<torch::Tensor> grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto logsvp_ddT = grad(/*temp=*/saved[0]);
  return {grad_outputs[0] * logsvp_ddT};
}

}  // namespace kintera
