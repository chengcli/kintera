#pragma once

// torch
#include <torch/torch.h>

namespace kintera {

inline torch::Tensor svp_sio_Visscher(torch::Tensor T) {
  auto log10p = 8.203 - 25898.9 / T;
  return 1.E5 * torch::pow(10., log10p);
}

inline torch::Tensor svp_sio_Visscher_logsvp_ddT(torch::Tensor T) {
  return 25898.9 / (T * T);
}

}  // namespace kintera
