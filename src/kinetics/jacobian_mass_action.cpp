// kintera
#include <kintera/thermo/eval_uhs.hpp>

#include "jacobian.hpp"

namespace kintera {

torch::Tensor jacobian_mass_action(torch::Tensor rate, torch::Tensor stoich,
                                   torch::Tensor conc,
                                   torch::optional<torch::Tensor> temp,
                                   torch::optional<torch::Tensor> rc_ddT,
                                   torch::optional<SpeciesThermo> op) {
  // temp, rc_ddT and op should be all provided or neither
  TORCH_CHECK(temp.has_value() == rc_ddT.has_value(),
              "Both rc_ddT and op must be provided or neither.");

  TORCH_CHECK(rc_ddT.has_value() == op.has_value(),
              "Both rc_ddT and op must be provided or neither.");

  // broadcast stoich to match rate shape
  std::vector<int64_t> vec(rate.dim(), 1);

  vec[rate.dim() - 1] = stoich.size(0);
  vec.push_back(stoich.size(1));

  auto rate_sign = -rate.sign().unsqueeze(-2);
  auto stoich_sign = stoich.view(vec) * rate_sign;
  auto stoich_all = stoich_sign.clamp_min(0.0).transpose(-1, -2);

  // forward reaction mask
  auto jacobian = stoich_all * rate.unsqueeze(-1) / conc.unsqueeze(-2);

  // add temperature derivative if provided
  if (rc_ddT.has_value()) {
    // TODO(cli) narrow to thermo species
    auto intEng_R = eval_intEng_R(temp.value(), conc, op.value());
    auto cv_R = eval_cv_R(temp.value(), conc, op.value());
    auto cv_vol = (cv_R * conc).sum(-1, /*keepdim=*/true);
    jacobian -= rate.abs().unsqueeze(-1) * rc_ddT.value().unsqueeze(-1) *
                intEng_R.unsqueeze(-2) / cv_vol.unsqueeze(-1);
  }

  return jacobian;
}

}  // namespace kintera
