// kintera
#include <kintera/constants.h>

#include <kintera/thermo/eval_uhs.hpp>

#include "kinetics.hpp"

namespace kintera {

void KineticsImpl::_jacobian_mass_action(
    torch::Tensor temp, torch::Tensor conc, torch::Tensor cvol,
    torch::Tensor rate, torch::optional<torch::Tensor> logrc_ddT, int begin,
    int end, torch::Tensor &out) const {
  auto stoich_local = (-stoich.slice(1, begin, end)).clamp_min(0.0).t();

  // forward reaction mask
  out = stoich_local * rate.unsqueeze(-1) / conc.unsqueeze(-2);

  // add temperature derivative if provided
  if (logrc_ddT.has_value()) {
    auto intEng = eval_intEng_R(temp, conc, options) * constants::Rgas;
    out -= rate.unsqueeze(-1) *
           logrc_ddT.value().slice(-1, begin, end).unsqueeze(-1) *
           intEng.unsqueeze(-2) / cvol.unsqueeze(-1).unsqueeze(-1);
  }
}

}  // namespace kintera
