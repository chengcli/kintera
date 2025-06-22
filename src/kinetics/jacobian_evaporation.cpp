// kintera
#include <kintera/constants.h>

#include <kintera/thermo/eval_uhs.hpp>
#include <kintera/thermo/log_svp.hpp>
#include <kintera/thermo/relative_humidity.hpp>

#include "kinetics.hpp"

namespace kintera {

void KineticsImpl::_jacobian_evaporation(
    torch::Tensor temp, torch::Tensor conc, torch::Tensor cvol,
    torch::Tensor rate, torch::optional<torch::Tensor> logrc_ddT, int begin,
    int end, torch::Tensor& out) const {
  auto stoich_local = stoich.slice(1, begin, end);

  // evaluate relative humidity
  auto rh = relative_humidity(temp, conc, -stoich_local, options.evaporation());

  // mark reactants
  auto sm = stoich_local.clamp_max(0.).t();

  // mark products
  auto sp = stoich_local.clamp_min(0.).t();

  // extend concentration
  auto conc2 = conc.unsqueeze(-1).unsqueeze(-1);

  // calculate jacobian
  out = -rh / (1. - rh) * sp / conc2 - sm / conc2;

  // add temperature derivative
  // evaluate svp function
  auto logsvp_ddT = LogSVPFunc::grad(temp);
  auto intEng = eval_intEng_R(temp, conc, options) * constants::Rgas;
  out -= (logsvp_ddT / (1. - rh)).unsqueeze(-1) * intEng.unsqueeze(-2) /
         cvol.unsqueeze(-1).unsqueeze(-1);

  // flag saturated reactions
  auto jsat = rh > 1.0 - 1.e-6;
  out.masked_fill_(jsat.unsqueeze(-1), 0.0);

  out *= rate.unsqueeze(-1);

  // add temperature derivative if provided
  if (logrc_ddT.has_value()) {
  }
}

}  // namespace kintera
