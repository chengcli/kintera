// kintera
#include <kintera/thermo/eval_uhs.hpp>
#include <kintera/thermo/log_svp.hpp>
#include <kintera/thermo/relative_humidity.hpp>

#include "jacobian.hpp"

namespace kintera {

torch::Tensor jacobian_evaporation(torch::Tensor rate, torch::Tensor temp,
                                   torch::Tensor pres, torch::Tensor stoich,
                                   torch::Tensor conc, ThermoOptions op,
                                   double ftol) {
  // evaluate svp function
  LogSVPFunc::init(op.react());

  // evaluate relative humidity
  auto rh = relative_humidity(temp, pres, conc, -stoich, op);

  // mark reactants
  auto sm = stoich.clamp_max(0.).t();

  // mark products
  auto sp = stoich.clamp_min(0.).t();

  // extend concentration
  auto conc2 = conc.unsqueeze(-1).unsqueeze(-1);

  // calculate jacobian
  auto jacobian = -rh / (1. - rh) * sp / conc2 - sm / conc2;

  // add temperature derivative
  auto logsvp_ddT = LogSVPFunc::grad(temp);
  auto intEng_R = eval_intEng_R(temp, conc, op);
  auto cv_R = eval_cv_R(temp, conc, op);
  auto cv_vol = (cv_R * conc).sum(-1, /*keepdim=*/true);
  jacobian -= (logsvp_ddT / (1. - rh)).unsqueeze(-1) * intEng_R.unsqueeze(-2) /
              cv_vol.unsqueeze(-1);

  // flag saturated reactions
  auto jsat = rh > 1.0 - ftol;
  jacobian.masked_fill_(jsat.unsqueeze(-1), 0.0);

  return rate.unsqueeze(-1) * jacobian;
}

}  // namespace kintera
