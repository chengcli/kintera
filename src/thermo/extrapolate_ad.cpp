// C/C++
#include <cmath>

// kintera
#include <kintera/constants.h>

#include "eval_uhs.hpp"
#include "extrapolate_ad.hpp"
#include "log_svp.hpp"

namespace kintera {

torch::Tensor effective_cp_mole(torch::Tensor temp, torch::Tensor pres,
                                torch::Tensor xfrac, torch::Tensor gain,
                                ThermoX &thermo,
                                torch::optional<torch::Tensor> conc) {
  LogSVPFunc::init(thermo->options.nucleation());

  auto logsvp_ddT = LogSVPFunc::grad(temp);
  auto rate_ddT = std::get<0>(torch::linalg_lstsq(gain, logsvp_ddT));

  if (!conc.has_value()) {
    conc = thermo->compute("TPX->V", {temp, pres, xfrac});
  }

  auto enthalpy =
      eval_enthalpy_R(temp, conc.value(), thermo->options) * constants::Rgas;
  auto cp = eval_cp_R(temp, conc.value(), thermo->options) * constants::Rgas;

  auto cp_normal = (cp * xfrac).sum(-1);
  auto cp_latent = (enthalpy.matmul(thermo->stoich) * rate_ddT).sum(-1);

  return cp_normal + cp_latent;
}

void extrapolate_ad_(torch::Tensor temp, torch::Tensor pres,
                     torch::Tensor xfrac, ThermoX &thermo, double dlnp) {
  auto conc = thermo->compute("TPX->V", {temp, pres, xfrac});
  auto entropy_vol = thermo->compute("TPV->S", {temp, pres, conc});
  auto entropy_mole0 = entropy_vol / conc.sum(-1);

  int iter = 0;
  pres *= exp(dlnp);
  while (iter++ < thermo->options.max_iter()) {
    auto gain = thermo->forward(temp, pres, xfrac);
    conc = thermo->compute("TPX->V", {temp, pres, xfrac});

    auto cp_mole = effective_cp_mole(temp, pres, xfrac, gain, thermo, conc);

    entropy_vol = thermo->compute("TPV->S", {temp, pres, conc});
    auto entropy_mole = entropy_vol / conc.sum(-1);

    temp *= 1. + (entropy_mole0 - entropy_mole) / cp_mole;

    if ((entropy_mole0 - entropy_mole).abs().max().item<double>() <
        10 * thermo->options.ftol()) {
      break;
    }
  }

  if (iter >= thermo->options.max_iter()) {
    TORCH_WARN("extrapolate_ad does not converge after ",
               thermo->options.max_iter(), " iterations.");
  }
}

}  // namespace kintera
