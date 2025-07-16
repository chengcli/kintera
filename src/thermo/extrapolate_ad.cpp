// C/C++
#include <cmath>

// kintera
#include <kintera/constants.h>

#include "eval_uhs.hpp"
#include "log_svp.hpp"
#include "thermo.hpp"

namespace kintera {

torch::Tensor ThermoXImpl::effective_cp(torch::Tensor temp, torch::Tensor pres,
                                        torch::Tensor xfrac, torch::Tensor gain,
                                        torch::optional<torch::Tensor> conc) {
  if (!conc.has_value()) {
    conc = compute("TPX->V", {temp, pres, xfrac});
  }

  if (!gain.defined()) {  // no-op
    auto cp = eval_cp_R(temp, conc.value(), options) * constants::Rgas;
    return (cp * xfrac).sum(-1);
  }

  LogSVPFunc::init(options.nucleation());

  auto logsvp_ddT = LogSVPFunc::grad(temp);
  auto rate_ddT = std::get<0>(torch::linalg_lstsq(gain, logsvp_ddT));

  auto enthalpy =
      eval_enthalpy_R(temp, conc.value(), options) * constants::Rgas;
  auto cp = eval_cp_R(temp, conc.value(), options) * constants::Rgas;

  auto cp_normal = (cp * xfrac).sum(-1);
  auto cp_latent = (enthalpy.matmul(stoich) * rate_ddT).sum(-1);

  return cp_normal + cp_latent;
}

void ThermoXImpl::extrapolate_ad(torch::Tensor temp, torch::Tensor pres,
                                 torch::Tensor xfrac, double dlnp) {
  auto conc = compute("TPX->V", {temp, pres, xfrac});
  auto entropy_vol = compute("TPV->S", {temp, pres, conc});
  auto entropy_mole0 = entropy_vol / conc.sum(-1);

  int iter = 0;
  pres *= exp(dlnp);
  while (iter++ < options.max_iter()) {
    auto gain = forward(temp, pres, xfrac);

    conc = compute("TPX->V", {temp, pres, xfrac});

    auto cp_mole = effective_cp(temp, pres, xfrac, gain, conc);

    entropy_vol = compute("TPV->S", {temp, pres, conc});
    auto entropy_mole = entropy_vol / conc.sum(-1);

    temp *= 1. + (entropy_mole0 - entropy_mole) / cp_mole;

    if ((entropy_mole0 - entropy_mole).abs().max().item<double>() <
        10 * options.ftol()) {
      break;
    }
  }

  if (iter >= options.max_iter()) {
    TORCH_WARN("extrapolate_ad does not converge after ", options.max_iter(),
               " iterations.");
  }
}

void ThermoXImpl::extrapolate_ad(torch::Tensor temp, torch::Tensor pres,
                                 torch::Tensor xfrac, double grav, double dz) {
  auto conc = compute("TPX->V", {temp, pres, xfrac});
  auto entropy_vol = compute("TPV->S", {temp, pres, conc});
  auto entropy_mole0 = entropy_vol / conc.sum(-1);

  auto gain = forward(temp, pres, xfrac);
  auto cp_mole = effective_cp(temp, pres, xfrac, gain, conc);
  auto cp_mole0 = cp_mole.clone();
  auto mmw = (mu * xfrac).sum(-1);

  int iter = 0;
  torch::Tensor temp1;
  while (iter++ < options.max_iter()) {
    temp1 = temp - 2. * grav * mmw / (cp_mole + cp_mole0) * dz;

    entropy_vol = compute("TPV->S", {temp1, pres, conc});
    auto entropy_mole = entropy_vol / conc.sum(-1);

    pres *= 1. - (entropy_mole0 - entropy_mole) / constants::Rgas;

    if ((entropy_mole0 - entropy_mole).abs().max().item<double>() <
        10 * options.ftol()) {
      break;
    }

    auto gain = forward(temp1, pres, xfrac);
    conc = compute("TPX->V", {temp1, pres, xfrac});
    auto cp_mole = effective_cp(temp1, pres, xfrac, gain, conc);
  }

  temp -= temp - temp1;

  if (iter >= options.max_iter()) {
    TORCH_WARN("extrapolate_ad does not converge after ", options.max_iter(),
               " iterations.");
  }
}

}  // namespace kintera
