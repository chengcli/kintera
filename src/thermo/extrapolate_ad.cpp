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

  torch::Tensor rate_ddT;

  if (gain.device().is_cpu()) {
    rate_ddT = std::get<0>(torch::linalg_lstsq(gain, logsvp_ddT));
  } else {
    auto pinv = torch::linalg_pinv(gain, /*atol=*/1e-6);
    rate_ddT = pinv.matmul(logsvp_ddT.unsqueeze(-1)).squeeze(-1);
  }

  auto enthalpy =
      eval_enthalpy_R(temp, conc.value(), options) * constants::Rgas;
  auto cp = eval_cp_R(temp, conc.value(), options) * constants::Rgas;

  auto cp_normal = (cp * xfrac).sum(-1);
  auto cp_latent = (enthalpy.matmul(stoich) * rate_ddT).sum(-1);

  return cp_normal + cp_latent;
}

void ThermoXImpl::extrapolate_ad(torch::Tensor temp, torch::Tensor pres,
                                 torch::Tensor xfrac, double dlnp,
                                 bool verbose) {
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
                                 torch::Tensor xfrac, double grav, double dz,
                                 bool verbose) {
  auto conc = compute("TPX->V", {temp, pres, xfrac});
  auto entropy_vol = compute("TPV->S", {temp, pres, conc});
  auto entropy_mole0 = entropy_vol / conc.sum(-1);

  if (verbose) {
    std::cout << "Initial State: T = [" << temp.min().item<double>() << ", "
              << temp.max().item<double>() << "] K" << std::endl;
    std::cout << "               P = [" << pres.min().item<double>() << ", "
              << pres.max().item<double>() << "] Pa" << std::endl;
    std::cout << "               S = [" << entropy_mole0.min().item<double>()
              << ", " << entropy_mole0.max().item<double>() << "] J/K/mol"
              << std::endl;
  }

  auto gain = forward(temp, pres, xfrac);
  auto cp_mole = effective_cp(temp, pres, xfrac, gain, conc);
  auto cp_mole0 = cp_mole.clone();
  auto mmw = (mu * xfrac).sum(-1);

  int iter = 0;
  std::vector<torch::Tensor> temp_list;
  std::vector<torch::Tensor> pres_list;
  std::vector<torch::Tensor> entropy_list;

  torch::Tensor temp1 = temp.clone();
  torch::Tensor pres1 =
      pres * torch::exp(-grav * mmw * dz / (constants::Rgas * temp));

  while (iter++ < options.max_iter()) {
    temp_list.push_back(temp1.clone());
    pres_list.push_back(pres1.clone());

    if (verbose) {
      std::cout << "Iter " << iter << std::endl;
      std::cout << "temp1 = [" << temp1.min().item<double>() << ", "
                << temp1.max().item<double>() << "] K" << std::endl;
      std::cout << "pres1 = [" << pres1.min().item<double>() << ", "
                << pres1.max().item<double>() << "] Pa" << std::endl;
    }

    entropy_vol = compute("TPV->S", {temp1, pres1, conc});
    auto entropy_mole = entropy_vol / conc.sum(-1);
    entropy_list.push_back(entropy_mole.clone());

    if (verbose) {
      std::cout << "entropy_mole = [" << entropy_mole.min().item<double>()
                << ", " << entropy_mole.max().item<double>() << "] J/K/mol"
                << std::endl;
    }

    if ((entropy_mole0 - entropy_mole).abs().max().item<double>() <
        10 * options.ftol()) {
      break;
    }

    temp1 *= 1. + (entropy_mole0 - entropy_mole) / cp_mole;
    pres1 = pres * torch::exp(-2. * grav * mmw * dz /
                              (constants::Rgas * (temp + temp1)));

    // make average of all iterates to improve stability
    if (iter % 2 == 0) {
      // weight by distance from current iterate
      auto w1 = entropy_list[0] - entropy_mole0;
      auto w2 = entropy_list[1] - entropy_mole0;
      w1 = w1 * w1 / (w1 * w1 + w2 * w2 + 1e-10);
      w2 = 1. - w1;

      temp1 = w2 * temp_list[0] + w1 * temp_list[1];
      pres1 = w2 * pres_list[0] + w1 * pres_list[1];

      if (verbose) {
        std::cout << "Averaging over last " << temp_list.size() << " iterates."
                  << std::endl;
        std::cout << "temp1 = [" << temp1.min().item<double>() << ", "
                  << temp1.max().item<double>() << "] K" << std::endl;
        std::cout << "pres1 = [" << pres1.min().item<double>() << ", "
                  << pres1.max().item<double>() << "] Pa" << std::endl;
      }

      temp_list.clear();
      pres_list.clear();
      entropy_list.clear();
    }

    auto gain = forward(temp1, pres1, xfrac);
    conc = compute("TPX->V", {temp1, pres1, xfrac});
    auto cp_mole = effective_cp(temp1, pres1, xfrac, gain, conc);
  }

  temp.copy_(temp1);
  pres.copy_(pres1);

  if (iter >= options.max_iter()) {
    TORCH_WARN("extrapolate_ad does not converge after ", options.max_iter(),
               " iterations.");
  }
}

}  // namespace kintera
