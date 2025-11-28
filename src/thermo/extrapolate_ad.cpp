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

  LogSVPFunc::init(options->nucleation());

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
  if (verbose) {
    std::cout << "Extrapolating adiabat with dlnp = " << dlnp << std::endl;
  }

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
    std::cout << "Composition:   X = { ";
    for (int i = 0; i < xfrac.size(-1); ++i) {
      std::cout << "[" << xfrac.select(-1, i).min().item<double>() << ", "
                << xfrac.select(-1, i).max().item<double>() << "], ";
    }
    std::cout << "}" << std::endl;
  }

  int iter = 0;
  pres *= exp(dlnp);
  while (iter++ < options->max_iter()) {
    auto gain = forward(temp, pres, xfrac);

    conc = compute("TPX->V", {temp, pres, xfrac});

    auto cp_mole = effective_cp(temp, pres, xfrac, gain, conc);

    entropy_vol = compute("TPV->S", {temp, pres, conc});
    auto entropy_mole = entropy_vol / conc.sum(-1);

    if (verbose) {
      std::cout << "Iter " << iter << std::endl;
      std::cout << "temp = [" << temp.min().item<double>() << ", "
                << temp.max().item<double>() << "] K" << std::endl;
      std::cout << "pres = [" << pres.min().item<double>() << ", "
                << pres.max().item<double>() << "] Pa" << std::endl;

      std::cout << "entropy_mole = [" << entropy_mole.min().item<double>()
                << ", " << entropy_mole.max().item<double>() << "] J/K/mol"
                << std::endl;

      std::cout << "Composition: X = { ";
      for (int i = 0; i < xfrac.size(-1); ++i) {
        std::cout << "[" << xfrac.select(-1, i).min().item<double>() << ", "
                  << xfrac.select(-1, i).max().item<double>() << "], ";
      }
      std::cout << "}" << std::endl;
    }

    if ((entropy_mole0 - entropy_mole).abs().max().item<double>() <
        10 * options->ftol()) {
      break;
    }

    temp *= 1. + (entropy_mole0 - entropy_mole) / cp_mole;
  }

  if (iter >= options->max_iter()) {
    TORCH_WARN("extrapolate_ad does not converge after ", options->max_iter(),
               " iterations.");
  }
}

void ThermoXImpl::extrapolate_ad(torch::Tensor temp, torch::Tensor pres,
                                 torch::Tensor xfrac, double grav, double dz,
                                 bool verbose) {
  if (verbose) {
    std::cout << "Extrapolating adiabat over dz = " << dz << " m"
              << " with gravity = " << grav << " m/s^2" << std::endl;
  }

  auto conc = compute("TPX->V", {temp, pres, xfrac});
  auto rho0 = compute("V->D", {conc});
  auto entropy_mole0 = compute("TPV->S", {temp, pres, conc}) / conc.sum(-1);

  if (verbose) {
    std::cout << "Initial State: T = [" << temp.min().item<double>() << ", "
              << temp.max().item<double>() << "] K" << std::endl;
    std::cout << "               P = [" << pres.min().item<double>() << ", "
              << pres.max().item<double>() << "] Pa" << std::endl;
    std::cout << "               S = [" << entropy_mole0.min().item<double>()
              << ", " << entropy_mole0.max().item<double>() << "] J/K/mol"
              << std::endl;
    std::cout << "Composition:   X = { ";
    for (int i = 0; i < xfrac.size(-1); ++i) {
      std::cout << "[" << xfrac.select(-1, i).min().item<double>() << ", "
                << xfrac.select(-1, i).max().item<double>() << "], ";
    }
    std::cout << "}" << std::endl;
  }

  auto mmw = (mu * xfrac).sum(-1);
  auto pres0 = pres.clone();
  pres.set_(pres0 * exp(-grav * mmw * dz / (constants::Rgas * temp)));

  int iter = 0;
  // std::vector<torch::Tensor> temp_list;
  // std::vector<torch::Tensor> pres_list;
  // std::vector<torch::Tensor> entropy_list;

  while (iter++ < options->max_iter()) {
    // temp_list.push_back(temp.clone());
    // pres_list.push_back(pres.clone());

    auto gain = forward(temp, pres, xfrac);
    conc = compute("TPX->V", {temp, pres, xfrac});
    auto cp_mole = effective_cp(temp, pres, xfrac, gain, conc);
    auto entropy_mole = compute("TPV->S", {temp, pres, conc}) / conc.sum(-1);
    // entropy_list.push_back(entropy_mole.clone());

    if (verbose) {
      std::cout << "Iter " << iter << std::endl;
      std::cout << "temp = [" << temp.min().item<double>() << ", "
                << temp.max().item<double>() << "] K" << std::endl;
      std::cout << "pres = [" << pres.min().item<double>() << ", "
                << pres.max().item<double>() << "] Pa" << std::endl;

      std::cout << "entropy_mole = [" << entropy_mole.min().item<double>()
                << ", " << entropy_mole.max().item<double>() << "] J/K/mol"
                << std::endl;

      std::cout << "Composition: X = { ";
      for (int i = 0; i < xfrac.size(-1); ++i) {
        std::cout << "[" << xfrac.select(-1, i).min().item<double>() << ", "
                  << xfrac.select(-1, i).max().item<double>() << "], ";
      }
      std::cout << "}" << std::endl;
    }

    if ((entropy_mole0 - entropy_mole).abs().max().item<double>() <
        10 * options->ftol()) {
      break;
    }

    /*if (iter % 2 == 0) {
      // weight by distance from previous iterate
      auto w1 = entropy_list[0] - entropy_mole0;
      auto w2 = entropy_list[1] - entropy_mole0;
      w1 = w1 * w1 / (w1 * w1 + w2 * w2 + 1e-10);
      w2 = 1. - w1;

      temp.set_(w2 * temp_list[0] + w1 * temp_list[1]);
      pres.set_(w2 * pres_list[0] + w1 * pres_list[1]);

      if (verbose) {
        std::cout << "  Weighted averaging over last " << temp_list.size()
                  << " iterates." << std::endl;
        std::cout << "    temp = [" << temp.min().item<double>() << ", "
                  << temp.max().item<double>() << "] K" << std::endl;
        std::cout << "    pres = [" << pres.min().item<double>() << ", "
                  << pres.max().item<double>() << "] Pa" << std::endl;
      }

      temp_list.clear();
      pres_list.clear();
      entropy_list.clear();
    } else {*/
    int sub_iter = 3;
    auto pres1 = pres.clone();
    auto temp1 = temp.clone();
    // total gas mole fractions
    auto xg = xfrac.narrow(-1, 0, options->vapor_ids().size()).sum(-1);
    while (sub_iter-- > 0) {
      auto rho = compute("V->D", {conc});
      pres.set_(pres0 - 0.5 * (rho + rho0) * grav * dz);
      auto dlnp = pres.log() - pres1.log();
      temp.set_(temp1 * (1. + (entropy_mole0 - entropy_mole +
                               xg * constants::Rgas * dlnp) /
                                  cp_mole));
      conc = compute("TPX->V", {temp, pres, xfrac});
      if (verbose) {
        std::cout << "  Sub-iter: pres = [" << pres.min().item<double>() << ", "
                  << pres.max().item<double>() << "] Pa" << std::endl;
        std::cout << "            temp = [" << temp.min().item<double>() << ", "
                  << temp.max().item<double>() << "] K" << std::endl;
        std::cout << "            dlnp = [" << dlnp.min().item<double>() << ", "
                  << dlnp.max().item<double>() << "]" << std::endl;
      }
    }
    //}
  }

  if (iter >= options->max_iter()) {
    TORCH_WARN("extrapolate_ad does not converge after ", options->max_iter(),
               " iterations.");
  }
}

}  // namespace kintera
