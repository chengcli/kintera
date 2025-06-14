// C/C++
#include <cmath>

// kintera
#include "extrapolate_ad.hpp"

namespace kintera {

torch::Tensor effective_cp_mole(torch::Tensor temp, torch::Tensor pres,
                                torch::Tensor xfrac, ThermoX &thermo,
                                double dT) {
  thermo->forward(temp, pres, xfrac);
  auto conc = thermo->compute("TPX->V", {temp, pres, xfrac});
  auto enthalpy_vol = thermo->compute("TV->H", {temp, conc});
  auto enthalpy_mole1 = enthalpy_vol / conc.sum(-1);

  temp += dT;
  thermo->forward(temp, pres, xfrac);
  conc = thermo->compute("TPX->V", {temp, pres, xfrac});
  enthalpy_vol = thermo->compute("TV->H", {temp, conc});
  auto enthalpy_mole2 = enthalpy_vol / conc.sum(-1);

  return (enthalpy_mole2 - enthalpy_mole1) / dT;
}

void extrapolate_ad_(torch::Tensor temp, torch::Tensor pres,
                     torch::Tensor xfrac, ThermoX &thermo, double dlnp,
                     double ftol) {
  auto temp0 = temp.clone();
  thermo->forward(temp, pres, xfrac);

  auto conc = thermo->compute("TPX->V", {temp, pres, xfrac});
  auto entropy_vol = thermo->compute("TPV->S", {temp, pres, conc});
  auto entropy_mole0 = entropy_vol / conc.sum(-1);

  int iter = 0;
  pres *= exp(dlnp);
  while (iter++ < thermo->options.max_iter()) {
    thermo->forward(temp, pres, xfrac);
    conc = thermo->compute("TPX->V", {temp, pres, xfrac});

    auto cp_mole = effective_cp_mole(temp, pres, xfrac, thermo, ftol / 10.);
    auto cp_mole1 = thermo->compute("TV->cp", {temp, conc}) / conc.sum(-1);
    if ((cp_mole - cp_mole1).abs().max().item<double>() < ftol) {
      cp_mole = cp_mole1;
    }

    entropy_vol = thermo->compute("TPV->S", {temp, pres, conc});
    auto entropy_mole = entropy_vol / conc.sum(-1);

    auto temp_pre = temp.clone();
    temp *= 1. + (entropy_mole0 - entropy_mole) / cp_mole;

    if ((temp - temp_pre).abs().max().item<double>() < ftol) {
      break;
    }
  }

  if (iter >= thermo->options.max_iter()) {
    TORCH_WARN("max iteration reached");
  }
}

}  // namespace kintera
