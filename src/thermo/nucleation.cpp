// C/C++
#include <functional>

// kintera
#include <kintera/vapors/ammonia_vapors.hpp>
#include <kintera/vapors/ammonium_hydrosulfide_vapors.hpp>
#include <kintera/vapors/water_vapors.hpp>

#include "nucleation.hpp"

namespace kintera {

std::string concatenate(const Composition& comp, char sep) {
  if (comp.empty()) return "";
  if (comp.size() == 1) return comp.begin()->first;

  std::string result = comp.begin()->first;

  for (auto it = std::next(comp.begin()); it != comp.end(); ++it) {
    result += sep + it->first;
  }

  return result;
}

std::function<torch::Tensor(torch::Tensor)> find_svp(
    const Composition& reactants, const std::string& svp_name) {
  std::string name = concatenate(reactants, '-') + '-' + svp_name;

  if (name == "NH3-H2S-lewis" || name == "H2S-NH3-lewis") {
    return svp_nh3_h2s_Lewis;
  } else if (name == "H2O-antoine") {
    return svp_h2o_Antoine;
  } else if (name == "NH3-antoine") {
    return svp_nh3_Antoine;
  }

  throw std::runtime_error("No SVP function found for reaction '" + name +
                           "'.");
}

std::function<torch::Tensor(torch::Tensor)> find_logsvp_ddT(
    const Composition& reactants, const std::string& svp_name) {
  std::string name = concatenate(reactants, '-') + '-' + svp_name;

  if (name == "NH3-H2S-lewis" || name == "H2S-NH3-lewis") {
    return logsvp_ddT_nh3_h2s_Lewis;
  } else if (name == "H2O-antoine") {
    return nullptr;  // logsvp_ddT_h2o_Antoine;
  } else if (name == "NH3-antoine") {
    return nullptr;  // logsvp_ddT_nh3_Antoine;
  }

  throw std::runtime_error("No SVP_DDT function found for reaction '" + name +
                           "'.");
}

Nucleation Nucleation::from_yaml(const YAML::Node& node) {
  Nucleation nuc;

  TORCH_CHECK(node["type"].as<std::string>() == "nucleation",
              "Reaction type is not nucleation");

  TORCH_CHECK(node["rate-constant"],
              "'rate-constant' is not defined in the reaction");

  TORCH_CHECK(node["equation"], "'equation' is not defined in the reaction");

  // reaction equation
  nuc.reaction() = Reaction(node["equation"].as<std::string>());

  // rate constants
  auto rate_constant = node["rate-constant"];
  if (rate_constant["minT"]) {
    nuc.minT(rate_constant["minT"].as<double>());
  }

  if (rate_constant["maxT"]) {
    nuc.maxT(rate_constant["maxT"].as<double>());
  }

  TORCH_CHECK(rate_constant["formula"],
              "'formula' is not defined in the rate-constant");

  auto formula = rate_constant["formula"].as<std::string>();
  if (formula == "ideal") {
    auto t3 = rate_constant["T3"].as<double>();
    auto p3 = rate_constant["P3"].as<double>();
    auto beta = rate_constant["beta"].as<double>();
    auto delta = rate_constant["delta"].as<double>();
    nuc.func() = [=](torch::Tensor T) -> torch::Tensor {
      return p3 * exp((1. - t3 / T) * beta - delta * log(T / t3));
    };
    nuc.logf_ddT() = [=](torch::Tensor T) -> torch::Tensor {
      return beta * t3 / (T * T) - delta / T;
    };
    if (delta > 0.) {
      nuc.maxT(std::min(nuc.maxT(), beta * t3 / delta));
    }
  } else {
    nuc.func() = find_svp(nuc.reaction().reactants(), formula);
    nuc.logf_ddT() = find_logsvp_ddT(nuc.reaction().reactants(), formula);
  }

  return nuc;
}

torch::Tensor Nucleation::eval_func(torch::Tensor tem) const {
  int order = reaction().reactants().size();
  auto out = func()(tem);
  out.masked_fill_(tem < minT(), -1.);
  out.masked_fill_(tem > maxT(), -1.);
  return out;
}

torch::Tensor Nucleation::eval_logf_ddT(torch::Tensor tem) const {
  int order = reaction().reactants().size();
  auto out = logf_ddT()(tem) - order / tem;
  out.masked_fill_(tem < minT(), 0.);
  out.masked_fill_(tem > maxT(), 0.);
  return out;
}

}  // namespace kintera
