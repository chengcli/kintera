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

Nucleation::Nucleation(const std::string& equation, const std::string& name,
                       const std::map<std::string, double>& params) {
  if (params.find("minT") != params.end()) {
    min_tem(params.at("minT"));
  }

  if (params.find("maxT") != params.end()) {
    max_tem(params.at("maxT"));
  }

  reaction() = Reaction(equation);

  if (name == "ideal") {
    auto t3 = params.at("T3");
    auto p3 = params.at("P3");
    auto beta = params.at("beta");
    auto delta = params.at("delta");
    func() = [=](torch::Tensor T) -> torch::Tensor {
      return p3 * exp((1. - t3 / T) * beta - delta * log(T / t3));
    };
    logf_ddT() = [=](torch::Tensor T) -> torch::Tensor {
      return beta * t3 / (T * T) - delta / T;
    };
    if (delta > 0.) {
      max_tem(std::min(max_tem(), beta * t3 / delta));
    }
  } else {
    func() = find_svp(reaction().reactants(), name);
    logf_ddT() = find_logsvp_ddT(reaction().reactants(), name);
  }
}

torch::Tensor Nucleation::eval_func(torch::Tensor tem) const {
  int order = reaction().reactants().size();
  auto out = func()(tem);
  out.masked_fill_(tem < min_tem(), -1.);
  out.masked_fill_(tem > max_tem(), -1.);
  return out;
}

torch::Tensor Nucleation::eval_logf_ddT(torch::Tensor tem) const {
  int order = reaction().reactants().size();
  auto out = logf_ddT()(tem) - order / tem;
  out.masked_fill_(tem < min_tem(), 0.);
  out.masked_fill_(tem > max_tem(), 0.);
  return out;
}

}  // namespace kintera
