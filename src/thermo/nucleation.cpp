// C/C++
#include <functional>

// kintera
#include <kintera/vapors/h2o.hpp>
#include <kintera/vapors/nh3.hpp>
#include <kintera/vapors/nh3_h2s.hpp>
#include <kintera/vapors/svp_ideal.hpp>

#include "nucleation.hpp"

namespace kintera {

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

  TORCH_CHECK(get_user_func1().finds(formula) != get_user_func1().end(),
              "Formula '", formula, "' is not defined in the user functions");
  nuc.func() = get_user_func1()[formula];

  TORCH_CHECK(
      get_user_func1().finds(formula + "_ddT") != get_user_func1().end(),
      "Formula '", formula, "' is not defined in the user functions");
  nuc.func_ddT() = find_logsvp_ddT(nuc.reaction().reactants(), formula);

  return nuc;
}

torch::Tensor Nucleation::eval_func(torch::Tensor tem) const {
  int order = reaction().reactants().size();
  auto out = func()(tem);
  out.masked_fill_(tem < minT(), -1.);
  out.masked_fill_(tem > maxT(), -1.);
  return out;
}

torch::Tensor Nucleation::eval_func_ddT(torch::Tensor tem) const {
  int order = reaction().reactants().size();
  auto out = func_ddT()(tem) - order / tem;
  out.masked_fill_(tem < minT(), 0.);
  out.masked_fill_(tem > maxT(), 0.);
  return out;
}

}  // namespace kintera
