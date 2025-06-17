// yaml
#include <yaml-cpp/yaml.h>

// kintera
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

  TORCH_CHECK(get_user_func1().find(formula) != get_user_func1().end(),
              "Formula '", formula, "' is not defined in the user functions");
  nuc.func() = get_user_func1()[formula];

  TORCH_CHECK(get_user_func1().find(formula + "_ddT") != get_user_func1().end(),
              "Formula '", formula, "' is not defined in the user functions");
  nuc.func_ddT() = get_user_func1()[formula + "_ddT"];

  return nuc;
}

}  // namespace kintera
