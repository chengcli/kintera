// torch
#include <torch/torch.h>

// kintera
#include "condenser.hpp"

namespace kintera {

CondenserOptions CondenserOptions::from_yaml(std::string const& filename) {
  CondenserOptions cond;

  YAML::Node root = YAML::LoadFile(filename);

  TORCH_CHECK(root["reactions"],
              "'reactions' is not defined in the configuration file");

  for (auto const& node : root["reactions"]) {
    if (!node["type"] || (node["type"].as<std::string>() != "nucleation")) {
      continue;
    }
    // only process nucleation reactions

    TORCH_CHECK(node["rate-constant"],
                "'rate-constant' is not defined in the reaction");
    TORCH_CHECK(node["equation"], "'equation' is not defined in the reaction");
    // reaction equation
    reaction(Reaction(node["equation"].as<std::string>()));

    // rate constants
    auto rate_constant = node["rate-constant"];
    if (rate_constant["minT"]) {
      minT(rate_constant["minT"].as<double>());
    }

    if (rate_constant["maxT"]) {
      maxT(rate_constant["maxT"].as<double>());
    }
  }

  return cond;
}

}  // namespace kintera
