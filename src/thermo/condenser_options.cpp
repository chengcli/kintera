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

  for (auto const& rxn_node : root["reactions"]) {
    if (!rxn_node["type"] ||
        (rxn_node["type"].as<std::string>() != "nucleation")) {
      continue;
    }

    cond.react().push_back(Nucleation::from_yaml(rxn_node));
  }

  return cond;
}

}  // namespace kintera
