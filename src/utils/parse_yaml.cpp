// C/C++
#include <cctype>
#include <cstdlib>
#include <stdexcept>

// yaml-cpp
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/kinetics/arrhenius.hpp>
#include <kintera/reaction.hpp>

namespace kintera {

std::map<Reaction, torch::nn::AnyModule> parse_reactions_yaml(
    const std::string& filename) {
  std::map<Reaction, torch::nn::AnyModule> reaction_rates;

  YAML::Node root = YAML::LoadFile(filename);
  printf("Loading complete\n");
  for (const auto& rxn_node : root) {
    std::string equation = rxn_node["equation"].as<std::string>();

    Reaction reaction(equation);

    if (rxn_node["orders"]) {
      const auto& orders = rxn_node["orders"];
      for (const auto& order : orders) {
        std::string species = order.first.as<std::string>();
        reaction.orders()[species] = order.second.as<double>();
      }
    } else {
      for (const auto& species : reaction.reactants()) {
        reaction.orders()[species.first] = 1.0;
      }
      if (reaction.reversible()) {
        for (const auto& species : reaction.products()) {
          reaction.orders()[species.first] = 1.0;
        }
      }
    }

    // if (rxn_node["efficiencies"]) {
    //     const auto& effs = rxn_node["efficiencies"];
    //     for (const auto& eff : effs) {
    //         std::string species = eff.first.as<std::string>();
    //         double value = eff.second.as<double>();
    //     }
    // }

    std::string type = "arrhenius";  // default type
    if (rxn_node["type"]) {
      type = rxn_node["type"].as<std::string>();
    }

    // TODO: Implement the support of other reaction types
    if (type == "arrhenius") {
      auto op = ArrheniusOptionsImpl::from_yaml(rxn_node["rate-constant"]);
      reaction_rates[reaction] = torch::nn::AnyModule(Arrhenius(op));
    } else if (type == "three-body") {
      printf("Three-body reaction not implemented\n");
      continue;
    } else if (type == "falloff") {
      printf("Falloff reaction not implemented\n");
      continue;
    } else {
      printf("Unknown reaction type: %s\n", type.c_str());
      continue;
    }
  }

  return reaction_rates;
}

std::vector<Reaction> parse_reactions_yaml(
    const std::string& filename, std::vector<std::string> const& types) {
  std::vector<Reaction> reactions;

  YAML::Node root = YAML::LoadFile(filename);
  printf("Loading complete\n");

  for (auto const& type : types) {
    for (auto const& rxn_node : root) {
      if (!rxn_node["type"]) {
        TORCH_CHECK(false, "Reaction type not specified");
      }

      if (rxn_node["type"].as<std::string>() != type) {
        continue;
      }

      if (!rxn_node["equation"]) {
        TORCH_CHECK(false, "Reaction equation not specified");
      }

      std::string equation = rxn_node["equation"].as<std::string>();
      reactions.emplace_back(equation);
    }
  }

  return reactions;
}

}  // namespace kintera
