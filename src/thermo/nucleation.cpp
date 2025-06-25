// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include "nucleation.hpp"

namespace kintera {

extern std::vector<std::string> species_names;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set,
                        NucleationOptions op) {
  for (auto const& react : op.reactions()) {
    // go through reactants
    for (auto& [name, _] : react.reactants()) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name,
                  " not found in species list");
      vapor_set.insert(name);
    }

    // go through products
    for (auto& [name, _] : react.products()) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name,
                  " not found in species list");
      cloud_set.insert(name);
    }
  }
}

NucleationOptions NucleationOptions::from_yaml(const YAML::Node& root) {
  NucleationOptions options;

  for (auto const& rxn_node : root) {
    TORCH_CHECK(rxn_node["type"], "Reaction type not specified");

    if (rxn_node["type"].as<std::string>() != "nucleation") {
      continue;
    }

    TORCH_CHECK(rxn_node["equation"],
                "'equation' is not defined in the reaction");

    std::string equation = rxn_node["equation"].as<std::string>();
    options.reactions().push_back(Reaction(equation));

    TORCH_CHECK(rxn_node["rate-constant"],
                "'rate-constant' is not defined in the reaction");

    // rate constants
    auto node = rxn_node["rate-constant"];
    options.minT().push_back(node["minT"].as<double>(0.));
    options.maxT().push_back(node["maxT"].as<double>(1.e4));

    TORCH_CHECK(node["formula"],
                "'formula' is not defined in the rate-constant");

    auto formula = node["formula"].as<std::string>();

    TORCH_CHECK(get_user_func1().find(formula) != get_user_func1().end(),
                "Formula '", formula, "' is not defined in the user functions");
    options.logsvp().push_back(get_user_func1()[formula]);

    TORCH_CHECK(
        get_user_func1().find(formula + "_ddT") != get_user_func1().end(),
        "Formula '", formula, "' is not defined in the user functions");
    options.logsvp_ddT().push_back(get_user_func1()[formula + "_ddT"]);
  }

  return options;
}

}  // namespace kintera
