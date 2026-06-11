// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include "nucleation.hpp"

namespace kintera {

extern std::vector<std::string> species_names;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set,
                        NucleationOptions op) {
  for (auto const& react : op->reactions()) {
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

NucleationOptions NucleationOptionsImpl::from_yaml(
    const YAML::Node& root,
    std::shared_ptr<NucleationOptionsImpl> derived_type_ptr) {
  auto options =
      derived_type_ptr ? derived_type_ptr : NucleationOptionsImpl::create();

  for (auto const& rxn_node : root) {
    TORCH_CHECK(rxn_node["type"], "Reaction type not specified");

    if (rxn_node["type"].as<std::string>() != options->name()) continue;

    TORCH_CHECK(rxn_node["equation"],
                "'equation' is not defined in the reaction");

    auto equation = rxn_node["equation"].as<std::string>();
    options->reactions().push_back(Reaction(equation));

    TORCH_CHECK(rxn_node["rate-constant"],
                "'rate-constant' is not defined in the reaction");

    // rate constants
    auto node = rxn_node["rate-constant"];
    options->minT().push_back(node["minT"].as<double>(0.));
    options->maxT().push_back(node["maxT"].as<double>(1.e4));

    TORCH_CHECK(node["formula"],
                "'formula' is not defined in the rate-constant");

    auto formula = node["formula"].as<std::string>();
    options->logsvp().push_back(formula);

    // Inline-parametrized generic SVP forms. When the formula is a named
    // function from the func table, the parameter vector is left empty and the
    // function pointer is used at evaluation time instead.
    std::vector<double> params;
    if (formula == "ideal") {
      TORCH_CHECK(node["T3"], "'T3' required for 'ideal' svp formula");
      TORCH_CHECK(node["P3"], "'P3' required for 'ideal' svp formula");
      TORCH_CHECK(node["beta"], "'beta' required for 'ideal' svp formula");
      TORCH_CHECK(node["gamma"], "'gamma' required for 'ideal' svp formula");
      double T3 = node["T3"].as<double>();
      double P3 = node["P3"].as<double>();
      double beta = node["beta"].as<double>();
      double gamma = node["gamma"].as<double>();
      // Optional solid branch below T3; defaults to the liquid coefficients,
      // i.e. a single-branch curve when betas/gammas are omitted.
      double betas = node["betas"].as<double>(beta);
      double gammas = node["gammas"].as<double>(gamma);
      params = {T3, P3, beta, gamma, betas, gammas};
    } else if (formula == "antoine") {
      TORCH_CHECK(node["A"], "'A' required for 'antoine' svp formula");
      TORCH_CHECK(node["B"], "'B' required for 'antoine' svp formula");
      TORCH_CHECK(node["C"], "'C' required for 'antoine' svp formula");
      double A = node["A"].as<double>();
      double B = node["B"].as<double>();
      double C = node["C"].as<double>();
      params = {A, B, C};
    }
    options->svp_params().push_back(params);
  }

  return options;
}

}  // namespace kintera
