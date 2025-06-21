// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include "evaporation.hpp"

namespace kintera {

extern std::vector<std::string> species_names;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set,
                        EvaporationOptions op) {
  for (auto& react : op.reactions()) {
    // go through reactants
    for (auto& [name, _] : react.reactants()) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name,
                  " not found in species list");
      cloud_set.insert(name);
    }

    // go through products
    for (auto& [name, _] : react.products()) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name,
                  " not found in species list");
      vapor_set.insert(name);
    }
  }
}

EvaporationOptions EvaporationOptions::from_yaml(const YAML::Node& root) {
  EvaporationOptions options;

  for (const auto& rxn_node : root) {
    TORCH_CHECK(rxn_node["type"], "Reaction type not specified");

    if (rxn_node["type"].as<std::string>() != "evaporation") {
      continue;
    }

    TORCH_CHECK(rxn_node["equation"],
                "'equation' is not defined in the reaction");

    std::string equation = rxn_node["equation"].as<std::string>();
    options.reactions().push_back(Reaction(equation));

    TORCH_CHECK(rxn_node["rate-constant"],
                "'rate-constant' is not defined in the reaction");

    auto node = rxn_node["rate-constant"];
    if (node["diff_c"]) {
      options.diff_c().push_back(node["diff_c"].as<double>());
    } else {
      options.diff_c().push_back(0.2);
    }

    if (node["diff_T"]) {
      options.diff_T().push_back(node["diff_T"].as<double>());
    } else {
      options.diff_T().push_back(1.75);
    }

    if (node["diff_P"]) {
      options.diff_P().push_back(node["diff_P"].as<double>());
    } else {
      options.diff_P().push_back(-1.);
    }

    if (node["vm"]) {
      options.vm().push_back(node["vm"].as<double>());
    } else {
      options.vm().push_back(18.);
    }

    if (node["diameter"]) {
      options.diameter().push_back(node["radius"].as<double>());
    } else {
      options.diameter().push_back(1.);
    }
  }

  return options;
}

EvaporationImpl::EvaporationImpl(EvaporationOptions const& options_)
    : options(std::move(options_)) {
  reset();
}

void EvaporationImpl::reset() {
  // log(cm^2/s) -> log(m^2/s)
  log_diff_c = register_buffer(
      "log_diff_c",
      torch::tensor(options.diff_c(), torch::kFloat64).log() + log(1.e-4));

  diff_T = register_buffer("diff_T",
                           torch::tensor(options.diff_T(), torch::kFloat64));
  diff_P = register_buffer("diff_P",
                           torch::tensor(options.diff_P(), torch::kFloat64));

  // log(cm^3/mol) -> log(m^3/mol)
  log_vm = register_buffer(
      "log_vm",
      torch::tensor(options.vm(), torch::kFloat64).log() + log(1.e-6));

  // log(cm) -> log(m)
  log_diameter = register_buffer(
      "log_diameter",
      torch::tensor(options.diameter(), torch::kFloat64).log() + log(1.e-2));
}

void EvaporationImpl::pretty_print(std::ostream& os) const {
  os << "Evaporation Rate: " << std::endl;

  for (size_t i = 0; i < options.diff_c().size(); ++i) {
    os << "(" << i + 1 << ") "
       << "diff_c =" << options.diff_c()[i] << " cm^2/s, "
       << "diff_T =" << options.diff_T()[i] << ", "
       << "diff_P =" << options.diff_P()[i] << ", "
       << "vm =" << options.vm()[i] << " cm^3/mol, "
       << "diameter=" << options.diameter()[i] << " cm" << std::endl;
  }
}

torch::Tensor EvaporationImpl::forward(
    torch::Tensor T, torch::Tensor P,
    std::map<std::string, torch::Tensor> const& other) {
  // check if T is already expanded
  torch::Tensor log_diff;
  if (T.sizes() == P.sizes()) {  // not yet expanded
    log_diff = log_diff_c + diff_T * (T / options.Tref()).log().unsqueeze(-1) +
               diff_P * (P / options.Pref()).log().unsqueeze(-1);
  } else {
    log_diff = log_diff_c + diff_T * (T / options.Tref()).log() +
               diff_P * (P / options.Pref()).log().unsqueeze(-1);
  }

  // Calculate the rate constant based on the diffusivity and molar volume
  return log(12.) + log_diff + log_vm - 2. * log_diameter;
}

}  // namespace kintera
