// yaml
#include <yaml-cpp/yaml.h>

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/units/units.hpp>

#include "arrhenius.hpp"

namespace kintera {

extern std::vector<std::string> species_names;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set, ArrheniusOptions op) {
  for (auto& react : op.reactions()) {
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
      vapor_set.insert(name);
    }
  }
}

ArrheniusOptions ArrheniusOptions::from_yaml(const YAML::Node& root,
                                             std::string const& other_type) {
  ArrheniusOptions options;

  for (auto const& rxn_node : root) {
    TORCH_CHECK(rxn_node["type"], "Reaction type not specified");

    if (rxn_node["type"].as<std::string>() != "arrhenius" &&
        rxn_node["type"].as<std::string>() != other_type) {
      continue;  // skip this reaction
    }

    TORCH_CHECK(rxn_node["equation"],
                "'equation' is not defined in the reaction");

    std::string equation = rxn_node["equation"].as<std::string>();
    options.reactions().push_back(Reaction(equation));

    // calcualte sum of reactant stoichiometric coefficients
    double sum_stoich = 0.;
    for (const auto& [_, coeff] : options.reactions().back().reactants()) {
      sum_stoich += coeff;
    }

    TORCH_CHECK(rxn_node["rate-constant"],
                "'rate-constant' is not defined in the reaction");

    auto node = rxn_node["rate-constant"];

    // default unit system is [mol, m, s]
    UnitSystem us;

    // input unit system is [molecule, cm, s]
    // [A] []^a []^b ... = molecule cm^-3 s^-1
    // [A] = molecule^(1 - a - b - ...) cm^(-3(1 - a - b - ...)) s^-1
    if (node["A"]) {
      auto unit = fmt::format("molecule^{} * cm^{} * s^-1", 1. - sum_stoich,
                              -3. * (1. - sum_stoich));
      options.A().push_back(us.convert_from(node["A"].as<double>(), unit));
    } else {
      options.A().push_back(1.);
    }

    options.b().push_back(node["b"].as<double>(0.));
    options.Ea_R().push_back(node["Ea_R"].as<double>(1.));
    options.E4_R().push_back(node["E4"].as<double>(0.));
  }

  return options;
}

ArrheniusImpl::ArrheniusImpl(ArrheniusOptions const& options_)
    : options(std::move(options_)) {
  reset();
}

void ArrheniusImpl::reset() {
  A = register_buffer("A", torch::tensor(options.A(), torch::kFloat64));
  b = register_buffer("b", torch::tensor(options.b(), torch::kFloat64));
  Ea_R =
      register_buffer("Ea_R", torch::tensor(options.Ea_R(), torch::kFloat64));
  E4_R =
      register_buffer("E4_R", torch::tensor(options.E4_R(), torch::kFloat64));
}

void ArrheniusImpl::pretty_print(std::ostream& os) const {
  os << "Arrhenius Rate: " << std::endl;

  for (size_t i = 0; i < options.A().size(); i++) {
    os << "(" << i + 1 << ") A = " << options.A()[i]
       << ", b = " << options.b()[i] << ", Ea_R = " << options.Ea_R()[i] << " K"
       << std::endl;
  }
}

torch::Tensor ArrheniusImpl::forward(
    torch::Tensor T, torch::Tensor P, torch::Tensor C,
    std::map<std::string, torch::Tensor> const& other) {
  // expand T if not yet
  auto temp = T.sizes() == P.sizes() ? T.unsqueeze(-1) : T;
  return A * (temp / options.Tref()).pow(b) * torch::exp(-Ea_R / temp);
}

}  // namespace kintera
