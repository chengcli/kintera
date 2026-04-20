// yaml
#include <yaml-cpp/yaml.h>

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/constants.h>

#include <kintera/units/units.hpp>

#include "three_body.hpp"

namespace kintera {

extern std::vector<std::string> species_names;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set, ThreeBodyOptions op) {
  for (auto& react : op->reactions()) {
    for (auto& [name, _] : react.reactants()) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      if (it == species_names.end()) continue;  // skip background/bath species
      vapor_set.insert(name);
    }
    for (auto& [name, _] : react.products()) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      if (it == species_names.end()) continue;  // skip background/bath species
      vapor_set.insert(name);
    }
  }
}

ThreeBodyOptions ThreeBodyOptionsImpl::from_yaml(const YAML::Node& root) {
  auto options = ThreeBodyOptionsImpl::create();

  for (auto const& rxn_node : root) {
    TORCH_CHECK(rxn_node["type"], "Reaction type not specified");

    if (rxn_node["type"].as<std::string>() != options->name()) {
      continue;
    }

    TORCH_CHECK(rxn_node["equation"],
                "'equation' is not defined in the reaction");
    std::string equation = rxn_node["equation"].as<std::string>();
    options->reactions().push_back(Reaction(equation));

    // Sum of reactant stoichiometric coefficients (for mass-action order)
    double sum_stoich = 0.;
    for (const auto& [_, coeff] : options->reactions().back().reactants()) {
      sum_stoich += coeff;
    }

    TORCH_CHECK(rxn_node["rate-constant"],
                "'rate-constant' is not defined in the reaction");
    auto rc = rxn_node["rate-constant"];

    // Default unit system: convert from (molecule, cm, s) to (mol, m, s)
    UnitSystem us;

    // --- Low-pressure (termolecular) rate constant ---
    // k0 has one extra order from the third body M
    TORCH_CHECK(rc["low-P-rate-constant"],
                "'low-P-rate-constant' is not defined for three-body "
                "reaction: ",
                equation);
    auto low_P = rc["low-P-rate-constant"];
    {
      double order_k0 = sum_stoich + 1.0;
      auto unit = fmt::format("molecule^{} * cm^{} * s^-1", 1. - order_k0,
                              -3. * (1. - order_k0));
      if (low_P["A"]) {
        options->A0().push_back(us.convert_from(low_P["A"].as<double>(), unit));
      } else {
        options->A0().push_back(1.);
      }
      options->b0().push_back(low_P["b"].as<double>(0.));
      options->Ea_R0().push_back(low_P["Ea_R"].as<double>(0.));
    }

    // --- High-pressure (bimolecular) rate constant ---
    TORCH_CHECK(rc["high-P-rate-constant"],
                "'high-P-rate-constant' is not defined for three-body "
                "reaction: ",
                equation);
    auto high_P = rc["high-P-rate-constant"];
    {
      double order_kinf = sum_stoich;
      auto unit = fmt::format("molecule^{} * cm^{} * s^-1", 1. - order_kinf,
                              -3. * (1. - order_kinf));
      if (high_P["A"]) {
        options->Ainf().push_back(
            us.convert_from(high_P["A"].as<double>(), unit));
      } else {
        options->Ainf().push_back(1.);
      }
      options->binf().push_back(high_P["b"].as<double>(0.));
      options->Ea_Rinf().push_back(high_P["Ea_R"].as<double>(0.));
    }
  }

  return options;
}

ThreeBodyImpl::ThreeBodyImpl(ThreeBodyOptions const& options_)
    : options(options_) {
  reset();
}

void ThreeBodyImpl::reset() {
  A0 = register_buffer("A0", torch::tensor(options->A0(), torch::kFloat64));
  b0 = register_buffer("b0", torch::tensor(options->b0(), torch::kFloat64));
  Ea_R0 = register_buffer("Ea_R0",
                          torch::tensor(options->Ea_R0(), torch::kFloat64));
  Ainf =
      register_buffer("Ainf", torch::tensor(options->Ainf(), torch::kFloat64));
  binf =
      register_buffer("binf", torch::tensor(options->binf(), torch::kFloat64));
  Ea_Rinf = register_buffer("Ea_Rinf",
                            torch::tensor(options->Ea_Rinf(), torch::kFloat64));
}

void ThreeBodyImpl::pretty_print(std::ostream& os) const {
  os << "ThreeBody Rate (Lindemann):" << std::endl;
  for (size_t i = 0; i < options->A0().size(); i++) {
    os << "  (" << i + 1 << ") k0: A0=" << options->A0()[i]
       << ", b0=" << options->b0()[i] << ", Ea_R0=" << options->Ea_R0()[i]
       << " K" << std::endl;
    os << "       kinf: Ainf=" << options->Ainf()[i]
       << ", binf=" << options->binf()[i]
       << ", Ea_Rinf=" << options->Ea_Rinf()[i] << " K" << std::endl;
  }
}

torch::Tensor ThreeBodyImpl::forward(
    torch::Tensor T, torch::Tensor P, torch::Tensor C,
    std::map<std::string, torch::Tensor> const& other) {
  if (options->reactions().empty()) {
    return torch::empty({0}, T.options());
  }

  // T shape: (...) or (..., nreaction) if expanded for autograd
  // P shape: (...)
  auto temp = T.sizes() == P.sizes() ? T.unsqueeze(-1) : T;
  auto pres = P.unsqueeze(-1);

  // Low-pressure Arrhenius: k0 = A0 * T^b0 * exp(-Ea_R0 / T)
  auto k0 = A0 * temp.pow(b0) * torch::exp(-Ea_R0 / temp);

  // High-pressure Arrhenius: kinf = Ainf * T^binf * exp(-Ea_Rinf / T)
  auto kinf = Ainf * temp.pow(binf) * torch::exp(-Ea_Rinf / temp);

  // Total concentration [M] = P / (R * T) in mol/m^3
  auto M = pres / (constants::Rgas * temp);

  // Standard Lindemann: k_eff = k0 * [M] / (1 + k0 * [M] / kinf)
  auto k0M = k0 * M;
  return k0M / (1.0 + k0M / kinf);
}

}  // namespace kintera
