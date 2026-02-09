// yaml
#include <yaml-cpp/yaml.h>

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/units/units.hpp>

#include "three_body.hpp"

namespace kintera {

extern std::vector<std::string> species_names;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set, ThreeBodyOptions op) {
  for (auto& react : op->reactions()) {
    for (auto& [name, _] : react.reactants()) {
      if (name == "M" || name == "(+M)") continue;
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name, " not found");
      vapor_set.insert(name);
    }

    for (auto& [name, _] : react.products()) {
      if (name == "M" || name == "(+M)") continue;
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name, " not found");
      vapor_set.insert(name);
    }

    for (auto& [name, _] : react.efficiencies()) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name, " not found");
      vapor_set.insert(name);
    }
  }
}

static void parse_arrhenius_params(const YAML::Node& node, double sum_stoich,
                                   UnitSystem& us, double& A, double& b,
                                   double& Ea_R) {
  // Map format {A: ..., b: ..., Ea: ...} - consistent with Arrhenius
  if (node["A"]) {
    auto unit = fmt::format("molecule^{} * cm^{} * s^-1", 1. - sum_stoich,
                            -3. * (1. - sum_stoich));
    A = us.convert_from(node["A"].as<double>(), unit);
  } else {
    A = 1.;
  }

  b = node["b"].as<double>(0.);

  if (node["Ea_R"]) {
    Ea_R = node["Ea_R"].as<double>();
  } else if (node["Ea"]) {
    std::string ea_str = node["Ea"].as<std::string>();
    double ea_val = 0.0;
    std::string ea_unit = "";
    std::istringstream iss(ea_str);
    iss >> ea_val >> ea_unit;

    if (ea_unit.find("cal") != std::string::npos) {
      Ea_R = ea_val / 1.987;
    } else if (ea_unit.find("kJ") != std::string::npos) {
      Ea_R = ea_val * 1000.0 / 8.314;
    } else if (ea_unit.find("J") != std::string::npos) {
      Ea_R = ea_val / 8.314;
    } else {
      Ea_R = ea_val;
    }
  } else {
    Ea_R = 0.;
  }
}

ThreeBodyOptions ThreeBodyOptionsImpl::from_yaml(
    const YAML::Node& root,
    std::shared_ptr<ThreeBodyOptionsImpl> derived_type_ptr) {
  auto options =
      derived_type_ptr ? derived_type_ptr : ThreeBodyOptionsImpl::create();
  UnitSystem us;

  for (auto const& rxn_node : root) {
    if (!rxn_node["type"]) continue;
    std::string type = rxn_node["type"].as<std::string>();
    if (type != "three-body") continue;

    TORCH_CHECK(rxn_node["equation"], "'equation' not defined");
    std::string equation = rxn_node["equation"].as<std::string>();
    Reaction reaction(equation);

    double sum_stoich = 0.;
    for (const auto& [name, coeff] : reaction.reactants()) {
      if (name != "M" && name != "(+M)") sum_stoich += coeff;
    }
    sum_stoich += 1.0;  // Add 1 for M

    double k0_A_val = 0., k0_b_val = 0., k0_Ea_R_val = 0.;

    TORCH_CHECK(rxn_node["rate-constant"], "'rate-constant' not defined");
    parse_arrhenius_params(rxn_node["rate-constant"], sum_stoich, us,
                           k0_A_val, k0_b_val, k0_Ea_R_val);

    Composition eff;
    if (rxn_node["efficiencies"]) {
      for (auto const& eff_node : rxn_node["efficiencies"]) {
        eff[eff_node.first.as<std::string>()] = eff_node.second.as<double>();
      }
    }
    reaction.efficiencies(eff);

    options->reactions().push_back(reaction);
    options->k0_A().push_back(k0_A_val);
    options->k0_b().push_back(k0_b_val);
    options->k0_Ea_R().push_back(k0_Ea_R_val);
    options->efficiencies().push_back(eff);
  }

  return options;
}

ThreeBodyImpl::ThreeBodyImpl(ThreeBodyOptions const& options_)
    : options(options_) {
  reset();
}

void ThreeBodyImpl::reset() {
  int nreaction = options->reactions().size();
  if (nreaction == 0) return;

  int nspecies = species_names.size();
  TORCH_CHECK(nspecies > 0, "species_names not initialized");

  k0_A = register_buffer("k0_A", torch::tensor(options->k0_A(), torch::kFloat64));
  k0_b = register_buffer("k0_b", torch::tensor(options->k0_b(), torch::kFloat64));
  k0_Ea_R = register_buffer("k0_Ea_R", torch::tensor(options->k0_Ea_R(), torch::kFloat64));

  // Build efficiency matrix: (nreaction, nspecies), default efficiency = 1.0
  std::vector<double> eff_matrix_data(nreaction * nspecies, 1.0);

  for (int i = 0; i < nreaction; i++) {
    const auto& eff_map = options->efficiencies()[i];
    for (int j = 0; j < nspecies; j++) {
      const std::string& sp_name = species_names[j];
      auto it = eff_map.find(sp_name);
      if (it != eff_map.end()) {
        eff_matrix_data[i * nspecies + j] = it->second;
      }
    }
  }

  efficiency_matrix = register_buffer(
      "efficiency_matrix",
      torch::tensor(eff_matrix_data, torch::kFloat64).view({nreaction, nspecies}));
}

void ThreeBodyImpl::pretty_print(std::ostream& os) const {
  os << "Three-Body Rate: " << std::endl;

  for (size_t i = 0; i < options->reactions().size(); i++) {
    os << "(" << i + 1 << ") " << options->reactions()[i].equation() << std::endl;
    os << "    k0: A = " << options->k0_A()[i] << ", b = " << options->k0_b()[i]
       << ", Ea_R = " << options->k0_Ea_R()[i] << " K" << std::endl;

    const auto& eff = options->efficiencies()[i];
    if (!eff.empty()) {
      os << "    Efficiencies: ";
      bool first = true;
      for (const auto& [sp, val] : eff) {
        if (!first) os << ", ";
        os << sp << "=" << val;
        first = false;
      }
      os << std::endl;
    }
  }
}

torch::Tensor ThreeBodyImpl::compute_k0(torch::Tensor T) const {
  auto Tref = options->Tref();
  return k0_A * (T / Tref).pow(k0_b) * torch::exp(-k0_Ea_R / T);
}

torch::Tensor ThreeBodyImpl::forward(torch::Tensor T, torch::Tensor P, torch::Tensor C,
                                      std::map<std::string, torch::Tensor> const& other) {
  int nreaction = options->reactions().size();
  if (nreaction == 0) {
    auto out_shape = T.sizes().vec();
    out_shape.push_back(0);
    return torch::empty(out_shape, T.options());
  }

  // Only unsqueeze T if it doesn't already have the reaction dimension
  auto temp = T.sizes() == P.sizes() ? T.unsqueeze(-1) : T;
  auto k0 = compute_k0(temp);

  torch::Tensor M_eff;
  int nspecies_kinetics, nspecies_full = efficiency_matrix.size(1);

  if (C.dim() >= 2 && C.size(-1) == nreaction) {
    // C has shape (..., nspecies, nreaction)
    // Extract concentration: C[..., :, 0] -> (..., nspecies)
    int last_dim = C.dim() - 1;
    auto C_actual = C.select(last_dim, 0);  // (..., nspecies)
    nspecies_kinetics = C_actual.size(-1);

    auto eff_matrix_kinetics = efficiency_matrix.narrow(1, 0, std::min(nspecies_kinetics, nspecies_full));
    auto eff_T = eff_matrix_kinetics.transpose(0, 1);
    M_eff = torch::matmul(C_actual, eff_T);
  } else {
    nspecies_kinetics = C.size(-1);
    auto eff_matrix_kinetics = efficiency_matrix.narrow(1, 0, std::min(nspecies_kinetics, nspecies_full));
    auto eff_T = eff_matrix_kinetics.transpose(0, 1);
    M_eff = torch::matmul(C, eff_T);
  }

  // Three-body: k = k0 * [M]_eff
  auto result = k0 * M_eff;

  return result;
}

}  // namespace kintera
