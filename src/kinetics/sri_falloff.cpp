// yaml
#include <yaml-cpp/yaml.h>

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/units/units.hpp>

#include "sri_falloff.hpp"

namespace kintera {

extern std::vector<std::string> species_names;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set, SRIFalloffOptions op) {
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

SRIFalloffOptions SRIFalloffOptionsImpl::from_yaml(
    const YAML::Node& root,
    std::shared_ptr<SRIFalloffOptionsImpl> derived_type_ptr) {
  auto options =
      derived_type_ptr ? derived_type_ptr : SRIFalloffOptionsImpl::create();
  UnitSystem us;

  for (auto const& rxn_node : root) {
    if (!rxn_node["type"]) continue;
    std::string type = rxn_node["type"].as<std::string>();
    if (type != "falloff") continue;

    // Only process if SRI is present
    if (!rxn_node["SRI"]) continue;

    TORCH_CHECK(rxn_node["equation"], "'equation' not defined");
    std::string equation = rxn_node["equation"].as<std::string>();
    Reaction reaction(equation);

    double sum_stoich = 0.;
    for (const auto& [name, coeff] : reaction.reactants()) {
      if (name != "M" && name != "(+M)") sum_stoich += coeff;
    }

    double k0_A_val = 0., k0_b_val = 0., k0_Ea_R_val = 0.;
    double kinf_A_val = 0., kinf_b_val = 0., kinf_Ea_R_val = 0.;

    TORCH_CHECK(rxn_node["low-P-rate-constant"], "'low-P-rate-constant' not defined");
    TORCH_CHECK(rxn_node["high-P-rate-constant"], "'high-P-rate-constant' not defined");

    parse_arrhenius_params(rxn_node["low-P-rate-constant"], sum_stoich + 1.,
                           us, k0_A_val, k0_b_val, k0_Ea_R_val);
    parse_arrhenius_params(rxn_node["high-P-rate-constant"], sum_stoich,
                           us, kinf_A_val, kinf_b_val, kinf_Ea_R_val);

    auto sri = rxn_node["SRI"];
    double sri_A_val = sri["A"].as<double>();
    double sri_B_val = sri["B"].as<double>();
    double sri_C_val = sri["C"].as<double>();
    double sri_D_val = sri["D"].as<double>(1.0);
    double sri_E_val = sri["E"].as<double>(0.0);

    reaction.falloff_type("SRI");

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
    options->kinf_A().push_back(kinf_A_val);
    options->kinf_b().push_back(kinf_b_val);
    options->kinf_Ea_R().push_back(kinf_Ea_R_val);
    options->sri_A().push_back(sri_A_val);
    options->sri_B().push_back(sri_B_val);
    options->sri_C().push_back(sri_C_val);
    options->sri_D().push_back(sri_D_val);
    options->sri_E().push_back(sri_E_val);
    options->efficiencies().push_back(eff);
  }

  return options;
}

SRIFalloffImpl::SRIFalloffImpl(SRIFalloffOptions const& options_)
    : options(options_) {
  reset();
}

void SRIFalloffImpl::reset() {
  int nreaction = options->reactions().size();
  if (nreaction == 0) return;

  int nspecies = species_names.size();
  TORCH_CHECK(nspecies > 0, "species_names not initialized");

  k0_A = register_buffer("k0_A", torch::tensor(options->k0_A(), torch::kFloat64));
  k0_b = register_buffer("k0_b", torch::tensor(options->k0_b(), torch::kFloat64));
  k0_Ea_R = register_buffer("k0_Ea_R", torch::tensor(options->k0_Ea_R(), torch::kFloat64));

  kinf_A = register_buffer("kinf_A", torch::tensor(options->kinf_A(), torch::kFloat64));
  kinf_b = register_buffer("kinf_b", torch::tensor(options->kinf_b(), torch::kFloat64));
  kinf_Ea_R = register_buffer("kinf_Ea_R", torch::tensor(options->kinf_Ea_R(), torch::kFloat64));

  sri_A = register_buffer("sri_A", torch::tensor(options->sri_A(), torch::kFloat64));
  sri_B = register_buffer("sri_B", torch::tensor(options->sri_B(), torch::kFloat64));
  sri_C = register_buffer("sri_C", torch::tensor(options->sri_C(), torch::kFloat64));
  sri_D = register_buffer("sri_D", torch::tensor(options->sri_D(), torch::kFloat64));
  sri_E = register_buffer("sri_E", torch::tensor(options->sri_E(), torch::kFloat64));

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

void SRIFalloffImpl::pretty_print(std::ostream& os) const {
  os << "SRI Falloff Rate: " << std::endl;

  for (size_t i = 0; i < options->reactions().size(); i++) {
    os << "(" << i + 1 << ") " << options->reactions()[i].equation() << std::endl;
    os << "    k0: A = " << options->k0_A()[i] << ", b = " << options->k0_b()[i]
       << ", Ea_R = " << options->k0_Ea_R()[i] << " K" << std::endl;
    os << "    kinf: A = " << options->kinf_A()[i] << ", b = " << options->kinf_b()[i]
       << ", Ea_R = " << options->kinf_Ea_R()[i] << " K" << std::endl;
    os << "    SRI: A = " << options->sri_A()[i] << ", B = " << options->sri_B()[i]
       << ", C = " << options->sri_C()[i] << ", D = " << options->sri_D()[i]
       << ", E = " << options->sri_E()[i] << std::endl;

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

torch::Tensor SRIFalloffImpl::compute_k0(torch::Tensor T) const {
  auto Tref = options->Tref();
  return k0_A * (T / Tref).pow(k0_b) * torch::exp(-k0_Ea_R / T);
}

torch::Tensor SRIFalloffImpl::compute_kinf(torch::Tensor T) const {
  auto Tref = options->Tref();
  return kinf_A * (T / Tref).pow(kinf_b) * torch::exp(-kinf_Ea_R / T);
}

torch::Tensor SRIFalloffImpl::compute_falloff_factor(torch::Tensor T, torch::Tensor Pr) const {
  auto temp = T;
  
  // Clamp Pr for numerical stability
  auto Pr_clamped = Pr.clamp(1e-300, 1e300);
  auto log10_Pr = torch::log10(Pr_clamped);
  auto log10_Pr_sq = log10_Pr * log10_Pr;
  auto denom = 1.0 + log10_Pr_sq;
  
  // Vectorized F_cent calculation for all reactions
  // F_cent = A*exp(-B/T) + exp(-T/C) + D*exp(-E/T) [for 5-param, where E != 0]
  // F_cent = A*exp(-B/T) + exp(-T/C) [for 3-param, where D=1.0, E=0.0]
  auto F_cent = sri_A.unsqueeze(0) * torch::exp(-sri_B.unsqueeze(0) / temp) +
                torch::exp(-temp / sri_C.unsqueeze(0));
  
  // Handle 5-param SRI: add D*exp(-E/T) term where E != 0
  // Create mask for 5-param reactions (E != 0)
  auto is_5param = torch::abs(sri_E) > 1e-10;
  auto E_exp_term = sri_D.unsqueeze(0) * torch::exp(-sri_E.unsqueeze(0) / temp);
  // Broadcast mask to match F_cent shape
  auto out_shape = F_cent.sizes();
  std::vector<int64_t> mask_shape;
  for (int64_t i = 0; i < out_shape.size() - 1; i++) {
    mask_shape.push_back(1);
  }
  mask_shape.push_back(out_shape.back());
  auto is_5param_broadcast = is_5param.toType(torch::kFloat64).unsqueeze(0).expand(out_shape);
  F_cent = F_cent + is_5param_broadcast * E_exp_term;
  
  // Clamp F_cent for numerical stability
  F_cent = F_cent.clamp(1e-300, 1e300);
  
  // Vectorized F calculation: F = F_cent^(1/(1 + log10(Pr)^2))
  auto F = F_cent.pow(1.0 / denom);
  
  return F;
}

torch::Tensor SRIFalloffImpl::forward(torch::Tensor T, torch::Tensor P, torch::Tensor C,
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
  auto kinf = compute_kinf(temp);

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

  // SRI falloff: k = k_Lindemann * F_SRI
  // k_Lindemann = k0*[M]_eff / (1 + Pr)
  auto kinf_clamped = kinf.clamp(1e-100);
  auto Pr = k0 * M_eff / kinf_clamped;
  auto F = compute_falloff_factor(temp, Pr);
  auto k_lindemann = k0 * M_eff / (1.0 + Pr);
  auto result = k_lindemann * F;

  return result;
}

}  // namespace kintera
