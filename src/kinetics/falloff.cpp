// yaml
#include <yaml-cpp/yaml.h>

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/units/units.hpp>

#include "falloff.hpp"

namespace kintera {

extern std::vector<std::string> species_names;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set, FalloffOptions op) {
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
  // Array format [A, b, Ea]
  if (node.IsSequence() && node.size() >= 3) {
    auto unit = fmt::format("molecule^{} * cm^{} * s^-1", 1. - sum_stoich,
                            -3. * (1. - sum_stoich));
    A = us.convert_from(node[0].as<double>(), unit);
    b = node[1].as<double>();
    Ea_R = node[2].as<double>() / 1.987;
    return;
  }

  // Map format {A: ..., b: ..., Ea: ...}
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

FalloffOptions FalloffOptionsImpl::from_yaml(
    const YAML::Node& root,
    std::shared_ptr<FalloffOptionsImpl> derived_type_ptr) {
  auto options =
      derived_type_ptr ? derived_type_ptr : FalloffOptionsImpl::create();
  UnitSystem us;

  for (auto const& rxn_node : root) {
    if (!rxn_node["type"]) continue;
    std::string type = rxn_node["type"].as<std::string>();
    if (type != "three-body" && type != "falloff") continue;

    TORCH_CHECK(rxn_node["equation"], "'equation' not defined");
    std::string equation = rxn_node["equation"].as<std::string>();
    Reaction reaction(equation);

    double sum_stoich = 0.;
    for (const auto& [name, coeff] : reaction.reactants()) {
      if (name != "M" && name != "(+M)") sum_stoich += coeff;
    }
    if (type == "three-body") sum_stoich += 1.0;

    double k0_A_val = 0., k0_b_val = 0., k0_Ea_R_val = 0.;
    double kinf_A_val = 0., kinf_b_val = 0., kinf_Ea_R_val = 0.;
    bool is_three_body = false;
    FalloffType falloff_type = FalloffType::None;

    double troe_A_val = 0., troe_T3_val = 0., troe_T1_val = 0., troe_T2_val = 0.;
    double sri_A_val = 0., sri_B_val = 0., sri_C_val = 0.;
    double sri_D_val = 1.0, sri_E_val = 0.0;

    if (type == "three-body") {
      TORCH_CHECK(rxn_node["rate-constant"], "'rate-constant' not defined");
      parse_arrhenius_params(rxn_node["rate-constant"], sum_stoich, us,
                             k0_A_val, k0_b_val, k0_Ea_R_val);
      kinf_A_val = 1e100;
      is_three_body = true;
      reaction.falloff_type("none");
    } else {
      TORCH_CHECK(rxn_node["low-P-rate-constant"], "'low-P-rate-constant' not defined");
      TORCH_CHECK(rxn_node["high-P-rate-constant"], "'high-P-rate-constant' not defined");

      parse_arrhenius_params(rxn_node["low-P-rate-constant"], sum_stoich + 1.,
                             us, k0_A_val, k0_b_val, k0_Ea_R_val);
      parse_arrhenius_params(rxn_node["high-P-rate-constant"], sum_stoich,
                             us, kinf_A_val, kinf_b_val, kinf_Ea_R_val);

      if (rxn_node["Troe"]) {
        auto troe = rxn_node["Troe"];
        troe_A_val = troe["A"].as<double>();
        troe_T3_val = troe["T3"].as<double>();
        troe_T1_val = troe["T1"].as<double>();
        troe_T2_val = troe["T2"].as<double>(0.);
        falloff_type = FalloffType::Troe;
        reaction.falloff_type("Troe");
      } else if (rxn_node["SRI"]) {
        auto sri = rxn_node["SRI"];
        sri_A_val = sri["A"].as<double>();
        sri_B_val = sri["B"].as<double>();
        sri_C_val = sri["C"].as<double>();
        sri_D_val = sri["D"].as<double>(1.0);
        sri_E_val = sri["E"].as<double>(0.0);
        falloff_type = FalloffType::SRI;
        reaction.falloff_type("SRI");
      } else {
        reaction.falloff_type("none");
      }
    }

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
    options->falloff_types().push_back(static_cast<int>(falloff_type));
    options->is_three_body().push_back(is_three_body);

    options->troe_A().push_back(troe_A_val);
    options->troe_T3().push_back(troe_T3_val);
    options->troe_T1().push_back(troe_T1_val);
    options->troe_T2().push_back(troe_T2_val);

    options->sri_A().push_back(sri_A_val);
    options->sri_B().push_back(sri_B_val);
    options->sri_C().push_back(sri_C_val);
    options->sri_D().push_back(sri_D_val);
    options->sri_E().push_back(sri_E_val);

    options->efficiencies().push_back(eff);
  }

  return options;
}

FalloffImpl::FalloffImpl(FalloffOptions const& options_)
    : options(options_) {
  reset();
}

void FalloffImpl::reset() {
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

  std::vector<int> falloff_type_vec;
  for (auto ft : options->falloff_types()) {
    falloff_type_vec.push_back(ft);
  }
  falloff_type_flags = register_buffer("falloff_type_flags", torch::tensor(falloff_type_vec, torch::kInt32));

  std::vector<int> is_three_body_vec;
  for (auto itb : options->is_three_body()) {
    is_three_body_vec.push_back(itb ? 1 : 0);
  }
  is_three_body = register_buffer("is_three_body", torch::tensor(is_three_body_vec, torch::kInt32));

  troe_A = register_buffer("troe_A", torch::tensor(options->troe_A(), torch::kFloat64));
  troe_T3 = register_buffer("troe_T3", torch::tensor(options->troe_T3(), torch::kFloat64));
  troe_T1 = register_buffer("troe_T1", torch::tensor(options->troe_T1(), torch::kFloat64));
  troe_T2 = register_buffer("troe_T2", torch::tensor(options->troe_T2(), torch::kFloat64));

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

void FalloffImpl::pretty_print(std::ostream& os) const {
  os << "Falloff Rate: " << std::endl;

  for (size_t i = 0; i < options->reactions().size(); i++) {
    os << "(" << i + 1 << ") " << options->reactions()[i].equation() << std::endl;
    os << "    k0: A = " << options->k0_A()[i] << ", b = " << options->k0_b()[i]
       << ", Ea_R = " << options->k0_Ea_R()[i] << " K" << std::endl;
    os << "    kinf: A = " << options->kinf_A()[i] << ", b = " << options->kinf_b()[i]
       << ", Ea_R = " << options->kinf_Ea_R()[i] << " K" << std::endl;

    if (options->is_three_body()[i]) {
      os << "    Type: three-body" << std::endl;
    } else {
      os << "    Type: falloff (" << falloff_type_to_string(static_cast<FalloffType>(options->falloff_types()[i])) << ")" << std::endl;
    }

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

torch::Tensor FalloffImpl::compute_k0(torch::Tensor T) const {
  auto Tref = options->Tref();
  auto temp = T.unsqueeze(-1);
  return k0_A * (temp / Tref).pow(k0_b) * torch::exp(-k0_Ea_R / temp);
}

torch::Tensor FalloffImpl::compute_kinf(torch::Tensor T) const {
  auto Tref = options->Tref();
  auto temp = T.unsqueeze(-1);
  return kinf_A * (temp / Tref).pow(kinf_b) * torch::exp(-kinf_Ea_R / temp);
}

torch::Tensor FalloffImpl::compute_falloff_factor(torch::Tensor T, torch::Tensor Pr) const {
  int nreaction = efficiency_matrix.size(0);
  auto temp = T.unsqueeze(-1);
  auto result = torch::ones_like(Pr);
  
  auto Pr_clamped = Pr.clamp(1e-300, 1e300);
  auto log10_Pr = torch::log10(Pr_clamped);
  auto log10_Pr_sq = log10_Pr * log10_Pr;
  auto denom = 1.0 + log10_Pr_sq;
  
  for (int i = 0; i < nreaction; i++) {
    auto falloff_type = static_cast<FalloffType>(falloff_type_flags[i].item<int>());
    
    if (falloff_type == FalloffType::None) {
      continue;
    }
    
    torch::Tensor F_cent;
    
    if (falloff_type == FalloffType::Troe) {
      auto troe_A_val = troe_A[i].item<double>();
      auto troe_T3_val = troe_T3[i].item<double>();
      auto troe_T1_val = troe_T1[i].item<double>();
      auto troe_T2_val = troe_T2[i].item<double>();
      
      F_cent = (1.0 - troe_A_val) * torch::exp(-temp / troe_T3_val) +
                troe_A_val * torch::exp(-temp / troe_T1_val);
      
      if (std::abs(troe_T2_val) > 1e-10) {
        F_cent = F_cent + torch::exp(-troe_T2_val / temp);
      }
    } else if (falloff_type == FalloffType::SRI) {
      auto sri_A_val = sri_A[i].item<double>();
      auto sri_B_val = sri_B[i].item<double>();
      auto sri_C_val = sri_C[i].item<double>();
      auto sri_D_val = sri_D[i].item<double>();
      auto sri_E_val = sri_E[i].item<double>();
      
      F_cent = sri_A_val * torch::exp(-sri_B_val / temp) +
                torch::exp(-temp / sri_C_val);
      
      if (std::abs(sri_E_val) > 1e-10) {
        F_cent = F_cent + sri_D_val * torch::exp(-sri_E_val / temp);
      }
    } else {
      continue;
    }
    
    F_cent = F_cent.clamp(1e-300, 1e300);
    auto denom_i = denom.select(-1, i).unsqueeze(-1);
    auto F = F_cent.pow(1.0 / denom_i);
    result.select(-1, i).copy_(F.squeeze(-1));
  }
  
  return result;
}

torch::Tensor FalloffImpl::forward(torch::Tensor T, torch::Tensor P, torch::Tensor C,
                                    std::map<std::string, torch::Tensor> const& other) {
  int nreaction = options->reactions().size();
  if (nreaction == 0) {
    auto out_shape = T.sizes().vec();
    out_shape.push_back(0);
    return torch::empty(out_shape, T.options());
  }
  
  auto k0 = compute_k0(T);
  auto kinf = compute_kinf(T);
  
  torch::Tensor M_eff;
  int nspecies_kinetics, nspecies_full = efficiency_matrix.size(1);
  
  if (C.dim() >= 2 && C.size(-1) == nreaction) {
    // C has shape (..., nspecies, nreaction)
    // Extract concentration: C[..., :, 0] -> (..., nspecies)
    // Use select() to preserve gradients for autograd
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
  
  auto kinf_clamped = kinf.clamp(1e-100);
  auto Pr = k0 * M_eff / kinf_clamped;
  auto F = compute_falloff_factor(T, Pr);
  
  auto is_three_body_bool = is_three_body.toType(torch::kBool);
  auto k_three_body = k0 * M_eff;
  auto k_lindemann = k0 * M_eff / (1.0 + Pr);
  auto k_falloff = k_lindemann * F;
  
  auto out_shape = k0.sizes();
  std::vector<int64_t> mask_shape;
  for (int64_t i = 0; i < out_shape.size() - 1; i++) {
    mask_shape.push_back(1);
  }
  mask_shape.push_back(nreaction);
  
  auto mask = is_three_body_bool.view(mask_shape).expand(out_shape);
  auto result = torch::where(mask, k_three_body, k_falloff);
  
  return result;
}

}  // namespace kintera
