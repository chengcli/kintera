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

}  // namespace kintera
