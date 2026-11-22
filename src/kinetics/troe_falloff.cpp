// yaml
#include <yaml-cpp/yaml.h>

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/kinetics/reaction_species.hpp>
#include <kintera/species.hpp>
#include <kintera/units/units.hpp>

#include "arrhenius_params.hpp"
#include "third_body_rate.hpp"
#include "troe_falloff.hpp"

namespace kintera {

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set,
                        TroeFalloffOptions op) {
  add_reaction_species(vapor_set, op, true, true);
}

TroeFalloffOptions TroeFalloffOptionsImpl::from_yaml(
    const YAML::Node& root,
    std::shared_ptr<TroeFalloffOptionsImpl> derived_type_ptr) {
  auto options =
      derived_type_ptr ? derived_type_ptr : TroeFalloffOptionsImpl::create();
  UnitSystem us;

  for (auto const& rxn_node : root) {
    if (!rxn_node["type"]) continue;
    std::string type = rxn_node["type"].as<std::string>();
    if (type != "falloff") continue;

    // Only process if Troe is present
    if (!rxn_node["Troe"]) continue;

    auto [reaction, sum_stoich] = third_body_reaction(rxn_node);

    double k0_A_val = 0., k0_b_val = 0., k0_Ea_R_val = 0.;
    double kinf_A_val = 0., kinf_b_val = 0., kinf_Ea_R_val = 0.;

    TORCH_CHECK(rxn_node["low-P-rate-constant"],
                "'low-P-rate-constant' not defined");
    TORCH_CHECK(rxn_node["high-P-rate-constant"],
                "'high-P-rate-constant' not defined");

    parse_arrhenius_params(rxn_node["low-P-rate-constant"], sum_stoich + 1., us,
                           k0_A_val, k0_b_val, k0_Ea_R_val);
    parse_arrhenius_params(rxn_node["high-P-rate-constant"], sum_stoich, us,
                           kinf_A_val, kinf_b_val, kinf_Ea_R_val);

    auto troe = rxn_node["Troe"];
    double troe_A_val = troe["A"].as<double>();
    double troe_T3_val = troe["T3"].as<double>();
    double troe_T1_val = troe["T1"].as<double>();
    double troe_T2_val = troe["T2"].as<double>(0.);

    reaction.falloff_type("Troe");

    auto eff = parse_efficiencies(rxn_node);
    reaction.efficiencies(eff);

    options->reactions().push_back(reaction);
    options->k0_A().push_back(k0_A_val);
    options->k0_b().push_back(k0_b_val);
    options->k0_Ea_R().push_back(k0_Ea_R_val);
    options->kinf_A().push_back(kinf_A_val);
    options->kinf_b().push_back(kinf_b_val);
    options->kinf_Ea_R().push_back(kinf_Ea_R_val);
    options->troe_A().push_back(troe_A_val);
    options->troe_T3().push_back(troe_T3_val);
    options->troe_T1().push_back(troe_T1_val);
    options->troe_T2().push_back(troe_T2_val);
    options->efficiencies().push_back(eff);
  }

  return options;
}

TroeFalloffImpl::TroeFalloffImpl(TroeFalloffOptions const& options_)
    : options(options_) {
  reset();
}

void TroeFalloffImpl::reset() {
  int nreaction = options->reactions().size();
  if (nreaction == 0) return;

  int nspecies = species_names.size();
  TORCH_CHECK(nspecies > 0, "species_names not initialized");

  k0_A =
      register_buffer("k0_A", torch::tensor(options->k0_A(), torch::kFloat64));
  k0_b =
      register_buffer("k0_b", torch::tensor(options->k0_b(), torch::kFloat64));
  k0_Ea_R = register_buffer("k0_Ea_R",
                            torch::tensor(options->k0_Ea_R(), torch::kFloat64));

  kinf_A = register_buffer("kinf_A",
                           torch::tensor(options->kinf_A(), torch::kFloat64));
  kinf_b = register_buffer("kinf_b",
                           torch::tensor(options->kinf_b(), torch::kFloat64));
  kinf_Ea_R = register_buffer(
      "kinf_Ea_R", torch::tensor(options->kinf_Ea_R(), torch::kFloat64));

  troe_A = register_buffer("troe_A",
                           torch::tensor(options->troe_A(), torch::kFloat64));
  troe_T3 = register_buffer("troe_T3",
                            torch::tensor(options->troe_T3(), torch::kFloat64));
  troe_T1 = register_buffer("troe_T1",
                            torch::tensor(options->troe_T1(), torch::kFloat64));
  troe_T2 = register_buffer("troe_T2",
                            torch::tensor(options->troe_T2(), torch::kFloat64));

  efficiency_matrix = register_buffer(
      "efficiency_matrix",
      efficiency_matrix_from(options->efficiencies(), nreaction, nspecies));
}

void TroeFalloffImpl::pretty_print(std::ostream& os) const {
  os << "Troe Falloff Rate: " << std::endl;

  for (size_t i = 0; i < options->reactions().size(); i++) {
    os << "(" << i + 1 << ") " << options->reactions()[i].equation()
       << std::endl;
    os << "    k0: A = " << options->k0_A()[i] << ", b = " << options->k0_b()[i]
       << ", Ea_R = " << options->k0_Ea_R()[i] << " K" << std::endl;
    os << "    kinf: A = " << options->kinf_A()[i]
       << ", b = " << options->kinf_b()[i]
       << ", Ea_R = " << options->kinf_Ea_R()[i] << " K" << std::endl;
    os << "    Troe: A = " << options->troe_A()[i]
       << ", T3 = " << options->troe_T3()[i]
       << ", T1 = " << options->troe_T1()[i]
       << ", T2 = " << options->troe_T2()[i] << std::endl;

    print_efficiencies(os, options->efficiencies()[i]);
  }
}

torch::Tensor TroeFalloffImpl::compute_k0(torch::Tensor T) const {
  return arrhenius_rate(k0_A, k0_b, k0_Ea_R, T, options->Tref());
}

torch::Tensor TroeFalloffImpl::compute_kinf(torch::Tensor T) const {
  return arrhenius_rate(kinf_A, kinf_b, kinf_Ea_R, T, options->Tref());
}

torch::Tensor TroeFalloffImpl::compute_falloff_factor(torch::Tensor T,
                                                      torch::Tensor Pr) const {
  auto temp = T;

  // Clamp Pr for numerical stability
  auto Pr_clamped = Pr.clamp(1e-300, 1e300);
  auto log10_Pr = torch::log10(Pr_clamped);
  auto log10_Pr_sq = log10_Pr * log10_Pr;
  auto denom = 1.0 + log10_Pr_sq;

  // Vectorized F_cent calculation for all reactions
  // F_cent = (1-A)*exp(-T/T3) + A*exp(-T/T1) + exp(-T2/T) [if T2 != 0]
  auto F_cent =
      (1.0 - troe_A.unsqueeze(0)) * torch::exp(-temp / troe_T3.unsqueeze(0)) +
      troe_A.unsqueeze(0) * torch::exp(-temp / troe_T1.unsqueeze(0));

  // Handle 4-param Troe: add exp(-T2/T) term where T2 != 0
  // Create mask for 4-param reactions (T2 != 0)
  auto is_4param = torch::abs(troe_T2) > 1e-10;
  auto T2_exp_term = torch::exp(-troe_T2.unsqueeze(0) / temp);
  // Broadcast mask to match F_cent shape
  auto out_shape = F_cent.sizes();
  auto is_4param_broadcast =
      is_4param.toType(torch::kFloat64).unsqueeze(0).expand(out_shape);
  F_cent = F_cent + is_4param_broadcast * T2_exp_term;

  // Clamp F_cent for numerical stability
  F_cent = F_cent.clamp(1e-300, 1e300);

  // Vectorized F calculation: F = F_cent^(1/(1 + log10(Pr)^2))
  auto F = F_cent.pow(1.0 / denom);

  return F;
}

torch::Tensor TroeFalloffImpl::forward(
    torch::Tensor T, torch::Tensor P, torch::Tensor C,
    std::map<std::string, torch::Tensor> const& other) {
  int nreaction = options->reactions().size();
  if (nreaction == 0) return empty_rate(T);

  // Only unsqueeze T if it doesn't already have the reaction dimension
  auto temp = T.sizes() == P.sizes() ? T.unsqueeze(-1) : T;
  auto k0 = compute_k0(temp);
  auto kinf = compute_kinf(temp);

  bool expanded =
      expanded_layout(C, nreaction, efficiency_matrix.size(1), other);
  auto M_eff = third_body_concentration(C, efficiency_matrix, expanded);

  // Troe falloff: k = k_Lindemann * F_Troe
  // k_Lindemann = k0*[M]_eff / (1 + Pr)
  auto kinf_clamped = kinf.clamp(1e-100);
  auto Pr = k0 * M_eff / kinf_clamped;
  auto F = compute_falloff_factor(temp, Pr);
  auto k_lindemann = k0 * M_eff / (1.0 + Pr);
  auto result = k_lindemann * F;

  return result;
}

}  // namespace kintera
