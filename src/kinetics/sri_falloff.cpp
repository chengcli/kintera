// yaml
#include <yaml-cpp/yaml.h>

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/kinetics/reaction_species.hpp>
#include <kintera/species.hpp>
#include <kintera/units/units.hpp>

#include "arrhenius_params.hpp"
#include "sri_falloff.hpp"
#include "third_body_rate.hpp"

namespace kintera {

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set,
                        SRIFalloffOptions op) {
  add_reaction_species(vapor_set, op, true, true);
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

    auto sri = rxn_node["SRI"];
    double sri_A_val = sri["A"].as<double>();
    double sri_B_val = sri["B"].as<double>();
    double sri_C_val = sri["C"].as<double>();
    double sri_D_val = sri["D"].as<double>(1.0);
    double sri_E_val = sri["E"].as<double>(0.0);

    reaction.falloff_type("SRI");

    auto eff = parse_efficiencies(rxn_node);
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

  sri_A = register_buffer("sri_A",
                          torch::tensor(options->sri_A(), torch::kFloat64));
  sri_B = register_buffer("sri_B",
                          torch::tensor(options->sri_B(), torch::kFloat64));
  sri_C = register_buffer("sri_C",
                          torch::tensor(options->sri_C(), torch::kFloat64));
  sri_D = register_buffer("sri_D",
                          torch::tensor(options->sri_D(), torch::kFloat64));
  sri_E = register_buffer("sri_E",
                          torch::tensor(options->sri_E(), torch::kFloat64));

  efficiency_matrix = register_buffer(
      "efficiency_matrix",
      efficiency_matrix_from(options->efficiencies(), nreaction, nspecies));
}

void SRIFalloffImpl::pretty_print(std::ostream& os) const {
  os << "SRI Falloff Rate: " << std::endl;

  for (size_t i = 0; i < options->reactions().size(); i++) {
    os << "(" << i + 1 << ") " << options->reactions()[i].equation()
       << std::endl;
    os << "    k0: A = " << options->k0_A()[i] << ", b = " << options->k0_b()[i]
       << ", Ea_R = " << options->k0_Ea_R()[i] << " K" << std::endl;
    os << "    kinf: A = " << options->kinf_A()[i]
       << ", b = " << options->kinf_b()[i]
       << ", Ea_R = " << options->kinf_Ea_R()[i] << " K" << std::endl;
    os << "    SRI: A = " << options->sri_A()[i]
       << ", B = " << options->sri_B()[i] << ", C = " << options->sri_C()[i]
       << ", D = " << options->sri_D()[i] << ", E = " << options->sri_E()[i]
       << std::endl;

    print_efficiencies(os, options->efficiencies()[i]);
  }
}

torch::Tensor SRIFalloffImpl::compute_k0(torch::Tensor T) const {
  return arrhenius_rate(k0_A, k0_b, k0_Ea_R, T, options->Tref());
}

torch::Tensor SRIFalloffImpl::compute_kinf(torch::Tensor T) const {
  return arrhenius_rate(kinf_A, kinf_b, kinf_Ea_R, T, options->Tref());
}

torch::Tensor SRIFalloffImpl::compute_falloff_factor(torch::Tensor T,
                                                     torch::Tensor Pr) const {
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
  auto is_5param_broadcast =
      is_5param.toType(torch::kFloat64).unsqueeze(0).expand(out_shape);
  F_cent = F_cent + is_5param_broadcast * E_exp_term;

  // Clamp F_cent for numerical stability
  F_cent = F_cent.clamp(1e-300, 1e300);

  // Vectorized F calculation: F = F_cent^(1/(1 + log10(Pr)^2))
  auto F = F_cent.pow(1.0 / denom);

  return F;
}

torch::Tensor SRIFalloffImpl::forward(
    torch::Tensor T, torch::Tensor P, torch::Tensor C,
    std::map<std::string, torch::Tensor> const& other) {
  int nreaction = options->reactions().size();
  if (nreaction == 0) return empty_rate(T);

  // Only unsqueeze T if it doesn't already have the reaction dimension
  auto temp = T.sizes() == P.sizes() ? T.unsqueeze(-1) : T;
  auto k0 = compute_k0(temp);
  auto kinf = compute_kinf(temp);

  bool expanded = C.dim() >= 2 && C.size(-1) == nreaction;
  auto M_eff = third_body_concentration(C, efficiency_matrix, expanded);

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
