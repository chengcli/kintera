// yaml
#include <yaml-cpp/yaml.h>

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/kinetics/reaction_species.hpp>
#include <kintera/species.hpp>
#include <kintera/units/units.hpp>

#include "arrhenius_params.hpp"
#include "lindemann_falloff.hpp"
#include "third_body_rate.hpp"

namespace kintera {

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set,
                        LindemannFalloffOptions op) {
  add_reaction_species(vapor_set, op, true, true);
}

LindemannFalloffOptions LindemannFalloffOptionsImpl::from_yaml(
    const YAML::Node& root,
    std::shared_ptr<LindemannFalloffOptionsImpl> derived_type_ptr) {
  auto options = derived_type_ptr ? derived_type_ptr
                                  : LindemannFalloffOptionsImpl::create();
  UnitSystem us;

  for (auto const& rxn_node : root) {
    if (!rxn_node["type"]) continue;
    std::string type = rxn_node["type"].as<std::string>();
    if (type != "falloff") continue;

    // Skip if Troe or SRI is present
    if (rxn_node["Troe"] || rxn_node["SRI"]) continue;

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

    reaction.falloff_type("none");

    auto eff = parse_efficiencies(rxn_node);
    reaction.efficiencies(eff);

    options->reactions().push_back(reaction);
    options->k0_A().push_back(k0_A_val);
    options->k0_b().push_back(k0_b_val);
    options->k0_Ea_R().push_back(k0_Ea_R_val);
    options->kinf_A().push_back(kinf_A_val);
    options->kinf_b().push_back(kinf_b_val);
    options->kinf_Ea_R().push_back(kinf_Ea_R_val);
    options->efficiencies().push_back(eff);
  }

  return options;
}

LindemannFalloffImpl::LindemannFalloffImpl(
    LindemannFalloffOptions const& options_)
    : options(options_) {
  reset();
}

void LindemannFalloffImpl::reset() {
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

  efficiency_matrix = register_buffer(
      "efficiency_matrix",
      efficiency_matrix_from(options->efficiencies(), nreaction, nspecies));
}

void LindemannFalloffImpl::pretty_print(std::ostream& os) const {
  os << "Lindemann Falloff Rate: " << std::endl;

  for (size_t i = 0; i < options->reactions().size(); i++) {
    os << "(" << i + 1 << ") " << options->reactions()[i].equation()
       << std::endl;
    os << "    k0: A = " << options->k0_A()[i] << ", b = " << options->k0_b()[i]
       << ", Ea_R = " << options->k0_Ea_R()[i] << " K" << std::endl;
    os << "    kinf: A = " << options->kinf_A()[i]
       << ", b = " << options->kinf_b()[i]
       << ", Ea_R = " << options->kinf_Ea_R()[i] << " K" << std::endl;

    print_efficiencies(os, options->efficiencies()[i]);
  }
}

torch::Tensor LindemannFalloffImpl::compute_k0(torch::Tensor T) const {
  return arrhenius_rate(k0_A, k0_b, k0_Ea_R, T, options->Tref());
}

torch::Tensor LindemannFalloffImpl::compute_kinf(torch::Tensor T) const {
  return arrhenius_rate(kinf_A, kinf_b, kinf_Ea_R, T, options->Tref());
}

torch::Tensor LindemannFalloffImpl::forward(
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

  // Lindemann: k = k0*[M]_eff / (1 + Pr) where Pr = k0*[M]_eff / kinf
  auto kinf_clamped = kinf.clamp(1e-100);
  auto Pr = k0 * M_eff / kinf_clamped;
  auto result = k0 * M_eff / (1.0 + Pr);

  return result;
}

}  // namespace kintera
