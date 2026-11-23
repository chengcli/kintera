#pragma once

// C/C++
#include <algorithm>
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

// torch
#include <torch/torch.h>

// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/reaction.hpp>
#include <kintera/species.hpp>

namespace kintera {

// The parts of the third-body and falloff rate modules (ThreeBody,
// LindemannFalloff, TroeFalloff, SRIFalloff) that are the same in all four:
// the reader of one reaction node, the efficiency matrix, the Arrhenius
// limits, the effective third-body concentration and the efficiencies print.
// Each module keeps its own class, buffers and binding.

//! The reaction of a third-body or falloff node and the sum of its reactant
//! stoichiometric coefficients, M and (+M) excluded.
inline std::pair<Reaction, double> third_body_reaction(
    YAML::Node const& rxn_node) {
  TORCH_CHECK(rxn_node["equation"], "'equation' not defined");
  std::string equation = rxn_node["equation"].as<std::string>();
  Reaction reaction(equation);

  double sum_stoich = 0.;
  for (const auto& [name, coeff] : reaction.reactants()) {
    if (name != "M" && name != "(+M)") sum_stoich += coeff;
  }
  return {reaction, sum_stoich};
}

//! The per-species third-body efficiencies of a node, empty when absent.
inline Composition parse_efficiencies(YAML::Node const& rxn_node) {
  Composition eff;
  if (rxn_node["efficiencies"]) {
    for (auto const& eff_node : rxn_node["efficiencies"]) {
      eff[eff_node.first.as<std::string>()] = eff_node.second.as<double>();
    }
  }
  return eff;
}

//! The (nreaction, nspecies) efficiency matrix over the species registry,
//! 1 for every species a reaction does not list.
inline torch::Tensor efficiency_matrix_from(
    std::vector<Composition> const& efficiencies, int nreaction, int nspecies) {
  std::vector<double> eff_matrix_data(nreaction * nspecies, 1.0);

  for (int i = 0; i < nreaction; i++) {
    const auto& eff_map = efficiencies[i];
    for (int j = 0; j < nspecies; j++) {
      const std::string& sp_name = species_names[j];
      auto it = eff_map.find(sp_name);
      if (it != eff_map.end()) {
        eff_matrix_data[i * nspecies + j] = it->second;
      }
    }
  }

  return torch::tensor(eff_matrix_data, torch::kFloat64)
      .view({nreaction, nspecies});
}

//! Modified Arrhenius rate A (T/Tref)^b exp(-Ea_R/T), per reaction.
inline torch::Tensor arrhenius_rate(torch::Tensor const& A,
                                    torch::Tensor const& b,
                                    torch::Tensor const& Ea_R,
                                    torch::Tensor const& T, double Tref) {
  return A * (T / Tref).pow(b) * torch::exp(-Ea_R / T);
}

//! True when C carries KineticsImpl's expanded (..., nspecies, nreaction)
//! layout. KineticsImpl, the one caller that expands, says so with
//! other["conc_expanded"]; its species can be fewer than the registry, so no
//! shape test can tell its layout apart from a plain one. Without the flag
//! (a direct call) the concentration is taken as expanded only when it is at
//! least 3-D with both trailing sizes matching, nreaction and the registry
//! size, which a plain (..., nspecies) concentration cannot satisfy in 2-D
//! even when nspecies == nreaction.
inline bool expanded_layout(torch::Tensor const& C, int nreaction,
                            int nspecies_full,
                            std::map<std::string, torch::Tensor> const& other) {
  auto it = other.find("conc_expanded");
  if (it != other.end()) return it->second.item<bool>();
  return C.dim() >= 3 && C.size(-1) == nreaction && C.size(-2) == nspecies_full;
}

//! The registry ids of the species along the concentration's species axis,
//! other["conc_species"], when the caller gives them; an undefined tensor
//! otherwise.
inline torch::Tensor conc_species_from(
    std::map<std::string, torch::Tensor> const& other) {
  auto it = other.find("conc_species");
  return it == other.end() ? torch::Tensor() : it->second;
}

//! The effective third-body concentration [M]_eff of every reaction.
//! C is (..., nspecies), contracted with the efficiency matrix, or, when
//! `expanded`, (..., nspecies, nreaction): KineticsImpl expands the
//! concentration once per reaction so that a single backward pass over the
//! summed rate still yields d(k_r)/dC_s separately for each reaction. Every
//! slice along the reaction axis holds the same concentrations, so the
//! contraction runs over species and keeps the reaction axis; selecting one
//! slice away collapsed the whole sum_r d(k_r)/dC_s onto the first reaction
//! column and left every other reaction at exactly zero.
//! The efficiency matrix is laid out over the whole registry. When the
//! caller names the registry id of each concentration column
//! (`species_ids`, from other["conc_species"]), those columns are selected;
//! without the ids the concentration is taken to hold the first columns of
//! the registry, which is wrong as soon as a species no reaction uses is
//! listed before one that is.
inline torch::Tensor third_body_concentration(
    torch::Tensor const& C, torch::Tensor const& efficiency_matrix,
    bool expanded, torch::Tensor const& species_ids = torch::Tensor()) {
  int nspecies_full = efficiency_matrix.size(1);
  if (expanded) {
    int nspecies_kinetics = C.size(-2);
    if (species_ids.defined()) {
      TORCH_CHECK(species_ids.numel() == nspecies_kinetics,
                  "third_body_concentration: ", species_ids.numel(),
                  " species ids for ", nspecies_kinetics,
                  " concentration columns");
      auto eff_matrix_kinetics = efficiency_matrix.index_select(1, species_ids);
      return (C * eff_matrix_kinetics.transpose(0, 1)).sum(-2);
    }
    int ncontract = std::min(nspecies_kinetics, nspecies_full);
    auto eff_matrix_kinetics = efficiency_matrix.narrow(1, 0, ncontract);
    return (C.narrow(-2, 0, ncontract) * eff_matrix_kinetics.transpose(0, 1))
        .sum(-2);
  }
  int nspecies_kinetics = C.size(-1);
  auto eff_matrix_kinetics = efficiency_matrix.narrow(
      1, 0, std::min(nspecies_kinetics, nspecies_full));
  auto eff_T = eff_matrix_kinetics.transpose(0, 1);
  return torch::matmul(C, eff_T);
}

//! The (..., 0) rate of a module with no reactions.
inline torch::Tensor empty_rate(torch::Tensor const& T) {
  auto out_shape = T.sizes().vec();
  out_shape.push_back(0);
  return torch::empty(out_shape, T.options());
}

//! pretty_print's efficiencies line, when a reaction lists any.
inline void print_efficiencies(std::ostream& os, Composition const& eff) {
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

}  // namespace kintera
