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

//! The effective third-body concentration [M]_eff of every reaction.
//! C is (..., nspecies), contracted with the efficiency matrix, or, when
//! `expanded`, (..., nspecies, nreaction): KineticsImpl expands the
//! concentration once per reaction so that a single backward pass over the
//! summed rate still yields d(k_r)/dC_s separately for each reaction. Every
//! slice along the reaction axis holds the same concentrations, so the
//! contraction runs over species and keeps the reaction axis; selecting one
//! slice away collapsed the whole sum_r d(k_r)/dC_s onto the first reaction
//! column and left every other reaction at exactly zero.
inline torch::Tensor third_body_concentration(
    torch::Tensor const& C, torch::Tensor const& efficiency_matrix,
    bool expanded) {
  int nspecies_full = efficiency_matrix.size(1);
  if (expanded) {
    int nspecies_kinetics = C.size(-2);
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
