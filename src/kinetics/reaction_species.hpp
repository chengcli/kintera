#pragma once

// C/C++
#include <algorithm>
#include <set>
#include <string>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/species.hpp>

namespace kintera {

//! Insert every reactant and product name of `op`'s reactions (and, with
//! `with_efficiencies`, every efficiency name) into `into`, checking each
//! against the species registry; `skip_third_body` ignores "M" and "(+M)".
//! The body of add_to_vapor_cloud for every rate family but evaporation.
template <typename Options>
void add_reaction_species(std::set<std::string>& into, Options const& op,
                          bool skip_third_body, bool with_efficiencies) {
  auto add = [&](std::string const& name) {
    if (skip_third_body && (name == "M" || name == "(+M)")) return;
    auto it = std::find(species_names.begin(), species_names.end(), name);
    TORCH_CHECK(it != species_names.end(), "Species ", name,
                " not found in species list");
    into.insert(name);
  };
  for (auto& react : op->reactions()) {
    for (auto& [name, _] : react.reactants()) add(name);
    for (auto& [name, _] : react.products()) add(name);
    if (with_efficiencies)
      for (auto& [name, _] : react.efficiencies()) add(name);
  }
}

}  // namespace kintera
