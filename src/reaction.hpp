#pragma once

// C/C++
#include <map>
#include <string>

// base
#include "add_arg.h"

namespace kintera {

using Composition = std::map<std::string, double>;

struct Reaction {
  Reaction() = default;
  explicit Reaction(const std::string& equation);

  //! The chemical equation for this reaction
  std::string equation() const;

  //! Reactant species and stoichiometric coefficients
  ADD_ARG(Composition, reactants);

  //! Product species and stoichiometric coefficients
  ADD_ARG(Composition, products);

  ADD_ARG(bool, reversible) = false;
};

std::string to_string(Composition const& p);

}  // namespace kintera
