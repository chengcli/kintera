#pragma once

// C/C++
#include <map>
#include <string>

// arg
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

  //! Reaction orders
  ADD_ARG(Composition, orders);

  ADD_ARG(bool, reversible) = false;
};

std::string to_string(Composition const& p);

bool operator==(Reaction const& lhs, Reaction const& rhs);
bool operator<(Reaction const& lhs, Reaction const& rhs);

}  // namespace kintera

#undef ADD_ARG
