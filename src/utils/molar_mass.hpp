#pragma once

#include <kintera/reaction.hpp>
#include <string>
#include <vector>

namespace kintera {

//! Atomic molar mass [kg/mol].
double atomic_mass(std::string const &element);

//! Compound molar mass [kg/mol].
double molar_mass(Composition const &composition);

//! Component molar masses [kg/mol] from an element-by-component matrix.
std::vector<double> molar_masses(
    std::vector<std::string> const &elements,
    std::vector<std::vector<double>> const &element_matrix);

//! Component molar masses [kg/mol] in phase order from a chemistry YAML file.
std::vector<double> molar_masses_from_yaml(std::string const &filename);

}  // namespace kintera
