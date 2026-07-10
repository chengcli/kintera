#include <harp/compound.hpp>
#include <harp/element.hpp>

#include <yaml-cpp/yaml.h>

#include <torch/torch.h>

#include <map>

#include "molar_mass.hpp"

#include <kintera/utils/find_resource.hpp>

namespace kintera {

double atomic_mass(std::string const &element) {
  return harp::get_element_weight(element) * 1.e-3;
}

double molar_mass(Composition const &composition) {
  return harp::get_compound_weight(composition);
}

std::vector<double>
molar_masses(std::vector<std::string> const &elements,
             std::vector<std::vector<double>> const &element_matrix) {
  TORCH_CHECK(elements.size() == element_matrix.size(),
              "elements and element_matrix row counts must match");
  if (elements.empty())
    return {};

  auto ncomponent = element_matrix[0].size();
  for (auto const &row : element_matrix) {
    TORCH_CHECK(row.size() == ncomponent,
                "element_matrix rows must have equal length");
  }

  std::vector<double> result(ncomponent, 0.);
  for (size_t i = 0; i < ncomponent; ++i) {
    Composition composition;
    for (size_t e = 0; e < elements.size(); ++e) {
      if (element_matrix[e][i] != 0.) {
        composition[elements[e]] = element_matrix[e][i];
      }
    }
    result[i] = molar_mass(composition);
    TORCH_CHECK(result[i] > 0., "component ", i, " has no positive molar mass");
  }
  return result;
}

std::vector<double> molar_masses_from_yaml(std::string const &filename) {
  auto config = YAML::LoadFile(find_resource(filename));
  TORCH_CHECK(config["phases"], "chemistry YAML requires 'phases'");
  TORCH_CHECK(config["species"], "chemistry YAML requires 'species'");

  std::map<std::string, Composition> compositions;
  for (auto const &species_node : config["species"]) {
    TORCH_CHECK(species_node["name"], "species is missing 'name'");
    TORCH_CHECK(species_node["composition"],
                "species is missing 'composition'");
    auto name = species_node["name"].as<std::string>();
    TORCH_CHECK(compositions.emplace(name, Composition{}).second,
                "duplicate species definition: ", name);
    for (auto const &entry : species_node["composition"]) {
      compositions.at(name)[entry.first.as<std::string>()] =
          entry.second.as<double>();
    }
  }

  std::vector<double> result;
  for (auto const &phase_node : config["phases"]) {
    TORCH_CHECK(phase_node["species"], "phase is missing 'species'");
    for (auto const &component_node : phase_node["species"]) {
      auto component = component_node.as<std::string>();
      auto found = compositions.find(component);
      TORCH_CHECK(found != compositions.end(),
                  "phase component has no species definition: ", component);
      result.push_back(molar_mass(found->second));
    }
  }
  return result;
}

} // namespace kintera
