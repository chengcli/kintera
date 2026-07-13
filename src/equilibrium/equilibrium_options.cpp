#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <kintera/reaction.hpp>
#include <kintera/utils/find_resource.hpp>
#include <map>
#include <set>

#include "equilibrium.hpp"

namespace kintera {

EquilibriumOptions EquilibriumOptionsImpl::from_yaml(
    std::string const& filename, bool verbose) {
  auto config = YAML::LoadFile(find_resource(filename));
  TORCH_CHECK(config["phases"], "equilibrium YAML requires 'phases'");
  TORCH_CHECK(config["species"], "equilibrium YAML requires 'species'");
  TORCH_CHECK(config["reactions"], "equilibrium YAML requires 'reactions'");

  auto options = EquilibriumOptionsImpl::create();
  std::set<std::string> seen_components;
  int gas_phase_count = 0;

  for (auto const& phase_node : config["phases"]) {
    TORCH_CHECK(phase_node["name"], "phase is missing 'name'");
    TORCH_CHECK(phase_node["species"], "phase is missing 'species'");
    int phase_id = static_cast<int>(options->phases().size());
    options->phases().push_back(phase_node["name"].as<std::string>());

    std::string model;
    if (phase_node["thermo"]) {
      model = phase_node["thermo"].as<std::string>();
    } else if (phase_node["type"]) {
      model = phase_node["type"].as<std::string>();
    }
    if (model == "ideal-gas") {
      options->gas_phase(phase_id);
      ++gas_phase_count;
    }

    for (auto const& component_node : phase_node["species"]) {
      auto component = component_node.as<std::string>();
      TORCH_CHECK(seen_components.insert(component).second,
                  "component appears in more than one phase: ", component);
      options->components().push_back(component);
      options->phase_ids().push_back(phase_id);
    }
  }
  TORCH_CHECK(gas_phase_count == 1,
              "equilibrium YAML requires exactly one ideal-gas phase");

  std::map<std::string, std::map<std::string, double>> compositions;
  std::set<std::string> element_set;
  for (auto const& species_node : config["species"]) {
    TORCH_CHECK(species_node["name"], "species is missing 'name'");
    TORCH_CHECK(species_node["composition"],
                "species is missing 'composition'");
    auto name = species_node["name"].as<std::string>();
    TORCH_CHECK(compositions.find(name) == compositions.end(),
                "duplicate species definition: ", name);
    for (auto const& entry : species_node["composition"]) {
      auto element = entry.first.as<std::string>();
      compositions[name][element] = entry.second.as<double>();
      element_set.insert(element);
    }
  }
  for (auto const& component : options->components()) {
    TORCH_CHECK(compositions.find(component) != compositions.end(),
                "phase component has no species definition: ", component);
  }

  std::vector<Reaction> reactions;
  for (auto const& reaction_node : config["reactions"]) {
    if (reaction_node["type"] &&
        reaction_node["type"].as<std::string>() != "equilibrium") {
      continue;
    }
    TORCH_CHECK(reaction_node["equation"], "reaction is missing 'equation'");
    auto equation = reaction_node["equation"].as<std::string>();
    options->reactions().push_back(equation);
    reactions.emplace_back(equation);
  }
  TORCH_CHECK(!reactions.empty(), "equilibrium YAML contains no reactions");

  for (size_t j = 0; j < reactions.size(); ++j) {
    for (auto const& [name, coefficient] : reactions[j].reactants()) {
      auto found = std::find(options->components().begin(),
                             options->components().end(), name);
      TORCH_CHECK(found != options->components().end(),
                  "reaction references unknown component: ", name);
      (void)coefficient;
    }
    for (auto const& [name, coefficient] : reactions[j].products()) {
      auto found = std::find(options->components().begin(),
                             options->components().end(), name);
      TORCH_CHECK(found != options->components().end(),
                  "reaction references unknown component: ", name);
      (void)coefficient;
    }
    for (auto const& element : element_set) {
      double balance = 0.;
      for (auto const& [name, coefficient] : reactions[j].reactants()) {
        auto found = compositions.at(name).find(element);
        if (found != compositions.at(name).end())
          balance -= coefficient * found->second;
      }
      for (auto const& [name, coefficient] : reactions[j].products()) {
        auto found = compositions.at(name).find(element);
        if (found != compositions.at(name).end())
          balance += coefficient * found->second;
      }
      TORCH_CHECK(std::abs(balance) <= 1.e-10, "reaction ", j,
                  " does not conserve element ", element);
    }
  }

  if (config["equilibrium"]) {
    auto equilibrium = config["equilibrium"];
    options->standard_pressure(
        equilibrium["standard-pressure"].as<double>(1.e5));
    options->max_iter(equilibrium["max-iter"].as<int>(50));
    options->ftol(equilibrium["ftol"].as<double>(1.e-8));
    options->mole_floor(equilibrium["mole-floor"].as<double>(1.e-30));
  }

  options->validate();
  if (verbose) options->report(std::cout);
  return options;
}

}  // namespace kintera
