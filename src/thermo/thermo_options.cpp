// C/C++
#include <set>

// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

#include "thermo.hpp"

namespace kintera {

extern std::vector<std::string> species_names;
extern std::vector<double> species_weights;
extern bool species_initialized;

extern std::vector<double> species_cref_R;
extern std::vector<double> species_uref_R;
extern std::vector<double> species_sref_R;

ThermoOptions ThermoOptions::from_yaml(std::string const& filename) {
  TORCH_CHECK(
      species_initialized,
      "Species must be initialized before loading thermodynamics options.",
      "Please call init_species_from_yaml() first.");

  ThermoOptions thermo;
  auto config = YAML::LoadFile(filename);

  if (config["reference-state"]) {
    if (config["reference-state"]["Tref"])
      thermo.Tref(config["reference-state"]["Tref"].as<double>());
    if (config["reference-state"]["Pref"])
      thermo.Pref(config["reference-state"]["Pref"].as<double>());
  }

  std::set<std::string> vapor_set;
  std::set<std::string> cloud_set;

  thermo.Rd(constants::Rgas / species_weights[0]);

  // add reference species
  vapor_set.insert(species_names[0]);

  // register reactions
  TORCH_CHECK(config["reactions"],
              "'reactions' is not defined in the configuration file");

  for (auto const& node : config["reactions"]) {
    if (!node["type"] || (node["type"].as<std::string>() != "nucleation")) {
      continue;
    }
    thermo.react().push_back(Nucleation::from_yaml(node));

    // go through reactants
    for (auto& [name, _] : thermo.react().back().reaction().reactants()) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name,
                  " not found in species list");
      vapor_set.insert(name);
    }

    // go through products
    for (auto& [name, _] : thermo.react().back().reaction().products()) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name,
                  " not found in species list");
      cloud_set.insert(name);
    }
  }

  // register vapors
  for (const auto& sp : vapor_set) {
    auto it = std::find(species_names.begin(), species_names.end(), sp);
    int id = it - species_names.begin();
    thermo.vapor_ids().push_back(id);
  }

  // sort vapor ids
  std::sort(thermo.vapor_ids().begin(), thermo.vapor_ids().end());

  for (const auto& id : thermo.vapor_ids()) {
    thermo.mu_ratio().push_back(species_weights[id] / species_weights[0]);
    thermo.cref_R().push_back(species_cref_R[id]);
    thermo.uref_R().push_back(species_uref_R[id]);
    thermo.sref_R().push_back(species_sref_R[id]);
  }

  // register clouds
  for (const auto& sp : cloud_set) {
    auto it = std::find(species_names.begin(), species_names.end(), sp);
    int id = it - species_names.begin();
    thermo.cloud_ids().push_back(id);
  }

  // sort cloud ids
  std::sort(thermo.cloud_ids().begin(), thermo.cloud_ids().end());

  for (const auto& id : thermo.cloud_ids()) {
    thermo.mu_ratio().push_back(species_weights[id] / species_weights[0]);
    thermo.cref_R().push_back(species_cref_R[id]);
    thermo.uref_R().push_back(species_uref_R[id]);
    thermo.sref_R().push_back(species_sref_R[id]);
  }

  return thermo;
}

}  // namespace kintera
