// C/C++
#include <set>

// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

#include <kintera/kinetics/coagulation.hpp>
#include <kintera/kinetics/evaporation.hpp>

#include "thermo.hpp"

namespace kintera {

extern std::vector<std::string> species_names;
extern std::vector<double> species_weights;
extern bool species_initialized;

extern std::vector<double> species_cref_R;
extern std::vector<double> species_uref_R;
extern std::vector<double> species_sref_R;

ThermoOptions ThermoOptions::from_yaml(std::string const& filename) {
  if (!species_initialized) {
    init_species_from_yaml(filename);
  }

  auto config = YAML::LoadFile(filename);
  return ThermoOptions::from_yaml(config);
}

ThermoOptions ThermoOptions::from_yaml(YAML::Node const& config) {
  if (!species_initialized) {
    init_species_from_yaml(config);
  }

  ThermoOptions thermo;

  if (config["reference-state"]) {
    if (config["reference-state"]["Tref"])
      thermo.Tref(config["reference-state"]["Tref"].as<double>());
    if (config["reference-state"]["Pref"])
      thermo.Pref(config["reference-state"]["Pref"].as<double>());
  }

  if (config["dynamics"]) {
    if (config["dynamics"]["equation-of-state"]) {
      thermo.max_iter() =
          config["dynamics"]["equation-of-state"]["max-iter"].as<int>(10);
      thermo.ftol() =
          config["dynamics"]["equation-of-state"]["ftol"].as<double>(1e-6);
    }
  }

  std::set<std::string> vapor_set;
  std::set<std::string> cloud_set;

  // add reference species
  vapor_set.insert(species_names[0]);

  // register reactions
  if (config["reactions"]) {
    // add nucleation reactions
    thermo.nucleation() = NucleationOptions::from_yaml(config["reactions"]);
    add_to_vapor_cloud(vapor_set, cloud_set, thermo.nucleation());

    auto coagulation = CoagulationOptions::from_yaml(config["reactions"]);
    add_to_vapor_cloud(vapor_set, cloud_set, coagulation);

    auto evaporation = EvaporationOptions::from_yaml(config["reactions"]);
    add_to_vapor_cloud(vapor_set, cloud_set, evaporation);
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
    thermo.cref_R().push_back(species_cref_R[id]);
    thermo.uref_R().push_back(species_uref_R[id]);
    thermo.sref_R().push_back(species_sref_R[id]);
  }

  return thermo;
}

std::vector<Reaction> ThermoOptions::reactions() const {
  std::vector<Reaction> reactions;
  reactions.reserve(nucleation().reactions().size());

  for (const auto& reaction : nucleation().reactions()) {
    reactions.push_back(reaction);
  }

  return reactions;
}

}  // namespace kintera
