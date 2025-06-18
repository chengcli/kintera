// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include "kinetic_rate.hpp"

namespace kintera {

extern std::vector<std::string> species_names;
extern std::vector<double> species_weights;

KineticRateOptions KineticRateOptions::from_yaml(const std::string& filename) {
  KineticRateOptions kinetics;
  auto config = YAML::LoadFile(filename);

  // check if species are defined
  TORCH_CHECK(
      config["species"],
      "'species' is not defined in the thermodynamics configuration file");

  if (config["reference-state"]) {
    if (config["reference-state"]["Tref"])
      kinetics.Tref(config["reference-state"]["Tref"].as<double>());
    if (config["reference-state"]["Pref"])
      kinetics.Pref(config["reference-state"]["Pref"].as<double>());
  }

  for (const auto& sp : config["species"]) {
    if (species_names.find(sp["name"].as<std::string>()) ==
        species_names.end()) {
      species_names.push_back(sp["name"].as<std::string>());
      std::map<std::string, double> comp;

      for (const auto& it : sp["composition"]) {
        std::string key = it.first.as<std::string>();
        double value = it.second.as<double>();
        comp[key] = value;
      }
      species_weights.push_back(harp::get_compound_weight(comp));
    }
  }

  if (node["species"]) {
    options.species() = node["species"].as<std::vector<std::string>>();
  }
  if (node["reactions"]) {
    options.reactions() = node["reactions"].as<std::vector<Reaction>>();
  }
  if (node["arrhenius"]) {
    options.arrhenius() = node["arrhenius"].as<ArrheniusOptions>();
  }
  if (node["evaporation"]) {
    options.evaporation() = node["evaporation"].as<EvaporationOptions>();
  }

  return options;
}

}  // namespace kintera
