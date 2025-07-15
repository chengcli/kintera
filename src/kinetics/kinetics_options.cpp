// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include "kinetics.hpp"
#include "kinetics_formatter.hpp"

namespace kintera {

extern std::vector<std::string> species_names;
extern std::vector<double> species_weights;
extern bool species_initialized;

extern std::vector<double> species_cref_R;
extern std::vector<double> species_uref_R;
extern std::vector<double> species_sref_R;

KineticsOptions KineticsOptions::from_yaml(std::string const& filename) {
  if (!species_initialized) {
    init_species_from_yaml(filename);
  }

  KineticsOptions kinet;
  auto config = YAML::LoadFile(filename);

  if (config["reference-state"]) {
    if (config["reference-state"]["Tref"])
      kinet.Tref(config["reference-state"]["Tref"].as<double>());
    if (config["reference-state"]["Pref"])
      kinet.Pref(config["reference-state"]["Pref"].as<double>());
  }

  std::set<std::string> vapor_set;
  std::set<std::string> cloud_set;

  // register reactions
  if (!config["reactions"]) return kinet;

  // add arrhenius reactions
  kinet.arrhenius() = ArrheniusOptions::from_yaml(config["reactions"]);
  add_to_vapor_cloud(vapor_set, cloud_set, kinet.arrhenius());

  // add coagulation reactions
  kinet.coagulation() =
      ArrheniusOptions::from_yaml(config["reactions"], "coagulation");
  add_to_vapor_cloud(vapor_set, cloud_set, kinet.coagulation());

  // add evaporation reactions
  kinet.evaporation() = EvaporationOptions::from_yaml(config["reactions"]);
  add_to_vapor_cloud(vapor_set, cloud_set, kinet.evaporation());

  // register vapors
  for (const auto& sp : vapor_set) {
    auto it = std::find(species_names.begin(), species_names.end(), sp);
    int id = it - species_names.begin();
    kinet.vapor_ids().push_back(id);
  }

  // sort vapor ids
  std::sort(kinet.vapor_ids().begin(), kinet.vapor_ids().end());

  for (const auto& id : kinet.vapor_ids()) {
    kinet.cref_R().push_back(species_cref_R[id]);
    kinet.uref_R().push_back(species_uref_R[id]);
    kinet.sref_R().push_back(species_sref_R[id]);
  }

  // register clouds
  for (const auto& sp : cloud_set) {
    auto it = std::find(species_names.begin(), species_names.end(), sp);
    int id = it - species_names.begin();
    kinet.cloud_ids().push_back(id);
  }

  // sort cloud ids
  std::sort(kinet.cloud_ids().begin(), kinet.cloud_ids().end());

  for (const auto& id : kinet.cloud_ids()) {
    kinet.cref_R().push_back(species_cref_R[id]);
    kinet.uref_R().push_back(species_uref_R[id]);
    kinet.sref_R().push_back(species_sref_R[id]);
  }

  return kinet;
}

std::vector<Reaction> KineticsOptions::reactions() const {
  std::vector<Reaction> reactions;
  reactions.reserve(arrhenius().reactions().size() +
                    coagulation().reactions().size() +
                    evaporation().reactions().size());

  for (const auto& reaction : arrhenius().reactions()) {
    reactions.push_back(reaction);
  }
  for (const auto& reaction : coagulation().reactions()) {
    reactions.push_back(reaction);
  }
  for (const auto& reaction : evaporation().reactions()) {
    reactions.push_back(reaction);
  }

  return reactions;
}

}  // namespace kintera
