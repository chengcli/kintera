// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/arrhenius.hpp>
#include <kintera/evaporation.hpp>

namespace kintera {

extern std::vector<std::string> species_names;
extern std::vector<double> species_weights;
extern bool species_initialized;

KineticRateOptions KineticRateOptions::from_yaml(std::string const& filename) {
  TORCH_CHECK(
      species_initialized,
      "Species must be initialized before loading kinetic rate options.",
      "Please call init_species_from_yaml() first.");

  KineticRateOptions kinet;
  auto config = YAML::LoadFile(filename);

  if (config["reference-state"]) {
    if (config["reference-state"]["Tref"])
      kinet.Tref(config["reference-state"]["Tref"].as<double>());
    if (config["reference-state"]["Pref"])
      kinet.Pref(config["reference-state"]["Pref"].as<double>());
  }

  kinet.cref_R().push_back(cref_R[0]);
  kinet.uref_R().push_back(uref_R[0]);
  kinet.sref_R().push_back(sref_R[0]);

  // register reactions
  TORCH_CHECK(config["reactions"],
              "'reactions' is not defined in the configuration file");

  std::set<std::string> vapor_set;
  std::set<std::string> cloud_set;

  // add arrhenius reactions
  kinet.arrhenius() = ArrheniusOptions::from_yaml(config["reactions"]);
  add_to_vapor_cloud(vapor_set, cloud_set, kinet.arrhenius());

  // add coagulation reactions
  kinet.coagulation() = ArrheniusOptions::from_yaml(config["reactions"]);
  add_to_vapor_cloud(vapor_set, cloud_set, kinet.coagulation());

  // add evaporation reactions
  kinet.evaporation() = EvaporationOptions::from_yaml(config["reactions"]);
  add_to_vapor_cloud(vapor_set, cloud_set, kinet.evaporation());

  // register vapors
  for (const auto& sp : vapor_set) {
    auto it = species_names.find(sp.as<std::string>());
    int id = it - species_names.begin();
    kinet.vapor_ids().push_back(id);
  }

  // sort vapor ids
  std::sort(kinet.vapor_ids.begin(), kinet.vapor_ids.end());

  for (const auto& id : kinet.vapor_ids()) {
    kinet.cref_R().push_back(cref_R[id]);
    kinet.uref_R().push_back(uref_R[id]);
    kinet.sref_R().push_back(sref_R[id]);
  }

  // register clouds
  for (const auto& sp : cloud_set) {
    auto it = species_names.find(sp.as<std::string>());
    int id = it - species_names.begin();
    kinet.cloud_ids().push_back(id);
  }

  // sort cloud ids
  std::sort(kinet.cloud_ids.begin(), kinet.cloud_ids.end());

  for (const auto& id : kinet.cloud_ids) {
    kinet.cref_R().push_back(cref_R[id]);
    kinet.uref_R().push_back(uref_R[id]);
    kinet.sref_R().push_back(sref_R[id]);
  }

  return kinet;
}

}  // namespace kintera
