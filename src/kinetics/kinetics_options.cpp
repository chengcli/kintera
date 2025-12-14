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

KineticsOptions KineticsOptionsImpl::from_yaml(std::string const& filename,
                                               bool verbose) {
  auto config = YAML::LoadFile(filename);
  if (!config["reference-state"]) return nullptr;

  if (!species_initialized) {
    init_species_from_yaml(filename);
  }

  return KineticsOptionsImpl::from_yaml(config, verbose);
}

KineticsOptions KineticsOptionsImpl::from_yaml(YAML::Node const& config,
                                               bool verbose) {
  if (!config["reference-state"]) return nullptr;
  if (!species_initialized) {
    init_species_from_yaml(config);
  }

  auto kinet = KineticsOptionsImpl::create();
  kinet->verbose(verbose);

  if (config["reference-state"]["Tref"]) {
    kinet->Tref(config["reference-state"]["Tref"].as<double>());
    if (kinet->verbose()) {
      std::cout << fmt::format(
                       "[KineticsOptions] setting reference temperature Tref "
                       "= {} K",
                       kinet->Tref())
                << std::endl;
    }
  }

  if (config["reference-state"]["Pref"]) {
    kinet->Pref(config["reference-state"]["Pref"].as<double>());
    if (kinet->verbose()) {
      std::cout
          << fmt::format(
                 "[KineticsOptions] setting reference pressure Pref = {} Pa",
                 kinet->Pref())
          << std::endl;
    }
  }

  std::set<std::string> vapor_set;
  std::set<std::string> cloud_set;

  // register reactions
  if (!config["reactions"]) return kinet;

  // add arrhenius reactions
  kinet->arrhenius() = ArrheniusOptionsImpl::from_yaml(config["reactions"]);
  add_to_vapor_cloud(vapor_set, cloud_set, kinet->arrhenius());
  if (kinet->verbose()) {
    std::cout << fmt::format(
                     "[KineticsOptions] registered {} Arrhenius reactions",
                     kinet->arrhenius()->reactions().size())
              << std::endl;
  }

  // add coagulation reactions
  kinet->coagulation() = CoagulationOptionsImpl::from_yaml(config["reactions"]);
  add_to_vapor_cloud(vapor_set, cloud_set, kinet->coagulation());
  if (kinet->verbose()) {
    std::cout << fmt::format(
                     "[KineticsOptions] registered {} Coagulation reactions",
                     kinet->coagulation()->reactions().size())
              << std::endl;
  }

  // add evaporation reactions
  kinet->evaporation() = EvaporationOptionsImpl::from_yaml(config["reactions"]);
  add_to_vapor_cloud(vapor_set, cloud_set, kinet->evaporation());
  if (kinet->verbose()) {
    std::cout << fmt::format(
                     "[KineticsOptions] registered {} Evaporation reactions",
                     kinet->evaporation()->reactions().size())
              << std::endl;
  }

  // register vapors
  for (const auto& sp : vapor_set) {
    auto it = std::find(species_names.begin(), species_names.end(), sp);
    int id = it - species_names.begin();
    kinet->vapor_ids().push_back(id);
  }

  // sort vapor ids
  std::sort(kinet->vapor_ids().begin(), kinet->vapor_ids().end());
  if (kinet->verbose()) {
    std::cout << fmt::format("[KineticsOptions] registered vapor species: {}",
                             kinet->vapor_ids())
              << std::endl;
  }

  for (const auto& id : kinet->vapor_ids()) {
    kinet->cref_R().push_back(species_cref_R[id]);
    kinet->uref_R().push_back(species_uref_R[id]);
    kinet->sref_R().push_back(species_sref_R[id]);
  }

  // register clouds
  for (const auto& sp : cloud_set) {
    auto it = std::find(species_names.begin(), species_names.end(), sp);
    int id = it - species_names.begin();
    kinet->cloud_ids().push_back(id);
  }

  // sort cloud ids
  std::sort(kinet->cloud_ids().begin(), kinet->cloud_ids().end());
  if (kinet->verbose()) {
    std::cout << fmt::format("[KineticsOptions] registered cloud species: {}",
                             kinet->cloud_ids())
              << std::endl;
  }

  for (const auto& id : kinet->cloud_ids()) {
    kinet->cref_R().push_back(species_cref_R[id]);
    kinet->uref_R().push_back(species_uref_R[id]);
    kinet->sref_R().push_back(species_sref_R[id]);
  }

  return kinet;
}

std::vector<Reaction> KineticsOptionsImpl::reactions() const {
  std::vector<Reaction> reactions;
  reactions.reserve(arrhenius()->reactions().size() +
                    coagulation()->reactions().size() +
                    evaporation()->reactions().size());

  for (const auto& reaction : arrhenius()->reactions()) {
    reactions.push_back(reaction);
  }

  for (const auto& reaction : coagulation()->reactions()) {
    reactions.push_back(reaction);
  }

  for (const auto& reaction : evaporation()->reactions()) {
    reactions.push_back(reaction);
  }

  return reactions;
}

}  // namespace kintera
