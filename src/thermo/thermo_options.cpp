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

ThermoOptions ThermoOptionsImpl::from_yaml(std::string const& filename,
                                           bool verbose) {
  auto config = YAML::LoadFile(filename);
  if (!config["reference-state"]) return nullptr;

  if (!species_initialized) {
    init_species_from_yaml(filename);
  }

  return ThermoOptionsImpl::from_yaml(config, verbose);
}

ThermoOptions ThermoOptionsImpl::from_yaml(YAML::Node const& config,
                                           bool verbose) {
  if (!config["reference-state"]) return nullptr;
  if (!species_initialized) {
    init_species_from_yaml(config);
  }

  auto thermo = ThermoOptionsImpl::create();
  thermo->verbose(verbose);

  if (config["reference-state"]["Tref"]) {
    thermo->Tref(config["reference-state"]["Tref"].as<double>());
    if (thermo->verbose()) {
      std::cout << "[ThermoOptions] setting reference temperature Tref = "
                << thermo->Tref() << " K" << std::endl;
    }
  }

  if (config["reference-state"]["Pref"]) {
    thermo->Pref(config["reference-state"]["Pref"].as<double>());

    if (thermo->verbose()) {
      std::cout << "[ThermoOptions] setting reference pressure Pref = "
                << thermo->Pref() << " Pa" << std::endl;
    }
  }

  if (!species_initialized) {
    init_species_from_yaml(config);
  }

  if (config["dynamics"]) {
    if (config["dynamics"]["equation-of-state"]) {
      thermo->max_iter() =
          config["dynamics"]["equation-of-state"]["max-iter"].as<int>(10);
      if (thermo->verbose()) {
        std::cout << "[ThermoOptions] setting EOS max-iter = "
                  << thermo->max_iter() << std::endl;
      }

      thermo->ftol() =
          config["dynamics"]["equation-of-state"]["ftol"].as<double>(1e-6);
      if (thermo->verbose()) {
        std::cout << "[ThermoOptions] setting EOS ftol = " << thermo->ftol()
                  << std::endl;
      }
    }
  }

  std::set<std::string> vapor_set;
  std::set<std::string> cloud_set;

  // add reference species
  vapor_set.insert(species_names[0]);

  // register reactions
  if (config["reactions"]) {
    // add nucleation reactions
    thermo->nucleation() =
        NucleationOptionsImpl::from_yaml(config["reactions"]);
    add_to_vapor_cloud(vapor_set, cloud_set, thermo->nucleation());
    if (thermo->verbose()) {
      std::cout << fmt::format(
                       "[ThermoOptions] registered {} Nucleation reactions",
                       thermo->nucleation()->reactions().size())
                << std::endl;
    }

    // create temporary coagulation and evaporation options to add species
    auto coagulation = CoagulationOptionsImpl::from_yaml(config["reactions"]);
    add_to_vapor_cloud(vapor_set, cloud_set, coagulation);
    if (thermo->verbose()) {
      std::cout << fmt::format(
                       "[ThermoOptions] registered {} Coagulation reactions",
                       coagulation->reactions().size())
                << std::endl;
    }

    auto evaporation = EvaporationOptionsImpl::from_yaml(config["reactions"]);
    add_to_vapor_cloud(vapor_set, cloud_set, evaporation);
    if (thermo->verbose()) {
      std::cout << fmt::format(
                       "[ThermoOptions] registered {} Evaporation reactions",
                       evaporation->reactions().size())
                << std::endl;
    }
  }

  // register vapors
  for (const auto& sp : vapor_set) {
    auto it = std::find(species_names.begin(), species_names.end(), sp);
    int id = it - species_names.begin();
    thermo->vapor_ids().push_back(id);
  }

  // sort vapor ids
  std::sort(thermo->vapor_ids().begin(), thermo->vapor_ids().end());
  if (thermo->verbose()) {
    std::cout << fmt::format("[ThermoOptions] registered vapor species: {}",
                             thermo->vapor_ids())
              << std::endl;
  }

  for (const auto& id : thermo->vapor_ids()) {
    thermo->cref_R().push_back(species_cref_R[id]);
    thermo->uref_R().push_back(species_uref_R[id]);
    thermo->sref_R().push_back(species_sref_R[id]);
  }

  // register clouds
  for (const auto& sp : cloud_set) {
    auto it = std::find(species_names.begin(), species_names.end(), sp);
    int id = it - species_names.begin();
    thermo->cloud_ids().push_back(id);
  }

  // sort cloud ids
  std::sort(thermo->cloud_ids().begin(), thermo->cloud_ids().end());
  if (thermo->verbose()) {
    std::cout << fmt::format("[ThermoOptions] registered cloud species: {}",
                             thermo->cloud_ids())
              << std::endl;
  }

  for (const auto& id : thermo->cloud_ids()) {
    thermo->cref_R().push_back(species_cref_R[id]);
    thermo->uref_R().push_back(species_uref_R[id]);
    thermo->sref_R().push_back(species_sref_R[id]);
  }

  return thermo;
}

std::vector<Reaction> ThermoOptionsImpl::reactions() const {
  std::vector<Reaction> reactions;
  reactions.reserve(nucleation()->reactions().size());

  for (const auto& reaction : nucleation()->reactions()) {
    reactions.push_back(reaction);
  }

  return reactions;
}

}  // namespace kintera
