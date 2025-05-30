// yaml
#include <yaml-cpp/yaml.h>

// harp
#include <harp/compound.hpp>

// kintera
#include "thermo.hpp"

namespace kintera {

std::vector<std::string> species_names;
std::vector<double> species_weights;

ThermoOptions ThermoOptions::from_yaml(std::string const& filename) {
  ThermoOptions thermo;
  auto config = YAML::LoadFile(filename);

  // check if species are defined
  TORCH_CHECK(
      config["species"],
      "'species' is not defined in the thermodynamics configuration file");

  species_names.clear();
  species_weights.clear();

  if (config["reference-state"]) {
    if (config["reference-state"]["Tref"])
      thermo.Tref(config["reference-state"]["Tref"].as<double>());
    if (config["reference-state"]["Pref"])
      thermo.Pref(config["reference-state"]["Pref"].as<double>());
  }

  std::vector<double> cp_R, cv_R, u0_R;

  for (const auto& sp : config["species"]) {
    species_names.push_back(sp["name"].as<std::string>());
    std::map<std::string, double> comp;

    for (const auto& it : sp["composition"]) {
      std::string key = it.first.as<std::string>();
      double value = it.second.as<double>();
      comp[key] = value;
    }
    species_weights.push_back(harp::get_compound_weight(comp));

    if (sp["cp_R"]) {
      cp_R.push_back(sp["cp_R"].as<double>());
    } else {
      cp_R.push_back(7. / 2.);
    }

    if (sp["cv_R"]) {
      cv_R.push_back(sp["cv_R"].as<double>());
    } else {
      cv_R.push_back(5. / 2.);
    }

    if (sp["u0_R"]) {
      u0_R.push_back(sp["u0_R"].as<double>());
    } else {
      u0_R.push_back(0.);
    }
  }

  thermo.gammad(cp_R[0] / cv_R[0]);
  thermo.Rd(constants::Rgas / species_weights[0]);

  thermo.mu_ratio().clear();
  thermo.cv_R().clear();
  thermo.cp_R().clear();
  thermo.u0_R().clear();

  thermo.species().clear();
  thermo.species().push_back(species_names[0]);

  // register vapors
  if (config["vapor"]) {
    for (const auto& sp : config["vapor"]) {
      auto it = std::find(species_names.begin(), species_names.end(),
                          sp.as<std::string>());
      TORCH_CHECK(it != species_names.end(), "vapor species ",
                  sp.as<std::string>(), " not found in species list");
      thermo.vapor_ids().push_back(it - species_names.begin());
      thermo.species().push_back(sp.as<std::string>());
    }
  }

  for (int i = 0; i < thermo.vapor_ids().size(); ++i) {
    auto id = thermo.vapor_ids()[i];
    thermo.mu_ratio().push_back(species_weights[id] / species_weights[0]);
    thermo.cp_R().push_back(cp_R[id]);
    thermo.cv_R().push_back(cv_R[id]);
    thermo.u0_R().push_back(u0_R[id]);
  }

  // register clouds
  if (config["cloud"]) {
    for (const auto& sp : config["cloud"]) {
      auto it = std::find(species_names.begin(), species_names.end(),
                          sp.as<std::string>());
      TORCH_CHECK(it != species_names.end(), "cloud species ",
                  sp.as<std::string>(), " not found in species list");
      thermo.cloud_ids().push_back(it - species_names.begin());
      thermo.species().push_back(sp.as<std::string>());
    }
  }

  for (int i = 0; i < thermo.cloud_ids().size(); ++i) {
    auto id = thermo.cloud_ids()[i];
    thermo.mu_ratio().push_back(species_weights[id] / species_weights[0]);
    thermo.cp_R().push_back(cp_R[id]);
    thermo.cv_R().push_back(cv_R[id]);
    thermo.u0_R().push_back(u0_R[id]);
  }

  // register reactions
  TORCH_CHECK(config["reactions"],
              "'reactions' is not defined in the configuration file");

  for (auto const& node : config["reactions"]) {
    if (!node["type"] || (node["type"].as<std::string>() != "nucleation")) {
      continue;
    }
    thermo.react().push_back(Nucleation::from_yaml(node));
  }

  return thermo;
}

}  // namespace kintera
