// C/C++
#include <string>
#include <vector>

// yaml
#include <yaml-cpp/yaml.h>

// harp
#include <harp/compound.hpp>

// kintera
#include "species.hpp"

namespace kintera {

std::vector<std::string> species_names;
std::vector<double> species_weights;
std::vector<double>, cref_R;
std::vector<double>, uref_R;
std::vector<double>, sref_R;
bool species_initialized = false;

void init_species_from_yaml(std::string filename) {
  auto config = YAML::LoadFile(filename);

  // check if species are defined
  TORCH_CHECK(
      config["species"],
      "'species' is not defined in the thermodynamics configuration file");

  species_names.clear();
  species_weights.clear();
  cref_R.clear();
  uref_R.clear();
  sref_R.clear();

  for (const auto& sp : config["species"]) {
    species_names.push_back(sp["name"].as<std::string>());
    std::map<std::string, double> comp;

    for (const auto& it : sp["composition"]) {
      std::string key = it.first.as<std::string>();
      double value = it.second.as<double>();
      comp[key] = value;
    }
    species_weights.push_back(harp::get_compound_weight(comp));

    if (sp["cv_R"]) {
      cref_R.push_back(sp["cv_R"].as<double>());
    } else {
      cref_R.push_back(5. / 2.);
    }

    if (sp["u0_R"]) {
      uref_R.push_back(sp["u0_R"].as<double>());
    } else {
      uref_R.push_back(0.);
    }

    if (sp["s0_R"]) {
      sref_R.push_back(sp["u0_R"].as<double>());
    } else {
      sref_R.push_back(0.);
    }
  }

  species_initialized = true;
}

std::vector<string> SpeciesThermo::species() const {
  TORCH_CHECK(species_initialized,
              "Species must be initialized before calling species().",
              "Please call init_species_from_yaml() first.");

  std::vector<string> species_list;

  // add vapors
  for (int i = 0; i < vapor_ids().size(); ++i) {
    species_list.push_back(species_names[vapor_ids()[i]]);
  }

  // add clouds
  for (int i = 0; i < cloud_ids().size(); ++i) {
    species_list.push_back(species_names[cloud_ids()[i]]);
  }

  // add the reference species
  if (species_list.find(species_names[0]) == species_list.end()) {
    species_list.insert(species_list.begin(), species_names[0]);
  }

  return species_list;
}

}  // namespace kintera
