// C/C++
#include <algorithm>
#include <string>
#include <unordered_set>
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
std::vector<double> species_cref_R;
std::vector<double> species_uref_R;
std::vector<double> species_sref_R;
bool species_initialized = false;

template <typename T>
std::vector<size_t> find_common_(std::vector<T> const& a,
                                 std::vector<T> const& b) {
  std::unordered_set<T> a_set(a.begin(), a.end());
  std::vector<size_t> indices;

  for (size_t i = 0; i < b.size(); ++i) {
    if (a_set.count(b[i])) {
      indices.push_back(i);
    }
  }
  return indices;
}

void init_species_from_yaml(std::string filename) {
  auto config = YAML::LoadFile(filename);

  // check if species are defined
  if (!config["species"]) {
    throw std::runtime_error(
        "'species' is not defined in the kintera configuration file");
  }

  species_names.clear();
  species_weights.clear();
  species_cref_R.clear();
  species_uref_R.clear();
  species_sref_R.clear();

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
      species_cref_R.push_back(sp["cv_R"].as<double>());
    } else {
      species_cref_R.push_back(5. / 2.);
    }

    if (sp["u0_R"]) {
      species_uref_R.push_back(sp["u0_R"].as<double>());
    } else {
      species_uref_R.push_back(0.);
    }

    if (sp["s0_R"]) {
      species_sref_R.push_back(sp["u0_R"].as<double>());
    } else {
      species_sref_R.push_back(0.);
    }
  }

  species_initialized = true;
}

std::vector<std::string> SpeciesThermo::species() const {
  std::vector<std::string> species_list;

  // add vapors
  for (int i = 0; i < vapor_ids().size(); ++i) {
    species_list.push_back(species_names[vapor_ids()[i]]);
  }

  // add clouds
  for (int i = 0; i < cloud_ids().size(); ++i) {
    species_list.push_back(species_names[cloud_ids()[i]]);
  }

  return species_list;
}

std::vector<std::string> SpeciesThermo::copy_from(
    torch::Tensor tensor, SpeciesThermo const& other) const {
  // find my vapor ids in other
  for (int i = 0; i < vapor_ids().size(); ++i) {
    int id = vapor_ids()[i];
    int jd = other.vapor_ids().index(id);
  }
}

template <typename T>
std::vector<T> merge_vectors(std::vector<T> const& vec1,
                             std::vector<T> const& vec2) {
  std::vector<T> merged = vec1;
  merged.insert(merged.end(), vec2.begin(), vec2.end());
  return merged;
}

template <typename T>
std::vector<T> sort_vectors(std::vector<T> const& vec,
                            std::vector<int> const& indices) {
  std::vector<T> sorted(vec.size());
  std::transform(indices.begin(), indices.end(), sorted.begin(),
                 [&vec](int index) { return vec[index]; });
  return sorted;
}

SpeciesThermo merge_thermo(SpeciesThermo const& thermo1,
                           SpeciesThermo const& thermo2) {
  // return a new SpeciesThermo object with merged data
  SpeciesThermo merged;

  auto& vapor_ids = merged.vapor_ids();
  auto& cloud_ids = merged.cloud_ids();
  auto& cref_R = merged.cref_R();
  auto& uref_R = merged.uref_R();
  auto& sref_R = merged.sref_R();
  auto& intEng_R_extra = merged.intEng_R_extra();
  auto& cv_R_extra = merged.cv_R_extra();
  auto& cp_R_extra = merged.cp_R_extra();
  auto& entropy_R_extra = merged.entropy_R_extra();
  auto& czh = merged.czh();
  auto& czh_ddC = merged.czh_ddC();

  // concatenate fields
  vapor_ids = merge_vectors(thermo1.vapor_ids(), thermo2.vapor_ids());
  cloud_ids = merge_vectors(thermo1.cloud_ids(), thermo2.cloud_ids());

  cref_R = merge_vectors(thermo1.cref_R(), thermo2.cref_R());
  uref_R = merge_vectors(thermo1.uref_R(), thermo2.uref_R());
  sref_R = merge_vectors(thermo1.sref_R(), thermo2.sref_R());

  intEng_R_extra =
      merge_vectors(thermo1.intEng_R_extra(), thermo2.intEng_R_extra());
  cv_R_extra = merge_vectors(thermo1.cv_R_extra(), thermo2.cv_R_extra());
  cp_R_extra = merge_vectors(thermo1.cp_R_extra(), thermo2.cp_R_extra());
  entropy_R_extra =
      merge_vectors(thermo1.entropy_R_extra(), thermo2.entropy_R_extra());

  czh = merge_vectors(thermo1.czh(), thermo2.czh());
  czh_ddC = merge_vectors(thermo1.czh_ddC(), thermo2.czh_ddC());

  // identify duplicated vapor ids and remove them from all vectors
  int first = 0;
  std::set<int> seen_vapor_ids;

  while (first < vapor_ids.size()) {
    int vapor_id = vapor_ids[first];
    if (seen_vapor_ids.find(vapor_id) != seen_vapor_ids.end()) {
      // duplicate found, remove it from all vectors
      vapor_ids.erase(vapor_ids.begin() + first);
      cref_R.erase(cref_R.begin() + first);
      uref_R.erase(uref_R.begin() + first);
      sref_R.erase(sref_R.begin() + first);
      intEng_R_extra.erase(intEng_R_extra.begin() + first);
      cv_R_extra.erase(cv_R_extra.begin() + first);
      cp_R_extra.erase(cp_R_extra.begin() + first);
      entropy_R_extra.erase(entropy_R_extra.begin() + first);
      czh.erase(czh.begin() + first);
      czh_ddC.erase(czh_ddC.begin() + first);
    } else {
      seen_vapor_ids.insert(vapor_id);
      ++first;
    }
  }

  // argsort vapor ids
  auto vapor_sorted = std::sort(vapor_ids.begin(), vapor_ids.end(),
                                [](int a, int b) { return a < b; });

  // identify duplicated cloud ids and remove them from all vectors
  first = 0;
  int nvapor = vapor_ids.size();
  std::set<int> seen_cloud_ids;

  while (first < cloud_ids.size()) {
    int cloud_id = cloud_ids[first];
    if (seen_cloud_ids.find(cloud_id) != seen_cloud_ids.end()) {
      // duplicate found, remove it from all vectors
      cloud_ids.erase(cloud_ids.begin() + first);
      cref_R.erase(cref_R.begin() + nvapor + first);
      uref_R.erase(uref_R.begin() + nvapor + first);
      sref_R.erase(sref_R.begin() + nvapor + first);
      intEng_R_extra.erase(intEng_R_extra.begin() + nvapor + first);
      cv_R_extra.erase(cv_R_extra.begin() + nvapor + first);
      cp_R_extra.erase(cp_R_extra.begin() + nvapor + first);
      entropy_R_extra.erase(entropy_R_extra.begin() + nvapor + first);
      czh.erase(czh.begin() + nvapor + first);
      czh_ddC.erase(czh_ddC.begin() + nvapor + first);
    } else {
      seen_cloud_ids.insert(cloud_id);
      ++first;
    }
  }

  // argsort cloud ids
  auto cloud_sorted = std::sort(cloud_ids.begin(), cloud_ids.end(),
                                [](int a, int b) { return a < b; });

  auto sorted = merge_vectors(vapor_sorted, cloud_sorted);

  // re-arrange all vectors according to the sorted indices
  vapor_ids = sort_vectors(vapor_ids, vapor_sorted);
  cloud_ids = sort_vectors(cloud_ids, cloud_sorted);

  cref_R = sort_vectors(cref_R, sorted);
  uref_R = sort_vectors(uref_R, sorted);
  sref_R = sort_vectors(sref_R, sorted);

  intEng_R_extra = sort_vectors(intEng_R_extra, sorted);
  cv_R_extra = sort_vectors(cv_R_extra, sorted);
  cp_R_extra = sort_vectors(cp_R_extra, sorted);
  entropy_R_extra = sort_vectors(entropy_R_extra, sorted);

  czh = sort_vectors(czh, sorted);
  czh_ddC = sort_vectors(czh_ddC, sorted);

  return merged;
}

}  // namespace kintera
