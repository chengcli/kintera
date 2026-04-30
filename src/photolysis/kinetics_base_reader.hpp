#pragma once

// C/C++
#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace kintera {

struct KineticsOptionsImpl;
using KineticsOptions = std::shared_ptr<KineticsOptionsImpl>;

struct KBSpecies {
  std::string name;
  std::map<std::string, int> composition;
  double molecular_weight = 0.0;
  double hf_kcal = 0.0;

  int n_nasa9_ranges = 0;
  std::array<double, 9> nasa9_low = {};
  std::array<double, 9> nasa9_high = {};
  double nasa9_Tmid = 1000.0;
};

struct KBReaction {
  std::vector<std::string> reactants;
  std::vector<std::string> products;
  bool has_M = false;
  bool has_kinf = false;

  double A = 0, b = 0, Ea_R = 0;
  double A0 = 0, b0 = 0, Ea_R0 = 0;
  double A_inf = 0, b_inf = 0, Ea_R_inf = 0;
};

struct KBCrossSection {
  int type = 0;
  double temperature = 298.0;
  std::vector<double> wavelengths_nm;
  std::vector<double> values;
};

struct KBCrossSectionFile {
  std::string equation;
  std::vector<KBCrossSection> datasets;
};

struct KBMasterData {
  std::map<std::string, double> elements;
  std::vector<KBSpecies> species;
  std::vector<KBReaction> photolysis;
  std::vector<KBReaction> thermal;
};

KBMasterData parse_kinetics_base_master(std::string const& filepath);

std::vector<std::pair<std::string, std::string>> parse_kinetics_base_catalog(
    std::string const& filepath);

KBCrossSectionFile parse_kinetics_base_cross_section(
    std::string const& filepath);

void init_species_from_kinetics_base(std::string const& master_input_path);

KineticsOptions kinetics_options_from_kinetics_base(
    std::string const& master_input_path,
    std::string const& photo_catalog_path = "",
    std::string const& cross_dir = "", bool verbose = false);

}  // namespace kintera
