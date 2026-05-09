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
struct PhotoChemOptionsImpl;
using PhotoChemOptions = std::shared_ptr<PhotoChemOptionsImpl>;

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

struct KBPunHeader {
  int natom = 0;
  int nmol = 0;
  int nreact = 0;
  int npart = 0;
  int version = 0;
};

struct KBPunSpecies {
  int id = 0;
  std::string name;
  int first_reaction = 0;
  int n_reactions = 0;
  double molecular_weight = 0.0;
  std::vector<int> composition;
};

struct KBPunReaction {
  int id = 0;
  int reaction_type = 0;
  int n_products = 0;
  std::vector<int> reactant_ids;
  std::vector<int> product_ids;
  std::string raw_line;
};

struct KBPunNetwork {
  KBPunHeader header;
  std::map<std::string, double> elements;
  std::vector<KBPunSpecies> species;
  std::vector<KBPunReaction> reactions;
};

struct KBRunSelection {
  int nfix = 0;
  int nvarys = 0;
  int nvaryf = 0;
  int nphoto = 0;
  int nphots = 0;
  int nphotr = 0;
  int nphotd = 0;
  std::vector<int> fixed_species_ids;
  std::vector<int> varying_slow_species_ids;
  std::vector<int> varying_fast_species_ids;
  std::vector<int> photolysis_reaction_ids;
};

struct KBTitanNetwork {
  KBPunNetwork pun;
  KBRunSelection selection;
  std::vector<std::pair<std::string, std::string>> catalog;
  int resolved_cross_sections = 0;
  std::vector<int> missing_selected_photolysis_ids;
  std::vector<std::string> missing_cross_section_files;
};

struct KBAtmosphereProfile {
  std::string header;
  std::vector<double> altitude;
  std::vector<double> density;
  std::vector<double> temperature;
  std::vector<double> pressure;
  std::vector<double> eddy_diffusion;
  std::vector<double> wind;
  std::map<std::string, std::vector<double>> species_profiles;
};

KBMasterData parse_kinetics_base_master(std::string const& filepath);

KBPunNetwork parse_kinetics_base_pun(std::string const& filepath);

KBRunSelection parse_kinetics_base_run_input(std::string const& filepath);

KBAtmosphereProfile parse_kinetics_base_atmosphere(
    std::string const& filepath);

std::vector<std::pair<std::string, std::string>> parse_kinetics_base_catalog(
    std::string const& filepath);

KBCrossSectionFile parse_kinetics_base_cross_section(
    std::string const& filepath);

KBTitanNetwork parse_kinetics_base_titan(
    std::string const& pun_path, std::string const& run_input_path,
    std::string const& photo_catalog_path, std::string const& cross_dir);

void init_species_from_kinetics_base(std::string const& master_input_path);

KineticsOptions kinetics_options_from_kinetics_base(
    std::string const& master_input_path,
    std::string const& photo_catalog_path = "",
    std::string const& cross_dir = "", bool verbose = false);
PhotoChemOptions photochem_options_from_kinetics_base(
    std::string const& master_input_path,
    std::string const& photo_catalog_path = "",
    std::string const& cross_dir = "", bool verbose = false);

}  // namespace kintera
