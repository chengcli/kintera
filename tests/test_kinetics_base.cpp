// external
#include <gtest/gtest.h>

// C/C++
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

// torch
#include <torch/torch.h>

// kintera
#include <configure.h>

#include <kintera/kinetics/kinetics.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>
#include <kintera/photochem/kinetics_base_reader.hpp>
#include <kintera/photochem/photochem.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;

namespace kintera {
extern bool species_initialized;
}

static std::string data_dir() {
  return std::string(KINTERA_ROOT_DIR) + "/tests/kinetics_base/data/";
}

static std::string shell_quote(std::string const& value) {
  std::string quoted = "'";
  for (char c : value) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted += c;
    }
  }
  quoted += "'";
  return quoted;
}

static std::string read_text_file(std::filesystem::path const& path) {
  std::ifstream ifs(path);
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return buffer.str();
}

static void write_fresh_start_run_input(std::filesystem::path const& source,
                                        std::filesystem::path const& target) {
  std::ifstream ifs(source);
  ASSERT_TRUE(ifs.good()) << source;

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(ifs, line)) lines.push_back(line);

  bool in_specific_run_parameters = false;
  for (auto& current : lines) {
    std::string stripped = current;
    stripped.erase(0, stripped.find_first_not_of(" \t\r\n"));
    if (stripped.rfind("NTIME IEND", 0) == 0) {
      in_specific_run_parameters = true;
      continue;
    }
    if (!in_specific_run_parameters || stripped.empty()) continue;

    std::istringstream iss(current);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) tokens.push_back(token);
    if (tokens.size() < 11) break;

    tokens.back() = "0";
    std::ostringstream oss;
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (i > 0) oss << ' ';
      oss << tokens[i];
    }
    current = oss.str();
  }

  std::ofstream ofs(target);
  ASSERT_TRUE(ofs.good()) << target;
  for (auto const& output_line : lines) {
    ofs << output_line << '\n';
  }
}

static void symlink_or_fail(std::filesystem::path const& target,
                            std::filesystem::path const& link) {
  ASSERT_TRUE(std::filesystem::exists(target)) << target;
  std::error_code ec;
  std::filesystem::remove(link, ec);
  ec.clear();
  std::filesystem::create_symlink(target, link, ec);
  ASSERT_FALSE(ec) << "Could not create symlink " << link << " -> " << target
                   << ": " << ec.message();
}

TEST(KineticsBaseParser, ParseMasterInput) {
  auto master = parse_kinetics_base_master(data_dir() + "test_master.inp");

  // Check elements
  EXPECT_EQ(master.elements.size(), 4);
  EXPECT_NEAR(master.elements.at("O"), 16.00, 0.01);
  EXPECT_NEAR(master.elements.at("H"), 1.01, 0.01);
  EXPECT_NEAR(master.elements.at("N"), 14.01, 0.01);
  EXPECT_NEAR(master.elements.at("C"), 12.01, 0.01);

  // Check species
  EXPECT_GE(master.species.size(), 10);

  // Find O2
  auto it = std::find_if(master.species.begin(), master.species.end(),
                         [](auto const& s) { return s.name == "O2"; });
  ASSERT_NE(it, master.species.end());
  EXPECT_EQ(it->composition.at("O"), 2);
  EXPECT_NEAR(it->molecular_weight, 32.0, 0.1);
  EXPECT_GE(it->n_nasa9_ranges, 2);

  // Find O(1D)
  auto it2 = std::find_if(master.species.begin(), master.species.end(),
                          [](auto const& s) { return s.name == "O(1D)"; });
  ASSERT_NE(it2, master.species.end());
  EXPECT_EQ(it2->composition.at("O"), 1);
  EXPECT_NEAR(it2->hf_kcal, 104.9, 0.1);

  // Check N2
  auto it3 = std::find_if(master.species.begin(), master.species.end(),
                          [](auto const& s) { return s.name == "N2"; });
  ASSERT_NE(it3, master.species.end());
  EXPECT_EQ(it3->composition.at("N"), 2);

  std::cout << "Species: " << master.species.size() << std::endl;
  for (auto const& sp : master.species) {
    std::cout << "  " << sp.name << " (MW=" << sp.molecular_weight
              << ", HF=" << sp.hf_kcal << ", NT=" << sp.n_nasa9_ranges << ")"
              << std::endl;
  }

  // Check reactions
  std::cout << "Photolysis: " << master.photolysis.size() << std::endl;
  for (auto const& rxn : master.photolysis) {
    std::cout << "  ";
    for (auto const& r : rxn.reactants) std::cout << r << " + ";
    std::cout << "=> ";
    for (auto const& p : rxn.products) std::cout << p << " + ";
    std::cout << std::endl;
  }

  std::cout << "Thermal: " << master.thermal.size() << std::endl;
  for (auto const& rxn : master.thermal) {
    std::string type =
        rxn.has_M ? (rxn.has_kinf ? "falloff" : "three-body") : "arrhenius";
    std::cout << "  [" << type << "] ";
    for (auto const& r : rxn.reactants) std::cout << r << " + ";
    std::cout << "=> ";
    for (auto const& p : rxn.products) std::cout << p << " + ";
    if (rxn.has_M) {
      std::cout << " (A0=" << rxn.A0 << ", b0=" << rxn.b0
                << ", Ea_R0=" << rxn.Ea_R0 << ")";
      if (rxn.has_kinf)
        std::cout << " (Ainf=" << rxn.A_inf << ", binf=" << rxn.b_inf << ")";
    } else {
      std::cout << " (A=" << rxn.A << ", b=" << rxn.b << ", Ea_R=" << rxn.Ea_R
                << ")";
    }
    std::cout << std::endl;
  }

  EXPECT_GE(master.photolysis.size(), 5);
  EXPECT_GE(master.thermal.size(), 5);

  // Count reaction types
  int n_arrhenius = 0, n_three_body = 0, n_falloff = 0;
  for (auto const& rxn : master.thermal) {
    if (rxn.has_M) {
      if (rxn.has_kinf)
        ++n_falloff;
      else
        ++n_three_body;
    } else {
      ++n_arrhenius;
    }
  }
  std::cout << "  Arrhenius: " << n_arrhenius << std::endl;
  std::cout << "  Three-body: " << n_three_body << std::endl;
  std::cout << "  Falloff: " << n_falloff << std::endl;

  EXPECT_GE(n_arrhenius, 1);
  EXPECT_GE(n_three_body, 1);
  EXPECT_GE(n_falloff, 1);
}

TEST(KineticsBaseParser, ParseCatalog) {
  auto catalog = parse_kinetics_base_catalog(data_dir() + "test_catalog.dat");
  EXPECT_GE(catalog.size(), 3);

  std::cout << "Catalog entries:" << std::endl;
  for (auto const& [eq, fname] : catalog) {
    std::cout << "  " << eq << " -> " << fname << std::endl;
  }
}

TEST(KineticsBaseParser, ParseCrossSection) {
  auto csf = parse_kinetics_base_cross_section(data_dir() +
                                               "cross/CROSS_O2=O2_LORES1.DAT");

  EXPECT_FALSE(csf.datasets.empty());
  if (!csf.datasets.empty()) {
    EXPECT_EQ(csf.datasets[0].type, 0);
    EXPECT_GT(csf.datasets[0].wavelengths_nm.size(), 10);

    std::cout << "Cross-section: " << csf.equation << std::endl;
    std::cout << "  Datasets: " << csf.datasets.size() << std::endl;
    std::cout << "  Type: " << csf.datasets[0].type << std::endl;
    std::cout << "  N wavelengths: " << csf.datasets[0].wavelengths_nm.size()
              << std::endl;
    if (!csf.datasets[0].wavelengths_nm.empty()) {
      std::cout << "  First wavelength (nm): "
                << csf.datasets[0].wavelengths_nm.front() << std::endl;
      std::cout << "  Last wavelength (nm): "
                << csf.datasets[0].wavelengths_nm.back() << std::endl;
    }
  }

  // Check branching ratio file
  auto csf2 = parse_kinetics_base_cross_section(data_dir() +
                                                "cross/CROSS_O2=2O_LORES1.DAT");
  EXPECT_FALSE(csf2.datasets.empty());
  if (!csf2.datasets.empty()) {
    EXPECT_EQ(csf2.datasets[0].type, 2);
    std::cout << "Cross-section (branching): " << csf2.equation << std::endl;
    std::cout << "  Type: " << csf2.datasets[0].type << std::endl;
    std::cout << "  N wavelengths: " << csf2.datasets[0].wavelengths_nm.size()
              << std::endl;
  }
}

TEST(KineticsBaseParser, ParseMinimalPunAndRunInput) {
  auto pun = parse_kinetics_base_pun(data_dir() + "test_titan_minimal.pun");
  EXPECT_EQ(pun.header.natom, 2);
  EXPECT_EQ(pun.header.nmol, 3);
  EXPECT_EQ(pun.header.nreact, 2);
  EXPECT_EQ(pun.elements.at("H"), 1.01);
  EXPECT_EQ(pun.elements.at("C"), 12.01);
  ASSERT_EQ(pun.species.size(), 3);
  EXPECT_EQ(pun.species[0].name, "H");
  EXPECT_EQ(pun.species[1].name, "CH4");
  EXPECT_EQ(pun.species[2].name, "M");
  ASSERT_EQ(pun.reactions.size(), 2);
  EXPECT_EQ(pun.reactions[0].id, 1);
  EXPECT_EQ(pun.reactions[0].reaction_type, 2);
  EXPECT_EQ(pun.reactions[0].reactant_ids, std::vector<int>({2}));
  EXPECT_EQ(pun.reactions[0].product_ids, std::vector<int>({1, 1}));
  EXPECT_EQ(pun.reactions[1].id, 2);
  EXPECT_EQ(pun.reactions[1].reaction_type, 3);
  EXPECT_EQ(pun.reactions[1].reactant_ids, std::vector<int>({1, 2}));
  EXPECT_EQ(pun.reactions[1].product_ids, std::vector<int>({3, 1}));

  auto selection =
      parse_kinetics_base_run_input(data_dir() + "test_titan_run.inp");
  EXPECT_EQ(selection.nfix, 1);
  EXPECT_EQ(selection.nvarys, 0);
  EXPECT_EQ(selection.nvaryf, 2);
  EXPECT_EQ(selection.nphoto, 1);
  EXPECT_EQ(selection.fixed_species_ids, std::vector<int>({1}));
  EXPECT_EQ(selection.varying_fast_species_ids, std::vector<int>({2, 3}));
  EXPECT_EQ(selection.photolysis_reaction_ids, std::vector<int>({2}));
}

TEST(KineticsBaseParser, ParseExternalTitanPunIfAvailable) {
  const char* root_env = std::getenv("KINTERA_KINETICS_BASE_ROOT");
  if (root_env == nullptr || std::string(root_env).empty()) {
    GTEST_SKIP() << "Set KINTERA_KINETICS_BASE_ROOT to run full Titan parser "
                    "smoke test";
  }

  std::string root(root_env);
  if (!root.empty() && root.back() != '/') root += "/";

  auto titan = parse_kinetics_base_titan(
      root + "examples/titan/kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun",
      root + "examples/titan/ions_c6h7+_H2CN.inp-1",
      root + "examples/titan/Cheng_catalog_v4.dat",
      root + "examples/titan/Cheng_cross");
  auto const& pun = titan.pun;
  EXPECT_EQ(pun.header.natom, 8);
  EXPECT_EQ(pun.header.nmol, 268);
  EXPECT_EQ(pun.header.nreact, 2139);
  EXPECT_EQ(pun.header.npart, 517);
  EXPECT_EQ(pun.species.size(), 268);
  EXPECT_EQ(pun.reactions.size(), 2139);
  EXPECT_EQ(pun.species.front().name, "H");
  EXPECT_EQ(pun.species.back().name, "M");
  ASSERT_FALSE(pun.reactions.empty());
  EXPECT_FALSE(pun.reactions.front().reactant_ids.empty());
  EXPECT_FALSE(pun.reactions.front().product_ids.empty());

  auto const& selection = titan.selection;
  EXPECT_EQ(selection.nfix, 8);
  EXPECT_EQ(selection.nvarys, 0);
  EXPECT_EQ(selection.nvaryf, 120);
  EXPECT_EQ(selection.nphoto, 9);
  EXPECT_EQ(selection.fixed_species_ids.size(), 8);
  EXPECT_EQ(selection.varying_fast_species_ids.size(), 120);
  EXPECT_EQ(selection.photolysis_reaction_ids.size(), 9);
  EXPECT_EQ(selection.photolysis_reaction_ids.front(), 221);
  EXPECT_TRUE(titan.missing_selected_photolysis_ids.empty());

  EXPECT_EQ(titan.catalog.size(), 522);
  EXPECT_EQ(titan.resolved_cross_sections, 522);
  EXPECT_TRUE(titan.missing_cross_section_files.empty());
}

TEST(KineticsBaseParser, MatchKineticsBaseTitanFirstStepIfAvailable) {
  const char* root_env = std::getenv("KINTERA_KINETICS_BASE_ROOT");
  const char* exe_env = std::getenv("KINTERA_KINETICS_BASE_EXECUTABLE");
  if (root_env == nullptr || std::string(root_env).empty() ||
      exe_env == nullptr || std::string(exe_env).empty()) {
    GTEST_SKIP() << "Set KINTERA_KINETICS_BASE_ROOT and "
                    "KINTERA_KINETICS_BASE_EXECUTABLE to run the "
                    "KINETICS-base first-step oracle";
  }

  namespace fs = std::filesystem;
  fs::path root(root_env);
  fs::path exe(exe_env);
  ASSERT_TRUE(fs::exists(exe)) << exe;

  auto titan = parse_kinetics_base_titan(
      (root / "examples/titan/kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun")
          .string(),
      (root / "examples/titan/ions_c6h7+_H2CN.inp-1").string(),
      (root / "examples/titan/Cheng_catalog_v4.dat").string(),
      (root / "examples/titan/Cheng_cross").string());

  fs::path workdir =
      fs::temp_directory_path() / "kintera-kinetics-base-first-step";
  std::error_code ec;
  fs::remove_all(workdir, ec);
  ASSERT_TRUE(fs::create_directories(workdir, ec) || fs::exists(workdir))
      << workdir << ": " << ec.message();
  ASSERT_TRUE(fs::create_directories(workdir / "prod+loss", ec) ||
              fs::exists(workdir / "prod+loss"))
      << workdir / "prod+loss" << ": " << ec.message();

  auto titan_dir = root / "examples/titan";
  auto patched_run_input = workdir / "fort.81.fresh-start";
  write_fresh_start_run_input(titan_dir / "ions_c6h7+_H2CN.inp-1",
                              patched_run_input);

  symlink_or_fail(titan_dir / "kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun",
                  workdir / "fort.1");
  symlink_or_fail(titan_dir / "kintitan.truncate", workdir / "fort.3");
  symlink_or_fail(titan_dir / "kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.special",
                  workdir / "fort.4");
  symlink_or_fail(titan_dir / "titan_Cheng_N_ions_H2CN.bc_save",
                  workdir / "fort.15");
  symlink_or_fail(titan_dir / "Cheng_wavel.dat", workdir / "fort.20");
  symlink_or_fail(titan_dir / "flare_kin_oct2003.inp", workdir / "fort.21");
  symlink_or_fail(titan_dir / "kintitan-difrad-2.inp", workdir / "fort.27");
  symlink_or_fail(titan_dir / "Cheng_catalog_v4.dat", workdir / "fort.30");
  symlink_or_fail(titan_dir / "kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz",
                  workdir / "fort.50");
  symlink_or_fail(titan_dir / "kintitan_aerosol_interp_albedo.inp",
                  workdir / "fort.45");
  symlink_or_fail(titan_dir / "kintitan_aerosol_interp_asymm.inp",
                  workdir / "fort.47");
  symlink_or_fail(titan_dir / "kintitan_aerosol_interp_gr.inp",
                  workdir / "fort.46");
  symlink_or_fail(titan_dir / "Cheng_cross", workdir / "crossfilepath");
  symlink_or_fail(patched_run_input, workdir / "fort.81");

  // Keep Fortran outputs inside the scratch directory. The patched run input
  // starts from the provided atmosphere instead of requiring a restart file.
  std::ofstream(workdir / "kintitan.out.pun").close();
  std::ofstream(workdir / "kintitan.res").close();
  std::ofstream(workdir / "titandebug.dat").close();
  symlink_or_fail(workdir / "kintitan.out.pun", workdir / "fort.7");
  symlink_or_fail(workdir / "kintitan.res", workdir / "fort.10");
  symlink_or_fail(workdir / "titandebug.dat", workdir / "fort.11");

  auto stdout_path = workdir / "kinetics-base.stdout";
  std::string command = "cd " + shell_quote(workdir.string()) + " && " +
                        shell_quote(exe.string()) + " > " +
                        shell_quote(stdout_path.string()) + " 2>&1";
  int status = std::system(command.c_str());

  ASSERT_TRUE(fs::exists(stdout_path));
  auto output = read_text_file(stdout_path);
  EXPECT_EQ(status, 0) << "KINETICS-base execution failed; see "
                       << stdout_path;
  EXPECT_NE(output.find("CONCENTRATIONS OF   8 SPECIES ARE HELD CONSTANT"),
            std::string::npos);
  EXPECT_NE(output.find("CONCENTRATIONS OF 120 SPECIES TO BE CALCULATED WITH "
                        "VERTICAL TRANSPORT"),
            std::string::npos);
  EXPECT_NE(output.find("JDUST       N2          E           PROD"), std::string::npos);
  EXPECT_NE(output.find("GROUP  1 : H           H2          C           CH"),
            std::string::npos);
  EXPECT_NE(output.find("GROUP  2 : CH4"), std::string::npos);

  EXPECT_EQ(titan.selection.nfix, 8);
  EXPECT_EQ(titan.selection.nvaryf, 120);
  EXPECT_EQ(titan.selection.fixed_species_ids.size(), 8);
  EXPECT_EQ(titan.selection.varying_fast_species_ids.size(), 120);
  EXPECT_TRUE(fs::exists(workdir / "Reactions.dat"));
  auto reactions_output = read_text_file(workdir / "Reactions.dat");
  EXPECT_NE(reactions_output.find("H2"), std::string::npos);
  EXPECT_NE(reactions_output.find("2H"), std::string::npos);

  fs::remove_all(workdir, ec);
}

TEST_P(DeviceTest, KineticsBaseLoadNoXsec) {
  // Reset species state
  species_initialized = false;

  auto op = KineticsOptionsImpl::from_kinetics_base(
      data_dir() + "test_master.inp", "", "", true);

  ASSERT_NE(op, nullptr);

  auto reactions = op->reactions();
  std::cout << "Total reactions: " << reactions.size() << std::endl;
  for (auto const& rxn : reactions) {
    std::cout << "  " << rxn.equation() << std::endl;
  }

  EXPECT_GT(op->arrhenius()->reactions().size(), 0);
  EXPECT_GT(op->three_body()->reactions().size(), 0);
  EXPECT_GT(op->lindemann_falloff()->reactions().size(), 0);
  Kinetics kinet(op);
  kinet->to(device, dtype);

  std::cout << "Kinetics module created successfully" << std::endl;
  std::cout << "Stoich shape: " << kinet->stoich.sizes() << std::endl;
}

TEST_P(DeviceTest, KineticsBaseLoadWithXsec) {
  species_initialized = false;

  auto op = PhotoChemOptionsImpl::from_kinetics_base(
      data_dir() + "test_master.inp", data_dir() + "test_catalog.dat",
      data_dir() + "cross/", true);

  ASSERT_NE(op, nullptr);

  std::cout << "Photolysis reactions: " << op->photolysis()->reactions().size()
            << std::endl;
  std::cout << "Wavelength grid size: " << op->photolysis()->wavelength().size()
            << std::endl;
  std::cout << "Cross-section data size: "
            << op->photolysis()->cross_section().size() << std::endl;

  auto const& wave = op->photolysis()->wavelength();
  EXPECT_TRUE(std::find(wave.begin(), wave.end(), 317.5) != wave.end());

  PhotoChem photo(op);
  photo->to(device, dtype);
  auto query_wave = torch::tensor({317.5}, torch::device(device).dtype(dtype));
  auto query_temp = torch::tensor({298.0}, torch::device(device).dtype(dtype));
  auto xs = photo->photolysis->interp_cross_section(3, query_wave, query_temp);
  EXPECT_GT(xs[0][1].item<double>(), 0.0);
}

TEST_P(DeviceTest, KineticsBasePhotoChemRequiresCatalog) {
  species_initialized = false;

  EXPECT_THROW(PhotoChemOptionsImpl::from_kinetics_base(
                   data_dir() + "test_master.inp", "", "", true),
               c10::Error);
}

TEST_P(DeviceTest, KineticsBaseForward) {
  species_initialized = false;

  auto op_kinet = KineticsOptionsImpl::from_kinetics_base(
      data_dir() + "test_master.inp", data_dir() + "test_catalog.dat",
      data_dir() + "cross/");
  auto op_photo = PhotoChemOptionsImpl::from_kinetics_base(
      data_dir() + "test_master.inp", data_dir() + "test_catalog.dat",
      data_dir() + "cross/");

  Kinetics kinet(op_kinet);
  PhotoChem photo(op_photo);
  kinet->to(device, dtype);
  photo->to(device, dtype);

  auto species = op_kinet->species();
  int nspecies = species.size();
  std::cout << "Species (" << nspecies << "): ";
  for (auto const& s : species) std::cout << s << " ";
  std::cout << std::endl;

  auto conc =
      torch::ones({1, nspecies}, torch::device(device).dtype(dtype)) * 1e18;
  auto temp = 300.0 * torch::ones({1}, torch::device(device).dtype(dtype));
  auto pres = 1.0e5 * torch::ones({1}, torch::device(device).dtype(dtype));

  auto wave = photo->photolysis->wavelength.to(device, dtype);
  auto aflux = torch::ones_like(wave) * 1e14;

  auto [rate, rc_ddC, rc_ddT] = kinet->forward(temp, pres, conc);
  photo->photolysis->update_xs_diss_stacked(temp);
  auto photo_rate = photo->forward(temp, conc, aflux);

  std::cout << "Rate shape: " << rate.sizes() << std::endl;
  std::cout << "Rate: " << rate << std::endl;
  std::cout << "Photo rate: " << photo_rate << std::endl;

  auto du =
      rate.matmul(kinet->stoich.t()) + photo_rate.matmul(photo->stoich.t());
  std::cout << "du: " << du << std::endl;

  int nrxn = op_kinet->reactions().size();
  int n_reversible = 0;
  for (auto const& r : op_kinet->reactions()) {
    if (r.reversible()) n_reversible++;
  }
  EXPECT_EQ(rate.size(-1), (int64_t)(nrxn + n_reversible));
  EXPECT_EQ(photo_rate.size(-1), (int64_t)op_photo->reactions().size());
  EXPECT_TRUE(rate.isfinite().any().item<bool>());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
