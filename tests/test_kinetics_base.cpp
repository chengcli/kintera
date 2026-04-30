// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <configure.h>

#include <kintera/kinetics/kinetics.hpp>
#include <kintera/kinetics/kinetics_base_reader.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;

namespace kintera {
extern bool species_initialized;
}

static std::string data_dir() {
  return std::string(KINTERA_ROOT_DIR) + "/tests/kinetics_base/data/";
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
  EXPECT_GT(op->photolysis()->reactions().size(), 0);

  Kinetics kinet(op);
  kinet->to(device, dtype);

  std::cout << "Kinetics module created successfully" << std::endl;
  std::cout << "Stoich shape: " << kinet->stoich.sizes() << std::endl;
}

TEST_P(DeviceTest, KineticsBaseLoadWithXsec) {
  species_initialized = false;

  auto op = KineticsOptionsImpl::from_kinetics_base(
      data_dir() + "test_master.inp", data_dir() + "test_catalog.dat",
      data_dir() + "cross/", true);

  ASSERT_NE(op, nullptr);

  std::cout << "Photolysis reactions: " << op->photolysis()->reactions().size()
            << std::endl;
  std::cout << "Wavelength grid size: " << op->photolysis()->wavelength().size()
            << std::endl;
  std::cout << "Cross-section data size: "
            << op->photolysis()->cross_section().size() << std::endl;
}

TEST_P(DeviceTest, KineticsBaseForward) {
  species_initialized = false;

  auto op = KineticsOptionsImpl::from_kinetics_base(
      data_dir() + "test_master.inp", data_dir() + "test_catalog.dat",
      data_dir() + "cross/");

  Kinetics kinet(op);
  kinet->to(device, dtype);

  auto species = op->species();
  int nspecies = species.size();
  std::cout << "Species (" << nspecies << "): ";
  for (auto const& s : species) std::cout << s << " ";
  std::cout << std::endl;

  auto conc =
      torch::ones({1, nspecies}, torch::device(device).dtype(dtype)) * 1e18;
  auto temp = 300.0 * torch::ones({1}, torch::device(device).dtype(dtype));
  auto pres = 1.0e5 * torch::ones({1}, torch::device(device).dtype(dtype));

  int nwave = 10;
  auto wave =
      torch::linspace(100, 300, nwave, torch::device(device).dtype(dtype));
  auto aflux = torch::ones({nwave}, torch::device(device).dtype(dtype)) * 1e14;

  std::map<std::string, torch::Tensor> extra;
  extra["wavelength"] = wave;
  extra["actinic_flux"] = aflux;

  auto [rate, rc_ddC, rc_ddT] = kinet->forward(temp, pres, conc, extra);

  std::cout << "Rate shape: " << rate.sizes() << std::endl;
  std::cout << "Rate: " << rate << std::endl;

  auto du = rate.matmul(kinet->stoich.t());
  std::cout << "du: " << du << std::endl;

  int nrxn = op->reactions().size();
  int n_reversible = 0;
  for (auto const& r : op->reactions()) {
    if (r.reversible()) n_reversible++;
  }
  EXPECT_EQ(rate.size(-1), (int64_t)(nrxn + n_reversible));
  EXPECT_TRUE(rate.isfinite().any().item<bool>());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
