// C/C++
#include <algorithm>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/kinetics/kinetics.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>
#include <kintera/kinetics/photolysis.hpp>
#include <kintera/utils/parse_comp_string.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;

class PhotolysisKineticsTest : public DeviceTest {
 protected:
  void SetUp() override {
    DeviceTest::SetUp();
    // Initialize species names for testing
    extern std::vector<std::string> species_names;
    extern std::vector<double> species_weights;
    extern std::vector<double> species_cref_R;
    extern std::vector<double> species_uref_R;
    extern std::vector<double> species_sref_R;
    extern bool species_initialized;

    species_names = {"CH4", "CH3", "(1)CH2", "(3)CH2", "CH", "H2", "H", "N2"};
    species_weights = {16.0, 15.0, 14.0, 14.0, 13.0, 2.0, 1.0, 28.0};
    species_cref_R = {2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 1.5, 2.5};
    species_uref_R = {0., 0., 0., 0., 0., 0., 0., 0.};
    species_sref_R = {0., 0., 0., 0., 0., 0., 0., 0.};
    species_initialized = true;
  }
};

TEST_P(PhotolysisKineticsTest, KineticsOptionsWithPhotolysis) {
  auto kinet_opts = KineticsOptionsImpl::create();

  // Verify photolysis option exists
  EXPECT_NE(kinet_opts->photolysis(), nullptr);
  EXPECT_EQ(kinet_opts->photolysis()->reactions().size(), 0);
}

TEST_P(PhotolysisKineticsTest, PhotolysisReactionsInKinetics) {
  auto kinet_opts = KineticsOptionsImpl::create();

  // Add a photolysis reaction manually
  kinet_opts->photolysis()->wavelength() = {100., 150., 200.};
  kinet_opts->photolysis()->temperature() = {200., 300.};
  kinet_opts->photolysis()->reactions().push_back(Reaction("N2 => N2"));
  kinet_opts->photolysis()->cross_section() = {1.e-18, 2.e-18, 1.e-18};
  kinet_opts->photolysis()->branches().push_back(
      {parse_comp_string("N2:1")});

  // Add an Arrhenius reaction
  kinet_opts->arrhenius()->reactions().push_back(Reaction("CH4 => CH3 + H"));
  kinet_opts->arrhenius()->A() = {1.e10};
  kinet_opts->arrhenius()->b() = {0.};
  kinet_opts->arrhenius()->Ea_R() = {10000.};
  kinet_opts->arrhenius()->E4_R() = {0.};

  // Total reactions should include both
  auto all_reactions = kinet_opts->reactions();
  EXPECT_EQ(all_reactions.size(), 2);
}

TEST_P(PhotolysisKineticsTest, KineticsModuleWithPhotolysis) {
  auto kinet_opts = KineticsOptionsImpl::create();

  // Setup species thermo
  kinet_opts->vapor_ids() = {7};  // N2
  kinet_opts->cref_R() = {2.5};
  kinet_opts->uref_R() = {0.};
  kinet_opts->sref_R() = {0.};

  // Add photolysis reaction
  kinet_opts->photolysis()->wavelength() = {100., 150., 200.};
  kinet_opts->photolysis()->temperature() = {200., 300.};
  kinet_opts->photolysis()->reactions().push_back(Reaction("N2 => N2"));
  kinet_opts->photolysis()->cross_section() = {1.e-18, 2.e-18, 1.e-18};
  kinet_opts->photolysis()->branches().push_back(
      {parse_comp_string("N2:1")});

  // Create kinetics module
  Kinetics kinet(kinet_opts);
  kinet->to(device, dtype);

  // Verify stoichiometry matrix
  EXPECT_EQ(kinet->stoich.size(1), 1);  // 1 photolysis reaction
}

TEST_P(PhotolysisKineticsTest, StoichiometryMatrixIncludesPhotolysis) {
  auto kinet_opts = KineticsOptionsImpl::create();

  // Setup minimal species thermo
  kinet_opts->vapor_ids() = {0, 7};  // CH4, N2
  kinet_opts->cref_R() = {2.5, 2.5};
  kinet_opts->uref_R() = {0., 0.};
  kinet_opts->sref_R() = {0., 0.};

  // Add Arrhenius reaction
  kinet_opts->arrhenius()->reactions().push_back(Reaction("CH4 => CH3 + H"));
  kinet_opts->arrhenius()->A() = {1.e10};
  kinet_opts->arrhenius()->b() = {0.};
  kinet_opts->arrhenius()->Ea_R() = {10000.};
  kinet_opts->arrhenius()->E4_R() = {0.};

  // Add photolysis reaction
  kinet_opts->photolysis()->wavelength() = {100., 200.};
  kinet_opts->photolysis()->temperature() = {200., 300.};
  kinet_opts->photolysis()->reactions().push_back(Reaction("N2 => N2"));
  kinet_opts->photolysis()->cross_section() = {1.e-18, 1.e-18};
  kinet_opts->photolysis()->branches().push_back(
      {parse_comp_string("N2:1")});

  Kinetics kinet(kinet_opts);
  kinet->to(device, dtype);

  // Should have 2 reactions total
  EXPECT_EQ(kinet->stoich.size(1), 2);
}

INSTANTIATE_TEST_SUITE_P(
    DeviceTests, PhotolysisKineticsTest,
    testing::Values(Parameters{torch::kCPU, torch::kFloat64},
                    Parameters{torch::kCUDA, torch::kFloat64}),
    [](const testing::TestParamInfo<PhotolysisKineticsTest::ParamType>& info) {
      std::string name = torch::Device(info.param.device_type).str();
      name += "_";
      name += torch::toString(info.param.dtype);
      std::replace(name.begin(), name.end(), '.', '_');
      return name;
    });

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

