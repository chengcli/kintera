// C/C++
#include <algorithm>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/kinetics/actinic_flux.hpp>
#include <kintera/kinetics/jacobian.hpp>
#include <kintera/kinetics/kinetics.hpp>
#include <kintera/kinetics/photolysis.hpp>
#include <kintera/utils/parse_comp_string.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;

class CH4PhotolysisTest : public DeviceTest {
 protected:
  void SetUp() override {
    DeviceTest::SetUp();
    // Initialize CH4 photolysis species
    extern std::vector<std::string> species_names;
    extern std::vector<double> species_weights;
    extern std::vector<double> species_cref_R;
    extern std::vector<double> species_uref_R;
    extern std::vector<double> species_sref_R;
    extern bool species_initialized;

    species_names = {"CH4", "CH3", "(1)CH2", "(3)CH2", "CH", "H2", "H", "N2"};
    species_weights = {16.04, 15.03, 14.03, 14.03, 13.02, 2.02, 1.01, 28.01};
    species_cref_R = {2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 1.5, 2.5};
    species_uref_R = {0., 0., 0., 0., 0., 0., 0., 0.};
    species_sref_R = {0., 0., 0., 0., 0., 0., 0., 0.};
    species_initialized = true;
  }
};

// Test CH4 photolysis cross-section integration
TEST_P(CH4PhotolysisTest, SingleBranchPhotolysis) {
  // Simple N2 photoabsorption test
  auto opts = PhotolysisOptionsImpl::create();
  opts->wavelength() = {100., 120., 140., 160., 180., 200.};
  opts->temperature() = {200., 300.};
  opts->reactions().push_back(Reaction("N2 => N2"));

  // Cross-section data (typical N2 absorption values in cm^2)
  opts->cross_section() = {1.e-18, 1.5e-18, 2.e-18, 1.5e-18, 1.e-18, 0.5e-18};
  opts->branches().push_back({parse_comp_string("N2:1")});

  Photolysis module(opts);
  module->to(device, dtype);

  // Test inputs
  auto temp = torch::tensor({250.}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({1.e5}, torch::device(device).dtype(dtype));
  auto conc = torch::zeros({1, 8}, torch::device(device).dtype(dtype));
  conc[0][7] = 1.e3;  // N2 concentration

  // Uniform actinic flux
  auto flux_data = create_uniform_flux(100., 200., 6, 1.e14, device, dtype);
  auto other = flux_data.to_map();

  auto rate = module->forward(temp, pres, conc, other);

  EXPECT_EQ(rate.dim(), 2);
  EXPECT_EQ(rate.size(-1), 1);
  // Rate should be positive (integral of positive xs * positive flux)
  EXPECT_GT(rate[0][0].item<double>(), 0.);
}

// Test multi-branch CH4 photolysis
TEST_P(CH4PhotolysisTest, MultiBranchPhotolysis) {
  auto opts = PhotolysisOptionsImpl::create();
  opts->wavelength() = {100., 120., 140., 160.};
  opts->temperature() = {200., 300.};

  // CH4 + hν -> various products
  opts->reactions().push_back(
      Reaction("CH4 => CH4 + CH3 + (1)CH2 + (3)CH2 + H2 + H"));

  // Cross-section for each branch at each wavelength
  // Format: [wave1_branch1, wave1_branch2, ..., wave2_branch1, ...]
  // 5 branches total (including photoabsorption)
  std::vector<double> xs_data;
  int nwave = 4;
  int nbranch = 5;
  for (int w = 0; w < nwave; w++) {
    // Photoabsorption (first branch)
    xs_data.push_back(0.5e-18);
    // CH3 + H branch
    xs_data.push_back(0.3e-18);
    // (1)CH2 + H2 branch
    xs_data.push_back(0.1e-18);
    // (3)CH2 + 2H branch
    xs_data.push_back(0.05e-18);
    // H2 + other products
    xs_data.push_back(0.05e-18);
  }
  opts->cross_section() = xs_data;

  std::vector<Composition> branches;
  branches.push_back(parse_comp_string("CH4:1"));       // photoabsorption
  branches.push_back(parse_comp_string("CH3:1 H:1"));   // CH3 + H
  branches.push_back(parse_comp_string("(1)CH2:1 H2:1"));  // singlet CH2 + H2
  branches.push_back(parse_comp_string("(3)CH2:1 H:2"));   // triplet CH2 + 2H
  branches.push_back(parse_comp_string("H2:1 H:1"));       // H2 + H
  opts->branches().push_back(branches);

  Photolysis module(opts);
  module->to(device, dtype);

  auto temp = torch::tensor({250.}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({1.e5}, torch::device(device).dtype(dtype));
  auto conc = torch::zeros({1, 8}, torch::device(device).dtype(dtype));
  conc[0][0] = 1.e3;  // CH4 concentration

  auto flux_data = create_solar_flux(100., 160., 4, 1.e14, device, dtype);
  auto other = flux_data.to_map();

  auto rate = module->forward(temp, pres, conc, other);

  EXPECT_GT(rate[0][0].item<double>(), 0.);
}

// Test effective stoichiometry calculation
TEST_P(CH4PhotolysisTest, EffectiveStoichiometry) {
  auto opts = PhotolysisOptionsImpl::create();
  opts->wavelength() = {100., 150., 200.};
  opts->temperature() = {200., 300.};

  opts->reactions().push_back(Reaction("N2 => N2 + N2"));

  // Two branches with different cross-sections
  opts->cross_section() = {1.e-18, 0.5e-18, 2.e-18, 0.5e-18, 1.e-18, 0.5e-18};

  std::vector<Composition> branches;
  branches.push_back(parse_comp_string("N2:1"));  // branch 1
  branches.push_back(parse_comp_string("N2:2"));  // branch 2 (double stoich)
  opts->branches().push_back(branches);

  Photolysis module(opts);
  module->to(device, dtype);

  auto wave =
      torch::tensor({100., 150., 200.}, torch::device(device).dtype(dtype));
  auto aflux = torch::ones({3}, torch::device(device).dtype(dtype));
  auto temp = torch::tensor({250.}, torch::device(device).dtype(dtype));

  auto eff_stoich = module->get_effective_stoich(0, wave, aflux, temp);

  // Should be weighted average of branch stoichiometries
  EXPECT_EQ(eff_stoich.size(0), 8);  // nspecies
}

// Test Jacobian for photolysis
TEST_P(CH4PhotolysisTest, PhotolysisJacobian) {
  // Setup a simple photolysis reaction
  auto rate = torch::tensor({1.e-5}, torch::device(device).dtype(dtype));
  auto stoich =
      torch::tensor({{-1.}, {1.}}, torch::device(device).dtype(dtype));
  auto conc = torch::tensor({1.e3, 0.}, torch::device(device).dtype(dtype));
  auto rc_ddC = torch::zeros({2, 1}, torch::device(device).dtype(dtype));

  auto jac = jacobian_photolysis(rate, stoich, conc, rc_ddC);

  EXPECT_EQ(jac.dim(), 2);
  EXPECT_EQ(jac.size(0), 2);
  EXPECT_EQ(jac.size(1), 2);

  // The Jacobian should show reactant depletion and product formation
  // d([A])/d[A] should be negative (reactant consumed)
  // d([B])/d[A] should be positive (product formed)
  EXPECT_LT(jac[0][0].item<double>(), 0.);  // d(dA/dt)/dA < 0
  EXPECT_GT(jac[1][0].item<double>(), 0.);  // d(dB/dt)/dA > 0
}

// Test integration with Kinetics module
TEST_P(CH4PhotolysisTest, KineticsIntegration) {
  auto kinet_opts = KineticsOptionsImpl::create();

  // Setup species thermo
  kinet_opts->vapor_ids() = {0, 7};  // CH4, N2
  kinet_opts->cref_R() = {2.5, 2.5};
  kinet_opts->uref_R() = {0., 0.};
  kinet_opts->sref_R() = {0., 0.};

  // Add photolysis reaction
  kinet_opts->photolysis()->wavelength() = {100., 150., 200.};
  kinet_opts->photolysis()->temperature() = {200., 300.};
  kinet_opts->photolysis()->reactions().push_back(Reaction("N2 => N2"));
  kinet_opts->photolysis()->cross_section() = {1.e-18, 2.e-18, 1.e-18};
  kinet_opts->photolysis()->branches().push_back({parse_comp_string("N2:1")});

  // Create kinetics module
  Kinetics kinet(kinet_opts);
  kinet->to(device, dtype);

  // Verify total reactions
  auto all_rxns = kinet_opts->reactions();
  EXPECT_EQ(all_rxns.size(), 1);  // Just the photolysis reaction
}

// Test cross-section interpolation
TEST_P(CH4PhotolysisTest, CrossSectionInterpolation) {
  auto opts = PhotolysisOptionsImpl::create();
  opts->wavelength() = {100., 200., 300.};
  opts->temperature() = {200., 400.};
  opts->reactions().push_back(Reaction("N2 => N2"));
  opts->cross_section() = {1.e-18, 3.e-18, 1.e-18};
  opts->branches().push_back({parse_comp_string("N2:1")});

  Photolysis module(opts);
  module->to(device, dtype);

  // Interpolate at midpoints
  auto query_wave =
      torch::tensor({150., 250.}, torch::device(device).dtype(dtype));
  auto query_temp = torch::tensor({300.}, torch::device(device).dtype(dtype));

  auto xs = module->interp_cross_section(0, query_wave, query_temp);

  // At 150 nm (between 100 and 200), xs should be between 1e-18 and 3e-18
  EXPECT_GT(xs[0][0].item<double>(), 1.e-18);
  EXPECT_LT(xs[0][0].item<double>(), 3.e-18);

  // At 250 nm (between 200 and 300), xs should be between 1e-18 and 3e-18
  EXPECT_GT(xs[1][0].item<double>(), 1.e-18);
  EXPECT_LT(xs[1][0].item<double>(), 3.e-18);
}

// Test with varying actinic flux profiles
TEST_P(CH4PhotolysisTest, VaryingActinicFlux) {
  auto opts = PhotolysisOptionsImpl::create();
  opts->wavelength() = {100., 150., 200., 250., 300.};
  opts->temperature() = {200., 300.};
  opts->reactions().push_back(Reaction("N2 => N2"));
  opts->cross_section() = {1.e-18, 2.e-18, 3.e-18, 2.e-18, 1.e-18};
  opts->branches().push_back({parse_comp_string("N2:1")});

  Photolysis module(opts);
  module->to(device, dtype);

  auto temp = torch::tensor({250.}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({1.e5}, torch::device(device).dtype(dtype));
  auto conc = torch::zeros({1, 8}, torch::device(device).dtype(dtype));

  // Test 1: Uniform flux
  auto flux1 = create_uniform_flux(100., 300., 5, 1.e14, device, dtype);
  auto rate1 = module->forward(temp, pres, conc, flux1.to_map());

  // Test 2: Solar-like flux (peaks at visible, low in UV)
  auto flux2 = create_solar_flux(100., 300., 5, 1.e14, device, dtype);
  auto rate2 = module->forward(temp, pres, conc, flux2.to_map());

  // Both should give positive rates
  EXPECT_GT(rate1[0][0].item<double>(), 0.);
  EXPECT_GT(rate2[0][0].item<double>(), 0.);

  // Rates should differ because flux profiles differ
  EXPECT_NE(rate1[0][0].item<double>(), rate2[0][0].item<double>());
}

INSTANTIATE_TEST_SUITE_P(
    DeviceTests, CH4PhotolysisTest,
    testing::Values(Parameters{torch::kCPU, torch::kFloat64},
                    Parameters{torch::kCUDA, torch::kFloat64}),
    [](const testing::TestParamInfo<CH4PhotolysisTest::ParamType>& info) {
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

