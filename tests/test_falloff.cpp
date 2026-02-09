// C/C++
#include <algorithm>
#include <cmath>
#include <set>

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/kinetics/kinetics.hpp>
#include <kintera/kinetics/lindemann_falloff.hpp>
#include <kintera/kinetics/sri_falloff.hpp>
#include <kintera/kinetics/three_body.hpp>
#include <kintera/kinetics/troe_falloff.hpp>
#include <kintera/kintera_formatter.hpp>
#include <kintera/species.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;

namespace kintera {
extern std::vector<std::string> species_names;
extern bool species_initialized;
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DeviceTest);

class FalloffEndToEndTest : public DeviceTest {
 protected:
  void SetUp() override {
    DeviceTest::SetUp();
    // Reset species state so each test's from_yaml properly reinitializes
    kintera::species_initialized = false;
    kintera::species_names = {"H2O2", "O", "H2O", "OH", "AR", "H2", "N2", "O2", "H", "HO2"};
  }
};

// ============================================================================
// Section 1: Options and Parsing
// ============================================================================

TEST_P(FalloffEndToEndTest, default_options) {
  auto three_body_opts = ThreeBodyOptionsImpl::create();
  EXPECT_TRUE(three_body_opts->reactions().empty());
  EXPECT_TRUE(three_body_opts->k0_A().empty());
  EXPECT_DOUBLE_EQ(three_body_opts->Tref(), 300.0);
  EXPECT_EQ(three_body_opts->units(), "molecule,cm,s");

  auto lindemann_opts = LindemannFalloffOptionsImpl::create();
  EXPECT_TRUE(lindemann_opts->reactions().empty());
  EXPECT_TRUE(lindemann_opts->k0_A().empty());
  EXPECT_TRUE(lindemann_opts->kinf_A().empty());

  auto troe_opts = TroeFalloffOptionsImpl::create();
  EXPECT_TRUE(troe_opts->reactions().empty());

  auto sri_opts = SRIFalloffOptionsImpl::create();
  EXPECT_TRUE(sri_opts->reactions().empty());
}

TEST_P(FalloffEndToEndTest, yaml_parsing_three_body_and_falloff) {
  std::string yaml_str = R"(
species:
  - name: H2O2
    composition: {H: 2, O: 2}
    cv_R: 2.5
  - name: O
    composition: {O: 1}
    cv_R: 2.5
  - name: H2O
    composition: {H: 2, O: 1}
    cv_R: 2.5
  - name: OH
    composition: {H: 1, O: 1}
    cv_R: 2.5
  - name: AR
    composition: {Ar: 1}
    cv_R: 2.5
  - name: H2
    composition: {H: 2}
    cv_R: 2.5
  - name: H
    composition: {H: 1}
    cv_R: 1.5
reference-state:
  Tref: 300.0
  Pref: 101325.0
reactions:
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
  efficiencies: {AR: 0.83, H2: 2.4, H2O: 15.4}
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
  efficiencies: {AR: 0.7, H2: 2.0}
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
  Troe: {A: 0.7346, T3: 94.0, T1: 1756.0, T2: 5182.0}
- equation: O + H2 (+ M) <=> H + OH (+ M)
  type: falloff
  high-P-rate-constant: {A: 1.0e+15, b: -2.0, Ea_R: 0.0}
  low-P-rate-constant: {A: 4.0e+19, b: -3.0, Ea_R: 0.0}
  SRI: {A: 0.54, B: 201.0, C: 1024.0}
)";

  YAML::Node config = YAML::Load(yaml_str);
  auto kinet_opts = KineticsOptionsImpl::from_yaml(config);

  ASSERT_NE(kinet_opts, nullptr);
  // Verify reactions are split into appropriate modules
  EXPECT_EQ(kinet_opts->three_body()->reactions().size(), 1);
  EXPECT_EQ(kinet_opts->lindemann_falloff()->reactions().size(), 1);
  EXPECT_EQ(kinet_opts->troe_falloff()->reactions().size(), 1);
  EXPECT_EQ(kinet_opts->sri_falloff()->reactions().size(), 1);
}

TEST_P(FalloffEndToEndTest, species_registration_from_efficiencies) {
  std::string yaml_str = R"(
species:
  - name: H2O2
    composition: {H: 2, O: 2}
    cv_R: 2.5
  - name: O
    composition: {O: 1}
    cv_R: 2.5
  - name: H2O
    composition: {H: 2, O: 1}
    cv_R: 2.5
  - name: AR
    composition: {Ar: 1}
    cv_R: 2.5
  - name: H2
    composition: {H: 2}
    cv_R: 2.5
  - name: N2
    composition: {N: 2}
    cv_R: 2.5
reference-state:
  Tref: 300.0
  Pref: 101325.0
reactions:
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
  efficiencies: {AR: 0.83, H2: 2.4, H2O: 15.4}
)";

  YAML::Node config = YAML::Load(yaml_str);
  auto kinet_opts = KineticsOptionsImpl::from_yaml(config);

  auto vapor_ids = kinet_opts->vapor_ids();
  int ar_idx = std::find(kintera::species_names.begin(), kintera::species_names.end(), "AR") - kintera::species_names.begin();
  int h2_idx = std::find(kintera::species_names.begin(), kintera::species_names.end(), "H2") - kintera::species_names.begin();
  int h2o_idx = std::find(kintera::species_names.begin(), kintera::species_names.end(), "H2O") - kintera::species_names.begin();

  EXPECT_TRUE(std::find(vapor_ids.begin(), vapor_ids.end(), ar_idx) != vapor_ids.end());
  EXPECT_TRUE(std::find(vapor_ids.begin(), vapor_ids.end(), h2_idx) != vapor_ids.end());
  EXPECT_TRUE(std::find(vapor_ids.begin(), vapor_ids.end(), h2o_idx) != vapor_ids.end());
}

// ============================================================================
// Section 2: Module Initialization
// ============================================================================

TEST_P(FalloffEndToEndTest, module_initialization) {
  std::string yaml_str = R"(
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
  efficiencies: {AR: 0.83, H2: 2.4, H2O: 15.4}
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
  efficiencies: {AR: 0.7, H2: 2.0}
)";

  YAML::Node root = YAML::Load(yaml_str);
  
  // Test three-body module
  auto three_body_opts = ThreeBodyOptionsImpl::from_yaml(root);
  ThreeBody three_body_module(three_body_opts);
  three_body_module->to(device, dtype);
  EXPECT_EQ(three_body_module->efficiency_matrix.size(0), 1);
  EXPECT_EQ(three_body_module->efficiency_matrix.size(1), kintera::species_names.size());
  EXPECT_EQ(three_body_module->k0_A.size(0), 1);

  int ar_idx = std::find(kintera::species_names.begin(), kintera::species_names.end(), "AR") - kintera::species_names.begin();
  int h2_idx = std::find(kintera::species_names.begin(), kintera::species_names.end(), "H2") - kintera::species_names.begin();
  int h2o_idx = std::find(kintera::species_names.begin(), kintera::species_names.end(), "H2O") - kintera::species_names.begin();

  auto eff0 = three_body_module->efficiency_matrix[0];
  EXPECT_DOUBLE_EQ(eff0[ar_idx].item<double>(), 0.83);
  EXPECT_DOUBLE_EQ(eff0[h2_idx].item<double>(), 2.4);
  EXPECT_DOUBLE_EQ(eff0[h2o_idx].item<double>(), 15.4);

  // Test Lindemann falloff module
  auto lindemann_opts = LindemannFalloffOptionsImpl::from_yaml(root);
  LindemannFalloff lindemann_module(lindemann_opts);
  lindemann_module->to(device, dtype);
  EXPECT_EQ(lindemann_module->efficiency_matrix.size(0), 1);
  EXPECT_EQ(lindemann_module->k0_A.size(0), 1);
  EXPECT_EQ(lindemann_module->kinf_A.size(0), 1);
  EXPECT_LT(lindemann_module->kinf_A[0].item<double>(), 1e90);
}

// ============================================================================
// Section 3: Rate Calculations (Forward)
// ============================================================================

TEST_P(FalloffEndToEndTest, forward_three_body) {
  std::string yaml_str = R"(
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
  efficiencies: {AR: 0.83, H2: 2.4}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = ThreeBodyOptionsImpl::from_yaml(root);
  ThreeBody module(opts);
  module->to(device, dtype);

  auto T = torch::tensor({300.0}, dtype).to(device);
  auto P = torch::tensor({101325.0}, dtype).to(device);
  auto C = torch::ones({9}, dtype).to(device) * 1e-3;
  std::map<std::string, torch::Tensor> other;

  auto rate = module->forward(T, P, C, other);

  EXPECT_EQ(rate.size(0), 1);
  EXPECT_EQ(rate.size(1), 1);
  EXPECT_GT(rate[0][0].item<double>(), 0.0);
}

TEST_P(FalloffEndToEndTest, forward_falloff_lindemann) {
  std::string yaml_str = R"(
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = LindemannFalloffOptionsImpl::from_yaml(root);
  LindemannFalloff module(opts);
  module->to(device, dtype);

  auto T = torch::tensor({300.0}, dtype).to(device);
  auto P = torch::tensor({101325.0}, dtype).to(device);
  auto C = torch::ones({9}, dtype).to(device) * 1e-3;
  std::map<std::string, torch::Tensor> other;

  auto rate = module->forward(T, P, C, other);

  EXPECT_EQ(rate.size(0), 1);
  EXPECT_EQ(rate.size(1), 1);
  EXPECT_GT(rate[0][0].item<double>(), 0.0);
}

TEST_P(FalloffEndToEndTest, forward_falloff_troe) {
  std::string yaml_str = R"(
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
  Troe: {A: 0.7346, T3: 94.0, T1: 1756.0, T2: 5182.0}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = TroeFalloffOptionsImpl::from_yaml(root);
  TroeFalloff module(opts);
  module->to(device, dtype);

  auto T = torch::tensor({300.0}, dtype).to(device);
  auto P = torch::tensor({101325.0}, dtype).to(device);
  auto C = torch::ones({9}, dtype).to(device) * 1e-3;
  std::map<std::string, torch::Tensor> other;

  auto rate = module->forward(T, P, C, other);

  EXPECT_EQ(rate.size(0), 1);
  EXPECT_EQ(rate.size(1), 1);
  EXPECT_GT(rate[0][0].item<double>(), 0.0);
}

TEST_P(FalloffEndToEndTest, forward_falloff_sri) {
  std::string yaml_str = R"(
- equation: O + H2 (+ M) <=> H + OH (+ M)
  type: falloff
  high-P-rate-constant: {A: 1.0e+15, b: -2.0, Ea_R: 0.0}
  low-P-rate-constant: {A: 4.0e+19, b: -3.0, Ea_R: 0.0}
  SRI: {A: 0.54, B: 201.0, C: 1024.0}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = SRIFalloffOptionsImpl::from_yaml(root);
  SRIFalloff module(opts);
  module->to(device, dtype);

  auto T = torch::tensor({300.0}, dtype).to(device);
  auto P = torch::tensor({101325.0}, dtype).to(device);
  auto C = torch::ones({9}, dtype).to(device) * 1e-3;
  std::map<std::string, torch::Tensor> other;

  auto rate = module->forward(T, P, C, other);

  EXPECT_EQ(rate.size(0), 1);
  EXPECT_EQ(rate.size(1), 1);
  EXPECT_GT(rate[0][0].item<double>(), 0.0);
}

TEST_P(FalloffEndToEndTest, forward_multiple_reactions) {
  // Test that each module can handle multiple reactions of the same type
  std::string yaml_str = R"(
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = LindemannFalloffOptionsImpl::from_yaml(root);
  LindemannFalloff module(opts);
  module->to(device, dtype);

  auto T = torch::tensor({300.0}, dtype).to(device);
  auto P = torch::tensor({101325.0}, dtype).to(device);
  auto C = torch::ones({9}, dtype).to(device) * 1e-3;
  std::map<std::string, torch::Tensor> other;

  auto rate = module->forward(T, P, C, other);

  EXPECT_EQ(rate.size(0), 1);
  EXPECT_EQ(rate.size(1), 2);
  EXPECT_GT(rate[0][0].item<double>(), 0.0);
  EXPECT_GT(rate[0][1].item<double>(), 0.0);
}

// ============================================================================
// Section 4: Integration with Kinetics
// ============================================================================

TEST_P(FalloffEndToEndTest, kinetics_integration) {
  std::string yaml_str = R"(
species:
  - name: H2O2
    composition: {H: 2, O: 2}
    cv_R: 2.5
  - name: O
    composition: {O: 1}
    cv_R: 2.5
  - name: H2O
    composition: {H: 2, O: 1}
    cv_R: 2.5
  - name: OH
    composition: {H: 1, O: 1}
    cv_R: 2.5
  - name: AR
    composition: {Ar: 1}
    cv_R: 2.5
  - name: H2
    composition: {H: 2}
    cv_R: 2.5
reference-state:
  Tref: 300.0
  Pref: 101325.0
reactions:
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
  efficiencies: {AR: 0.83, H2: 2.4}
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
)";

  YAML::Node config = YAML::Load(yaml_str);
  auto kinet_opts = KineticsOptionsImpl::from_yaml(config);
  Kinetics kinet(kinet_opts);
  kinet->to(device, dtype);

  EXPECT_EQ(kinet_opts->three_body()->reactions().size(), 1);
  EXPECT_EQ(kinet_opts->lindemann_falloff()->reactions().size(), 1);
  EXPECT_EQ(kinet->stoich.size(1), 2);

  auto temp = torch::tensor({300.0}, dtype).to(device);
  auto pres = torch::tensor({101325.0}, dtype).to(device);
  auto species = kinet_opts->species();
  int nspecies = species.size();
  auto conc = torch::ones({nspecies}, dtype).to(device) * 1e-3;

  auto [rate, rc_ddC, rc_ddT] = kinet->forward(temp, pres, conc);

  EXPECT_EQ(rate.size(0), 1);
  EXPECT_EQ(rate.size(1), 2);
  EXPECT_GT(rate[0][0].item<double>(), 0.0);
  EXPECT_GT(rate[0][1].item<double>(), 0.0);
}

// ============================================================================
// Section 5: Jacobian (Autograd Derivatives)
// ============================================================================

TEST_P(FalloffEndToEndTest, jacobian_three_body_concentration) {
  std::string yaml_str = R"(
species:
  - name: H2O2
    composition: {H: 2, O: 2}
    cv_R: 2.5
  - name: O
    composition: {O: 1}
    cv_R: 2.5
  - name: H2O
    composition: {H: 2, O: 1}
    cv_R: 2.5
  - name: AR
    composition: {Ar: 1}
    cv_R: 2.5
  - name: H2
    composition: {H: 2}
    cv_R: 2.5
reference-state:
  Tref: 300.0
  Pref: 101325.0
reactions:
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
  efficiencies: {AR: 0.83, H2: 2.4}
)";

  YAML::Node config = YAML::Load(yaml_str);
  auto kinet_opts = KineticsOptionsImpl::from_yaml(config);
  Kinetics kinet(kinet_opts);
  kinet->to(device, dtype);

  auto temp = torch::tensor({300.0}, dtype).to(device);
  auto pres = torch::tensor({101325.0}, dtype).to(device);
  auto species = kinet_opts->species();
  int nspecies = species.size();
  auto conc = torch::ones({nspecies}, dtype).to(device) * 1e-3;

  auto [rate, rc_ddC, rc_ddT] = kinet->forward(temp, pres, conc);
  auto cvol = torch::tensor({1.0}, dtype).to(device);
  auto raw_jac = kinet->jacobian(temp, conc, cvol, rate, rc_ddC, rc_ddT);

  // Raw jacobian has shape (batch..., nreaction, nspecies).
  // Full species-to-species jacobian: stoich @ raw_jac -> (batch..., nspecies, nspecies)
  auto full_jac = kinet->stoich.matmul(raw_jac);

  int ar_idx = -1, h2_idx = -1, h2o2_idx = -1;
  for (int i = 0; i < nspecies; i++) {
    if (species[i] == "AR") ar_idx = i;
    if (species[i] == "H2") h2_idx = i;
    if (species[i] == "H2O2") h2o2_idx = i;
  }
  ASSERT_GE(ar_idx, 0);
  ASSERT_GE(h2_idx, 0);
  ASSERT_GE(h2o2_idx, 0);

  // Squeeze batch dimension: (1, nspecies, nspecies) -> (nspecies, nspecies)
  auto jac_2d = full_jac.squeeze(0);
  ASSERT_EQ(jac_2d.dim(), 2);
  ASSERT_EQ(jac_2d.size(0), nspecies);
  ASSERT_EQ(jac_2d.size(1), nspecies);

  // Third-body species should have non-zero derivatives
  EXPECT_GT(std::abs(jac_2d[h2o2_idx][ar_idx].item<double>()), 1e-10);
  EXPECT_GT(std::abs(jac_2d[h2o2_idx][h2_idx].item<double>()), 1e-10);

  // H2 efficiency (2.4) > AR efficiency (0.83), so derivative should be larger
  EXPECT_GT(std::abs(jac_2d[h2o2_idx][h2_idx].item<double>()),
            std::abs(jac_2d[h2o2_idx][ar_idx].item<double>()));
}

TEST_P(FalloffEndToEndTest, jacobian_falloff_lindemann) {
  std::string yaml_str = R"(
species:
  - name: OH
    composition: {H: 1, O: 1}
    cv_R: 2.5
  - name: H2O2
    composition: {H: 2, O: 2}
    cv_R: 2.5
  - name: AR
    composition: {Ar: 1}
    cv_R: 2.5
  - name: N2
    composition: {N: 2}
    cv_R: 2.5
reference-state:
  Tref: 300.0
  Pref: 101325.0
reactions:
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
  efficiencies: {AR: 0.7, N2: 1.0}
)";

  YAML::Node config = YAML::Load(yaml_str);
  auto kinet_opts = KineticsOptionsImpl::from_yaml(config);
  Kinetics kinet(kinet_opts);
  kinet->to(device, dtype);

  auto temp = torch::tensor({300.0}, dtype).to(device);
  auto pres = torch::tensor({101325.0}, dtype).to(device);
  auto species = kinet_opts->species();
  int nspecies = species.size();
  // Use very small concentrations to keep reduced pressure Pr moderate
  // (k0/kinf ratio is ~1e19 after unit conversion, so need conc ~ 1e-20 for Pr ~ 1)
  auto conc = torch::ones({nspecies}, dtype).to(device) * 1e-21;

  auto [rate, rc_ddC, rc_ddT] = kinet->forward(temp, pres, conc);
  auto cvol = torch::tensor({1.0}, dtype).to(device);
  auto raw_jac = kinet->jacobian(temp, conc, cvol, rate, rc_ddC, rc_ddT);

  // Raw jacobian has shape (batch..., nreaction, nspecies).
  // Full species-to-species jacobian: stoich @ raw_jac -> (batch..., nspecies, nspecies)
  auto full_jac = kinet->stoich.matmul(raw_jac);

  int ar_idx = -1, n2_idx = -1, h2o2_idx = -1;
  for (int i = 0; i < nspecies; i++) {
    if (species[i] == "AR") ar_idx = i;
    if (species[i] == "N2") n2_idx = i;
    if (species[i] == "H2O2") h2o2_idx = i;
  }
  ASSERT_GE(ar_idx, 0);
  ASSERT_GE(n2_idx, 0);
  ASSERT_GE(h2o2_idx, 0);

  // Squeeze batch dimension: (1, nspecies, nspecies) -> (nspecies, nspecies)
  auto jac_2d = full_jac.squeeze(0);
  ASSERT_EQ(jac_2d.dim(), 2);
  ASSERT_EQ(jac_2d.size(0), nspecies);
  ASSERT_EQ(jac_2d.size(1), nspecies);

  // Derivatives should be non-zero
  EXPECT_GT(std::abs(jac_2d[h2o2_idx][ar_idx].item<double>()), 1e-10);
  EXPECT_GT(std::abs(jac_2d[h2o2_idx][n2_idx].item<double>()), 1e-10);

  // N2 efficiency (1.0) > AR efficiency (0.7)
  EXPECT_GT(std::abs(jac_2d[h2o2_idx][n2_idx].item<double>()),
            std::abs(jac_2d[h2o2_idx][ar_idx].item<double>()));
}

TEST_P(FalloffEndToEndTest, jacobian_temperature_derivative) {
  std::string yaml_str = R"(
species:
  - name: OH
    composition: {H: 1, O: 1}
    cv_R: 2.5
  - name: H2O2
    composition: {H: 2, O: 2}
    cv_R: 2.5
  - name: AR
    composition: {Ar: 1}
    cv_R: 2.5
reference-state:
  Tref: 300.0
  Pref: 101325.0
reactions:
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
  Troe:
    A: 0.42
    T3: 1e-30
    T1: 1e30
    T2: 1e30
)";

  YAML::Node config = YAML::Load(yaml_str);
  auto kinet_opts = KineticsOptionsImpl::from_yaml(config);
  kinet_opts->evolve_temperature(true);
  Kinetics kinet(kinet_opts);
  kinet->to(device, dtype);

  auto temp = torch::tensor({300.0}, dtype).to(device);
  auto pres = torch::tensor({101325.0}, dtype).to(device);
  auto species = kinet_opts->species();
  int nspecies = species.size();
  auto conc = torch::ones({nspecies}, dtype).to(device) * 1e-3;

  auto [rate, rc_ddC, rc_ddT] = kinet->forward(temp, pres, conc);

  ASSERT_TRUE(rc_ddT.has_value());
  EXPECT_GT(rc_ddT.value().numel(), 0);
  EXPECT_GT(std::abs(rc_ddT.value()[0][0].item<double>()), 1e-10);
}

TEST_P(FalloffEndToEndTest, jacobian_mixed_reaction_types) {
  std::string yaml_str = R"(
species:
  - name: H2O2
    composition: {H: 2, O: 2}
    cv_R: 2.5
  - name: O
    composition: {O: 1}
    cv_R: 2.5
  - name: H2O
    composition: {H: 2, O: 1}
    cv_R: 2.5
  - name: OH
    composition: {H: 1, O: 1}
    cv_R: 2.5
  - name: AR
    composition: {Ar: 1}
    cv_R: 2.5
reference-state:
  Tref: 300.0
  Pref: 101325.0
reactions:
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
  efficiencies: {AR: 0.83}
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
)";

  YAML::Node config = YAML::Load(yaml_str);
  auto kinet_opts = KineticsOptionsImpl::from_yaml(config);
  Kinetics kinet(kinet_opts);
  kinet->to(device, dtype);

  auto temp = torch::tensor({300.0}, dtype).to(device);
  auto pres = torch::tensor({101325.0}, dtype).to(device);
  auto species = kinet_opts->species();
  int nspecies = species.size();
  auto conc = torch::ones({nspecies}, dtype).to(device) * 1e-3;

  auto [rate, rc_ddC, rc_ddT] = kinet->forward(temp, pres, conc);
  auto cvol = torch::tensor({1.0}, dtype).to(device);
  auto raw_jac = kinet->jacobian(temp, conc, cvol, rate, rc_ddC, rc_ddT);

  // rate shape: (1, nreaction) where nreaction = 2
  EXPECT_EQ(rate.size(-1), 2);
  // rc_ddC shape: (nspecies, nreaction) where nreaction = 2
  EXPECT_EQ(rc_ddC.size(-1), 2);

  // Raw jacobian has shape (batch..., nreaction, nspecies).
  // Full species-to-species jacobian: stoich @ raw_jac -> (batch..., nspecies, nspecies)
  auto full_jac = kinet->stoich.matmul(raw_jac);

  // Squeeze batch dimension
  auto jac_2d = full_jac.squeeze(0);
  ASSERT_EQ(jac_2d.dim(), 2);
  ASSERT_EQ(jac_2d.size(0), nspecies);
  ASSERT_EQ(jac_2d.size(1), nspecies);

  int ar_idx = -1, h2o2_idx = -1;
  for (int i = 0; i < nspecies; i++) {
    if (species[i] == "AR") ar_idx = i;
    if (species[i] == "H2O2") h2o2_idx = i;
  }
  ASSERT_GE(ar_idx, 0);
  ASSERT_GE(h2o2_idx, 0);

  // AR should affect both reactions
  EXPECT_GT(std::abs(jac_2d[h2o2_idx][ar_idx].item<double>()), 1e-10);
}

// ============================================================================
// Section 6: Real Chemistry Test (sample_reaction_full.yaml)
// ============================================================================

TEST_P(FalloffEndToEndTest, real_chemistry_sample_reaction_full) {
  // Create complete YAML with all species from sample_reaction_full.yaml
  std::string yaml_str = R"(
species:
  - name: O
    composition: {O: 1}
    cv_R: 2.5
  - name: H2
    composition: {H: 2}
    cv_R: 2.5
  - name: H
    composition: {H: 1}
    cv_R: 1.5
  - name: OH
    composition: {H: 1, O: 1}
    cv_R: 2.5
  - name: O2
    composition: {O: 2}
    cv_R: 2.5
  - name: HO2
    composition: {H: 1, O: 2}
    cv_R: 2.5
  - name: H2O2
    composition: {H: 2, O: 2}
    cv_R: 2.5
  - name: H2O
    composition: {H: 2, O: 1}
    cv_R: 2.5
  - name: AR
    composition: {Ar: 1}
    cv_R: 2.5
reference-state:
  Tref: 300.0
  Pref: 101325.0
reactions:
- equation: O + H2 <=> H + OH
  type: arrhenius
  rate-constant: {A: 38.7, b: 2.7, Ea_R: 6260.0}
- equation: H + 2 O2 => HO2 + O2
  type: arrhenius
  rate-constant: {A: 2.08e+19, b: -1.24, Ea_R: 0.0}
- equation: 0.7 H2 + 0.6 OH + 1.2 O2 => H2O2 + O
  type: arrhenius
  rate-constant: {A: 3.981072e+04, b: 0.0, Ea_R: 9.252008e+04}
  orders: {H2: 0.8, O2: 1.0, OH: 2.0}
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
  efficiencies: {AR: 0.83, H2: 2.4, H2O: 15.4}
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: -1700.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
  efficiencies: {AR: 0.7, H2: 2.0, H2O: 6.0}
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: -1700.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
  Troe: {A: 0.51, T3: 1.000e-30, T1: 1.000e+30}
  efficiencies: {AR: 0.3, H2: 1.5, H2O: 2.7}
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: -1700.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
  Troe: {A: 0.7346, T3: 94.0, T1: 1756.0, T2: 5182.0}
  efficiencies: {AR: 0.7, H2: 2.0, H2O: 6.0}
- equation: O + H2 (+ M) <=> H + OH (+ M)
  type: falloff
  high-P-rate-constant: {A: 1.0e+15, b: -2.0, Ea: 1000.0 cal/mol}
  low-P-rate-constant: {A: 4.0e+19, b: -3.0, Ea: 0.0 cal/mol}
  SRI: {A: 0.54, B: 201.0, C: 1024.0}
- equation: H + HO2 (+ M) <=> H2 + O2 (+ M)
  type: falloff
  high-P-rate-constant: {A: 4.0e+15, b: -0.5, Ea: 100.0 cal/mol}
  low-P-rate-constant: {A: 7.0e+20, b: -1.0, Ea: 0.0 cal/mol}
  efficiencies: {AR: 0.7, H2: 2.0, H2O: 6.0}
  SRI: {A: 1.1, B: 700.0, C: 1234.0, D: 56.0, E: 0.7}
)";

  YAML::Node config = YAML::Load(yaml_str);
  auto kinet_opts = KineticsOptionsImpl::from_yaml(config);
  Kinetics kinet(kinet_opts);
  kinet->to(device, dtype);

  // Verify all falloff reactions were parsed into appropriate modules
  EXPECT_EQ(kinet_opts->three_body()->reactions().size(), 1);
  EXPECT_EQ(kinet_opts->lindemann_falloff()->reactions().size(), 1);  // 1 Lindemann reaction (no Troe/SRI)
  EXPECT_EQ(kinet_opts->troe_falloff()->reactions().size(), 2);  // 2 Troe reactions
  EXPECT_EQ(kinet_opts->sri_falloff()->reactions().size(), 2);  // 2 SRI reactions

  // Test forward() with realistic conditions
  auto temp = torch::tensor({300.0, 500.0, 1000.0}, dtype).to(device);
  auto pres = torch::tensor({101325.0, 101325.0, 101325.0}, dtype).to(device);
  auto species = kinet_opts->species();
  int nspecies = species.size();
  
  // Create realistic concentration profile (mole fractions)
  auto conc = torch::zeros({3, nspecies}, dtype).to(device);
  int o_idx = -1, h2_idx = -1, h_idx = -1, oh_idx = -1, o2_idx = -1, 
      ho2_idx = -1, h2o2_idx = -1, h2o_idx = -1, ar_idx = -1;
  for (int i = 0; i < nspecies; i++) {
    if (species[i] == "O") o_idx = i;
    if (species[i] == "H2") h2_idx = i;
    if (species[i] == "H") h_idx = i;
    if (species[i] == "OH") oh_idx = i;
    if (species[i] == "O2") o2_idx = i;
    if (species[i] == "HO2") ho2_idx = i;
    if (species[i] == "H2O2") h2o2_idx = i;
    if (species[i] == "H2O") h2o_idx = i;
    if (species[i] == "AR") ar_idx = i;
  }
  
  // Set realistic initial concentrations (mole fractions)
  conc.select(1, o2_idx) = 0.21;   // Air-like
  conc.select(1, ar_idx) = 0.79;   // Argon
  conc.select(1, h2_idx) = 0.01;   // Small H2
  conc.select(1, h2o_idx) = 0.01;  // Small H2O
  conc.select(1, oh_idx) = 1e-6;   // Trace OH
  conc.select(1, h2o2_idx) = 1e-8; // Trace H2O2
  
  // Convert to concentration (mol/m^3) - approximate: P/(RT) * mole_fraction
  auto R = 8.314;  // J/(mol·K)
  auto n_tot = (pres / (R * temp)).unsqueeze(-1);  // (batch, 1) mol/m^3
  conc = conc * n_tot;

  auto [rate, rc_ddC, rc_ddT] = kinet->forward(temp, pres, conc);

  // Verify output shapes
  EXPECT_EQ(rate.size(0), 3);  // 3 temperature points
  EXPECT_GT(rate.size(1), 0);  // At least some reactions
  
  // All rates should be non-negative
  EXPECT_TRUE(torch::all(rate >= 0.0).item<bool>());
  
  int nreaction = rate.size(1);
  // At least one reaction should have a positive rate
  EXPECT_GT(torch::sum(rate[0]).item<double>(), 0.0);
  
  // Test that rates vary with temperature
  auto rate_300K = rate[0];
  auto rate_1000K = rate[2];
  EXPECT_FALSE(torch::allclose(rate_300K, rate_1000K));

  // Test Jacobian computation
  auto cvol = torch::ones({3}, dtype).to(device);
  auto raw_jac = kinet->jacobian(temp, conc, cvol, rate, rc_ddC, rc_ddT);

  // Raw jacobian: (batch, nreaction, nspecies)
  EXPECT_EQ(raw_jac.size(0), 3);  // 3 temperature points
  EXPECT_EQ(raw_jac.size(1), nreaction);
  EXPECT_EQ(raw_jac.size(2), nspecies);

  // Full species-to-species jacobian: stoich @ raw_jac -> (batch, nspecies, nspecies)
  auto full_jac = kinet->stoich.matmul(raw_jac);
  EXPECT_EQ(full_jac.size(0), 3);
  EXPECT_EQ(full_jac.size(1), nspecies);
  EXPECT_EQ(full_jac.size(2), nspecies);

  // Test that third-body species (AR) affects reaction rates
  if (ar_idx >= 0 && h2o2_idx >= 0) {
    // AR should affect H2O2 consumption in three-body reaction
    EXPECT_GT(std::abs(full_jac[0][h2o2_idx][ar_idx].item<double>()), 1e-10);
  }
}

INSTANTIATE_TEST_SUITE_P(
    DeviceTests, FalloffEndToEndTest,
    testing::Values(Parameters{torch::kCPU, torch::kFloat64},
                    Parameters{torch::kCUDA, torch::kFloat64}),
    [](const testing::TestParamInfo<FalloffEndToEndTest::ParamType>& info) {
      std::string name = torch::Device(info.param.device_type).str();
      name += "_";
      name += torch::toString(info.param.dtype);
      std::replace(name.begin(), name.end(), '.', '_');
      return name;
    });
