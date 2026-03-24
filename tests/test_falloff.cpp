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
// Section 3: Rate Calculations (Forward) — standalone module tests
// ============================================================================

// Helper: build a simple stoich matrix from a Reaction for a module with 1 reaction
static torch::Tensor make_stoich(const Reaction& rxn, int nspecies) {
  auto stoich = torch::zeros({nspecies, 1}, torch::kFloat64);
  for (int i = 0; i < nspecies; i++) {
    auto it = rxn.reactants().find(kintera::species_names[i]);
    if (it != rxn.reactants().end()) {
      stoich[i][0] = -it->second;
    }
    it = rxn.products().find(kintera::species_names[i]);
    if (it != rxn.products().end()) {
      stoich[i][0] = it->second;
    }
  }
  return stoich;
}

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
  int nspecies = kintera::species_names.size();
  module->set_stoich(make_stoich(opts->reactions()[0], nspecies));
  module->to(device, dtype);

  auto temp = torch::tensor({300.0}, dtype).to(device);
  auto w = torch::ones({nspecies}, dtype).to(device) * 1e-3;
  auto du = torch::zeros({nspecies}, dtype).to(device);

  auto result = module->forward(du, w, temp, 1.0);

  // du should have non-zero species tendencies
  EXPECT_GT(torch::sum(torch::abs(result)).item<double>(), 0.0);
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
  int nspecies = kintera::species_names.size();
  module->set_stoich(make_stoich(opts->reactions()[0], nspecies));
  module->to(device, dtype);

  auto temp = torch::tensor({300.0}, dtype).to(device);
  auto w = torch::ones({nspecies}, dtype).to(device) * 1e-3;
  auto du = torch::zeros({nspecies}, dtype).to(device);

  auto result = module->forward(du, w, temp, 1.0);

  EXPECT_GT(torch::sum(torch::abs(result)).item<double>(), 0.0);
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
  int nspecies = kintera::species_names.size();
  module->set_stoich(make_stoich(opts->reactions()[0], nspecies));
  module->to(device, dtype);

  auto temp = torch::tensor({300.0}, dtype).to(device);
  auto w = torch::ones({nspecies}, dtype).to(device) * 1e-3;
  auto du = torch::zeros({nspecies}, dtype).to(device);

  auto result = module->forward(du, w, temp, 1.0);

  EXPECT_GT(torch::sum(torch::abs(result)).item<double>(), 0.0);
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
  int nspecies = kintera::species_names.size();
  module->set_stoich(make_stoich(opts->reactions()[0], nspecies));
  module->to(device, dtype);

  auto temp = torch::tensor({300.0}, dtype).to(device);
  auto w = torch::ones({nspecies}, dtype).to(device) * 1e-3;
  auto du = torch::zeros({nspecies}, dtype).to(device);

  auto result = module->forward(du, w, temp, 1.0);

  EXPECT_GT(torch::sum(torch::abs(result)).item<double>(), 0.0);
}

TEST_P(FalloffEndToEndTest, forward_multiple_reactions) {
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
  int nspecies = kintera::species_names.size();
  // 2 reactions, same stoich for both
  auto stoich = torch::zeros({nspecies, 2}, torch::kFloat64);
  for (int j = 0; j < 2; j++) {
    auto rxn = opts->reactions()[j];
    for (int i = 0; i < nspecies; i++) {
      auto it = rxn.reactants().find(kintera::species_names[i]);
      if (it != rxn.reactants().end()) stoich[i][j] = -it->second;
      it = rxn.products().find(kintera::species_names[i]);
      if (it != rxn.products().end()) stoich[i][j] = it->second;
    }
  }
  module->set_stoich(stoich);
  module->to(device, dtype);

  auto temp = torch::tensor({300.0}, dtype).to(device);
  auto w = torch::ones({nspecies}, dtype).to(device) * 1e-3;
  auto du = torch::zeros({nspecies}, dtype).to(device);

  auto result = module->forward(du, w, temp, 1.0);

  // Both reactions should contribute to du
  EXPECT_GT(torch::sum(torch::abs(result)).item<double>(), 0.0);
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
  auto species = kinet_opts->species();
  int nspecies = species.size();
  auto w = torch::ones({nspecies}, dtype).to(device) * 1e-3;
  auto du = torch::zeros({nspecies}, dtype).to(device);

  auto result = kinet->forward(du, w, temp, 1.0);

  // Species tendencies should be non-zero
  EXPECT_GT(torch::sum(torch::abs(result)).item<double>(), 0.0);
}

// ============================================================================
// Section 5: Jacobian via Autograd
// ============================================================================

// Helper: compute the species Jacobian d(du)/d(w) via autograd
static torch::Tensor compute_jacobian(Kinetics& kinet, torch::Tensor w,
                                       torch::Tensor temp, double dt) {
  int nspecies = w.size(-1);
  auto w_ad = w.clone().requires_grad_(true);
  auto du = torch::zeros_like(w_ad);
  auto result = kinet->forward(du, w_ad, temp, dt);

  auto jac = torch::zeros({nspecies, nspecies}, w.options());
  for (int i = 0; i < nspecies; i++) {
    if (w_ad.grad().defined()) w_ad.grad().zero_();
    // Use .select(-1, i).sum() to reduce to scalar for backward()
    result.select(-1, i).sum().backward(/*grad_tensors=*/{}, /*retain_graph=*/true);
    jac[i] = w_ad.grad().clone();
  }
  return jac;
}

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
  auto species = kinet_opts->species();
  int nspecies = species.size();
  auto w = torch::ones({nspecies}, dtype).to(device) * 1e-3;

  auto jac = compute_jacobian(kinet, w, temp, 1.0);

  int ar_idx = -1, h2_idx = -1, h2o2_idx = -1;
  for (int i = 0; i < nspecies; i++) {
    if (species[i] == "AR") ar_idx = i;
    if (species[i] == "H2") h2_idx = i;
    if (species[i] == "H2O2") h2o2_idx = i;
  }
  ASSERT_GE(ar_idx, 0);
  ASSERT_GE(h2_idx, 0);
  ASSERT_GE(h2o2_idx, 0);

  // Third-body species should have non-zero derivatives
  EXPECT_GT(std::abs(jac[h2o2_idx][ar_idx].item<double>()), 1e-10);
  EXPECT_GT(std::abs(jac[h2o2_idx][h2_idx].item<double>()), 1e-10);

  // H2 efficiency (2.4) > AR efficiency (0.83), so derivative should be larger
  EXPECT_GT(std::abs(jac[h2o2_idx][h2_idx].item<double>()),
            std::abs(jac[h2o2_idx][ar_idx].item<double>()));
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
  auto species = kinet_opts->species();
  int nspecies = species.size();
  // Use very small concentrations for moderate Pr
  auto w = torch::ones({nspecies}, dtype).to(device) * 1e-21;

  auto jac = compute_jacobian(kinet, w, temp, 1.0);

  int ar_idx = -1, n2_idx = -1, h2o2_idx = -1;
  for (int i = 0; i < nspecies; i++) {
    if (species[i] == "AR") ar_idx = i;
    if (species[i] == "N2") n2_idx = i;
    if (species[i] == "H2O2") h2o2_idx = i;
  }
  ASSERT_GE(ar_idx, 0);
  ASSERT_GE(n2_idx, 0);
  ASSERT_GE(h2o2_idx, 0);

  // Derivatives should be non-zero
  EXPECT_GT(std::abs(jac[h2o2_idx][ar_idx].item<double>()), 1e-10);
  EXPECT_GT(std::abs(jac[h2o2_idx][n2_idx].item<double>()), 1e-10);

  // N2 efficiency (1.0) > AR efficiency (0.7)
  EXPECT_GT(std::abs(jac[h2o2_idx][n2_idx].item<double>()),
            std::abs(jac[h2o2_idx][ar_idx].item<double>()));
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
  auto species = kinet_opts->species();
  int nspecies = species.size();
  auto w = torch::ones({nspecies}, dtype).to(device) * 1e-3;

  auto jac = compute_jacobian(kinet, w, temp, 1.0);

  ASSERT_EQ(jac.size(0), nspecies);
  ASSERT_EQ(jac.size(1), nspecies);

  int ar_idx = -1, h2o2_idx = -1;
  for (int i = 0; i < nspecies; i++) {
    if (species[i] == "AR") ar_idx = i;
    if (species[i] == "H2O2") h2o2_idx = i;
  }
  ASSERT_GE(ar_idx, 0);
  ASSERT_GE(h2o2_idx, 0);

  // AR should affect H2O2 through both reactions
  EXPECT_GT(std::abs(jac[h2o2_idx][ar_idx].item<double>()), 1e-10);
}

// ============================================================================
// Section 6: Real Chemistry Test
// ============================================================================

TEST_P(FalloffEndToEndTest, real_chemistry_sample_reaction_full) {
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
  EXPECT_EQ(kinet_opts->lindemann_falloff()->reactions().size(), 1);
  EXPECT_EQ(kinet_opts->troe_falloff()->reactions().size(), 2);
  EXPECT_EQ(kinet_opts->sri_falloff()->reactions().size(), 2);

  // Test forward() with a single temperature point
  auto temp = torch::tensor({500.0}, dtype).to(device);
  auto species = kinet_opts->species();
  int nspecies = species.size();
  
  auto w = torch::ones({nspecies}, dtype).to(device) * 1e-3;
  auto du = torch::zeros({nspecies}, dtype).to(device);

  auto result = kinet->forward(du, w, temp, 1.0);

  // Species tendencies should be non-zero
  EXPECT_GT(torch::sum(torch::abs(result)).item<double>(), 0.0);

  // Test Jacobian via autograd
  auto jac = compute_jacobian(kinet, w, temp, 1.0);
  EXPECT_EQ(jac.size(0), nspecies);
  EXPECT_EQ(jac.size(1), nspecies);

  // Jacobian should have non-zero entries
  EXPECT_GT(torch::sum(torch::abs(jac)).item<double>(), 0.0);
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
