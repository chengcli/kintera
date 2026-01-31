// C/C++
#include <algorithm>

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/kinetics/falloff.hpp>
#include <kintera/kintera_formatter.hpp>
#include <kintera/species.hpp>

using namespace kintera;

class FalloffForwardTest : public testing::Test {
 protected:
  void SetUp() override {
    // Initialize species names for testing
    kintera::species_names = {"H2O2", "O", "H2O", "OH", "AR", "H2", "N2", "O2", "H"};
  }
};

TEST_F(FalloffForwardTest, module_creation) {
  auto opts = FalloffOptionsImpl::create();
  Falloff module(opts);

  // Verify module was created successfully
  EXPECT_NE(opts, nullptr);
}

TEST_F(FalloffForwardTest, reset_with_empty_options) {
  auto opts = FalloffOptionsImpl::create();
  Falloff module(opts);

  // Reset should handle empty options gracefully
  module->reset();
  EXPECT_EQ(opts->reactions().size(), 0);
}

TEST_F(FalloffForwardTest, efficiency_matrix_construction) {
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
  auto opts = FalloffOptionsImpl::from_yaml(root);
  Falloff module(opts);

  // Check efficiency matrix shape: (nreaction, nspecies)
  EXPECT_EQ(module->efficiency_matrix.size(0), 2);  // 2 reactions
  EXPECT_EQ(module->efficiency_matrix.size(1), 9);  // 9 species

  // Reaction 0: AR=0.83, H2=2.4, H2O=15.4, others=1.0
  auto eff0 = module->efficiency_matrix[0];
  int ar_idx = std::find(kintera::species_names.begin(), kintera::species_names.end(), "AR") - kintera::species_names.begin();
  int h2_idx = std::find(kintera::species_names.begin(), kintera::species_names.end(), "H2") - kintera::species_names.begin();
  int h2o_idx = std::find(kintera::species_names.begin(), kintera::species_names.end(), "H2O") - kintera::species_names.begin();

  EXPECT_DOUBLE_EQ(eff0[ar_idx].item<double>(), 0.83);
  EXPECT_DOUBLE_EQ(eff0[h2_idx].item<double>(), 2.4);
  EXPECT_DOUBLE_EQ(eff0[h2o_idx].item<double>(), 15.4);

  // Check that other species have efficiency 1.0
  for (int i = 0; i < eff0.size(0); i++) {
    if (i != ar_idx && i != h2_idx && i != h2o_idx) {
      EXPECT_DOUBLE_EQ(eff0[i].item<double>(), 1.0);
    }
  }

  // Reaction 1: AR=0.7, H2=2.0, others=1.0
  auto eff1 = module->efficiency_matrix[1];
  EXPECT_DOUBLE_EQ(eff1[ar_idx].item<double>(), 0.7);
  EXPECT_DOUBLE_EQ(eff1[h2_idx].item<double>(), 2.0);
  // Check that H2O has efficiency 1.0 (not specified)
  EXPECT_DOUBLE_EQ(eff1[h2o_idx].item<double>(), 1.0);
}

TEST_F(FalloffForwardTest, falloff_type_flags) {
  std::string yaml_str = R"(
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
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

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = FalloffOptionsImpl::from_yaml(root);
  Falloff module(opts);

  ASSERT_EQ(module->falloff_type_flags.size(0), 4);

  // Reaction 0: None (three-body)
  EXPECT_EQ(module->falloff_type_flags[0].item<int>(), static_cast<int>(FalloffType::None));
  // Reaction 1: None (Lindemann)
  EXPECT_EQ(module->falloff_type_flags[1].item<int>(), static_cast<int>(FalloffType::None));
  // Reaction 2: Troe
  EXPECT_EQ(module->falloff_type_flags[2].item<int>(), static_cast<int>(FalloffType::Troe));
  // Reaction 3: SRI
  EXPECT_EQ(module->falloff_type_flags[3].item<int>(), static_cast<int>(FalloffType::SRI));
}

TEST_F(FalloffForwardTest, is_three_body_flags) {
  std::string yaml_str = R"(
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = FalloffOptionsImpl::from_yaml(root);
  Falloff module(opts);

  ASSERT_EQ(module->is_three_body.size(0), 2);

  // Reaction 0: three-body
  EXPECT_EQ(module->is_three_body[0].item<int>(), 1);
  // Reaction 1: falloff (not three-body)
  EXPECT_EQ(module->is_three_body[1].item<int>(), 0);
}

TEST_F(FalloffForwardTest, kinf_values_for_three_body) {
  std::string yaml_str = R"(
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = FalloffOptionsImpl::from_yaml(root);
  Falloff module(opts);

  // Reaction 0: k_inf should be very large (1e100) for three-body
  EXPECT_GT(module->kinf_A[0].item<double>(), 1e90);

  // Reaction 1: k_inf should have reasonable value (not 1e100)
  EXPECT_LT(module->kinf_A[1].item<double>(), 1e90);
  EXPECT_GT(module->kinf_A[1].item<double>(), 0.0);
}

TEST_F(FalloffForwardTest, all_tensors_registered) {
  std::string yaml_str = R"(
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = FalloffOptionsImpl::from_yaml(root);
  Falloff module(opts);

  // Verify all tensors are registered and have correct shapes
  int nreaction = 1;
  EXPECT_EQ(module->k0_A.size(0), nreaction);
  EXPECT_EQ(module->k0_b.size(0), nreaction);
  EXPECT_EQ(module->k0_Ea_R.size(0), nreaction);
  EXPECT_EQ(module->kinf_A.size(0), nreaction);
  EXPECT_EQ(module->kinf_b.size(0), nreaction);
  EXPECT_EQ(module->kinf_Ea_R.size(0), nreaction);
  EXPECT_EQ(module->falloff_type_flags.size(0), nreaction);
  EXPECT_EQ(module->is_three_body.size(0), nreaction);
  EXPECT_EQ(module->troe_A.size(0), nreaction);
  EXPECT_EQ(module->troe_T3.size(0), nreaction);
  EXPECT_EQ(module->troe_T1.size(0), nreaction);
  EXPECT_EQ(module->troe_T2.size(0), nreaction);
  EXPECT_EQ(module->sri_A.size(0), nreaction);
  EXPECT_EQ(module->sri_B.size(0), nreaction);
  EXPECT_EQ(module->sri_C.size(0), nreaction);
  EXPECT_EQ(module->sri_D.size(0), nreaction);
  EXPECT_EQ(module->sri_E.size(0), nreaction);
}

TEST_F(FalloffForwardTest, forward_efficiency_matrix_test) {
  // Test that efficiency matrix is used correctly in forward()
  std::string yaml_str = R"(
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
  efficiencies: {AR: 0.83, H2: 2.4, H2O: 15.4}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = FalloffOptionsImpl::from_yaml(root);
  Falloff module(opts);

  // Create concentration tensor with different values
  auto C = torch::ones({9}, torch::kFloat64);
  int ar_idx = std::find(kintera::species_names.begin(), kintera::species_names.end(), "AR") - kintera::species_names.begin();
  int h2_idx = std::find(kintera::species_names.begin(), kintera::species_names.end(), "H2") - kintera::species_names.begin();
  int h2o_idx = std::find(kintera::species_names.begin(), kintera::species_names.end(), "H2O") - kintera::species_names.begin();

  C[ar_idx] = 2.0;
  C[h2_idx] = 3.0;
  C[h2o_idx] = 4.0;

  auto T = torch::tensor({300.0}, torch::kFloat64);
  auto P = torch::tensor({101325.0}, torch::kFloat64);
  std::map<std::string, torch::Tensor> other;

  auto rate1 = module->forward(T, P, C, other);

  // Change concentrations - rate should change
  C[ar_idx] = 10.0;
  auto rate2 = module->forward(T, P, C, other);

  // Rate should be different (higher AR concentration should increase rate for three-body)
  EXPECT_GT(rate2[0].item<double>(), rate1[0].item<double>());
}

TEST_F(FalloffForwardTest, forward_three_body) {
  std::string yaml_str = R"(
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
  efficiencies: {AR: 0.83, H2: 2.4}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = FalloffOptionsImpl::from_yaml(root);
  Falloff module(opts);

  auto T = torch::tensor({300.0}, torch::kFloat64);
  auto P = torch::tensor({101325.0}, torch::kFloat64);
  auto C = torch::ones({9}, torch::kFloat64);
  std::map<std::string, torch::Tensor> other;

  auto rate = module->forward(T, P, C, other);

  EXPECT_EQ(rate.size(0), 1);  // 1 reaction

  // For three-body: k = k0 * [M]_eff
  // k0 at 300K ≈ 1.2e11 * (300/300)^(-1.0) = 1.2e11
  // [M]_eff ≈ sum of all concentrations (default efficiency 1.0)
  // Should be positive and non-zero
  EXPECT_GT(rate[0].item<double>(), 0.0);
}

TEST_F(FalloffForwardTest, forward_falloff_lindemann) {
  std::string yaml_str = R"(
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = FalloffOptionsImpl::from_yaml(root);
  Falloff module(opts);

  auto T = torch::tensor({300.0}, torch::kFloat64);
  auto P = torch::tensor({101325.0}, torch::kFloat64);
  auto C = torch::ones({9}, torch::kFloat64);
  std::map<std::string, torch::Tensor> other;

  auto rate = module->forward(T, P, C, other);

  EXPECT_EQ(rate.size(0), 1);
  // Rate should be positive
  EXPECT_GT(rate[0].item<double>(), 0.0);
}

TEST_F(FalloffForwardTest, forward_falloff_troe) {
  std::string yaml_str = R"(
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
  Troe: {A: 0.7346, T3: 94.0, T1: 1756.0, T2: 5182.0}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = FalloffOptionsImpl::from_yaml(root);
  Falloff module(opts);

  auto T = torch::tensor({300.0}, torch::kFloat64);
  auto P = torch::tensor({101325.0}, torch::kFloat64);
  auto C = torch::ones({9}, torch::kFloat64);
  std::map<std::string, torch::Tensor> other;

  auto rate = module->forward(T, P, C, other);

  EXPECT_EQ(rate.size(0), 1);
  EXPECT_GT(rate[0].item<double>(), 0.0);
}

TEST_F(FalloffForwardTest, forward_falloff_sri) {
  std::string yaml_str = R"(
- equation: O + H2 (+ M) <=> H + OH (+ M)
  type: falloff
  high-P-rate-constant: {A: 1.0e+15, b: -2.0, Ea_R: 0.0}
  low-P-rate-constant: {A: 4.0e+19, b: -3.0, Ea_R: 0.0}
  SRI: {A: 0.54, B: 201.0, C: 1024.0}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = FalloffOptionsImpl::from_yaml(root);
  Falloff module(opts);

  auto T = torch::tensor({300.0}, torch::kFloat64);
  auto P = torch::tensor({101325.0}, torch::kFloat64);
  auto C = torch::ones({9}, torch::kFloat64);
  std::map<std::string, torch::Tensor> other;

  auto rate = module->forward(T, P, C, other);

  EXPECT_EQ(rate.size(0), 1);
  EXPECT_GT(rate[0].item<double>(), 0.0);
}

TEST_F(FalloffForwardTest, forward_multiple_reactions) {
  // Test multiple reactions with different falloff types
  std::string yaml_str = R"(
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
  Troe: {A: 0.7346, T3: 94.0, T1: 1756.0, T2: 5182.0}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = FalloffOptionsImpl::from_yaml(root);
  Falloff module(opts);

  auto T = torch::tensor({300.0}, torch::kFloat64);
  auto P = torch::tensor({101325.0}, torch::kFloat64);
  auto C = torch::ones({9}, torch::kFloat64);
  std::map<std::string, torch::Tensor> other;

  auto rate = module->forward(T, P, C, other);

  // T has shape (1,), so rate has shape (1, nreaction) = (1, 3)
  EXPECT_EQ(rate.size(-1), 3);  // 3 reactions (last dimension)
  // All rates should be positive - access as rate[0][i] since first dim is 1
  EXPECT_GT(rate[0][0].item<double>(), 0.0);
  EXPECT_GT(rate[0][1].item<double>(), 0.0);
  EXPECT_GT(rate[0][2].item<double>(), 0.0);
  
  // Three-body and Lindemann should be different (different formulas)
  EXPECT_NE(rate[0][0].item<double>(), rate[0][1].item<double>());
}
