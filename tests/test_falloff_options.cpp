// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/kinetics/falloff.hpp>
#include <kintera/kintera_formatter.hpp>

using namespace kintera;

TEST(FalloffOptionsTest, default_options) {
  auto options = FalloffOptionsImpl::create();
  EXPECT_TRUE(options->reactions().empty());
  EXPECT_TRUE(options->k0_A().empty());
  EXPECT_TRUE(options->kinf_A().empty());
  EXPECT_TRUE(options->falloff_types().empty());
  EXPECT_TRUE(options->is_three_body().empty());
  EXPECT_DOUBLE_EQ(options->Tref(), 300.0);
  EXPECT_EQ(options->units(), "molecule,cm,s");
}

TEST(FalloffOptionsTest, simple_three_body) {
  // Reaction 4 from sample_reaction_full.yaml
  std::string yaml_str = R"(
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea: 0.0}
  efficiencies: {AR: 0.83, H2: 2.4, H2O: 15.4}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto options = FalloffOptionsImpl::from_yaml(root);

  ASSERT_EQ(options->reactions().size(), 1);
  // Note: equation() reconstructs from maps, order may differ
  // Just verify reactants and products are correct
  EXPECT_EQ(options->reactions()[0].reactants().size(), 2);
  EXPECT_EQ(options->reactions()[0].products().size(), 3);

  // Check it's marked as three-body
  ASSERT_EQ(options->is_three_body().size(), 1);
  EXPECT_TRUE(options->is_three_body()[0]);

  // Falloff type should be None (Lindemann)
  ASSERT_EQ(options->falloff_types().size(), 1);
  EXPECT_EQ(options->falloff_types()[0], static_cast<int>(FalloffType::None));

  // k_inf should be very large for three-body
  ASSERT_EQ(options->kinf_A().size(), 1);
  EXPECT_GT(options->kinf_A()[0], 1e90);

  // Check efficiencies
  ASSERT_EQ(options->efficiencies().size(), 1);
  const auto& eff = options->efficiencies()[0];
  EXPECT_EQ(eff.size(), 3);
  EXPECT_DOUBLE_EQ(eff.at("AR"), 0.83);
  EXPECT_DOUBLE_EQ(eff.at("H2"), 2.4);
  EXPECT_DOUBLE_EQ(eff.at("H2O"), 15.4);

  // Check reaction's falloff_type is set
  EXPECT_EQ(options->reactions()[0].falloff_type(), "none");
}

TEST(FalloffOptionsTest, falloff_lindemann) {
  // Reaction 5 from sample_reaction_full.yaml (Lindemann - no Troe/SRI)
  std::string yaml_str = R"(
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
  efficiencies: {AR: 0.7, H2: 2.0, H2O: 6.0}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto options = FalloffOptionsImpl::from_yaml(root);

  ASSERT_EQ(options->reactions().size(), 1);

  // Check it's NOT marked as three-body
  EXPECT_FALSE(options->is_three_body()[0]);

  // Falloff type should be None (Lindemann)
  EXPECT_EQ(options->falloff_types()[0], static_cast<int>(FalloffType::None));

  // k_inf should have a reasonable value (not 1e100)
  EXPECT_LT(options->kinf_A()[0], 1e90);
  EXPECT_GT(options->kinf_A()[0], 0);

  // Check k0 parameters
  EXPECT_GT(options->k0_A()[0], 0);
  EXPECT_DOUBLE_EQ(options->k0_b()[0], -0.9);

  // Check efficiencies
  const auto& eff = options->efficiencies()[0];
  EXPECT_EQ(eff.size(), 3);
  EXPECT_DOUBLE_EQ(eff.at("AR"), 0.7);
  EXPECT_DOUBLE_EQ(eff.at("H2"), 2.0);
  EXPECT_DOUBLE_EQ(eff.at("H2O"), 6.0);

  // Check reaction's falloff_type
  EXPECT_EQ(options->reactions()[0].falloff_type(), "none");
}

TEST(FalloffOptionsTest, falloff_troe_3param) {
  // Reaction 6 from sample_reaction_full.yaml (Troe 3-parameter)
  std::string yaml_str = R"(
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
  Troe: {A: 0.51, T3: 1.000e-30, T1: 1.000e+30}
  efficiencies: {AR: 0.3, H2: 1.5, H2O: 2.7}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto options = FalloffOptionsImpl::from_yaml(root);

  ASSERT_EQ(options->reactions().size(), 1);

  // Falloff type should be Troe
  EXPECT_EQ(options->falloff_types()[0], static_cast<int>(FalloffType::Troe));

  // Check Troe parameters
  EXPECT_DOUBLE_EQ(options->troe_A()[0], 0.51);
  EXPECT_DOUBLE_EQ(options->troe_T3()[0], 1.0e-30);
  EXPECT_DOUBLE_EQ(options->troe_T1()[0], 1.0e+30);
  EXPECT_DOUBLE_EQ(options->troe_T2()[0], 0.0);  // Not specified, should be 0

  // Check reaction's falloff_type
  EXPECT_EQ(options->reactions()[0].falloff_type(), "Troe");
}

TEST(FalloffOptionsTest, falloff_troe_4param) {
  // Reaction 7 from sample_reaction_full.yaml (Troe 4-parameter)
  std::string yaml_str = R"(
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
  Troe: {A: 0.7346, T3: 94.0, T1: 1756.0, T2: 5182.0}
  efficiencies: {AR: 0.7, H2: 2.0, H2O: 6.0}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto options = FalloffOptionsImpl::from_yaml(root);

  ASSERT_EQ(options->reactions().size(), 1);

  // Falloff type should be Troe
  EXPECT_EQ(options->falloff_types()[0], static_cast<int>(FalloffType::Troe));

  // Check Troe parameters (4-parameter)
  EXPECT_DOUBLE_EQ(options->troe_A()[0], 0.7346);
  EXPECT_DOUBLE_EQ(options->troe_T3()[0], 94.0);
  EXPECT_DOUBLE_EQ(options->troe_T1()[0], 1756.0);
  EXPECT_DOUBLE_EQ(options->troe_T2()[0], 5182.0);

  // Check reaction's falloff_type
  EXPECT_EQ(options->reactions()[0].falloff_type(), "Troe");
}

TEST(FalloffOptionsTest, falloff_sri_3param) {
  // Reaction 8 from sample_reaction_full.yaml (SRI 3-parameter)
  std::string yaml_str = R"(
- equation: O + H2 (+ M) <=> H + OH (+ M)
  type: falloff
  high-P-rate-constant: {A: 1.0e+15, b: -2.0, Ea_R: 0.0}
  low-P-rate-constant: {A: 4.0e+19, b: -3.0, Ea_R: 0.0}
  SRI: {A: 0.54, B: 201.0, C: 1024.0}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto options = FalloffOptionsImpl::from_yaml(root);

  ASSERT_EQ(options->reactions().size(), 1);

  // Falloff type should be SRI
  EXPECT_EQ(options->falloff_types()[0], static_cast<int>(FalloffType::SRI));

  // Check SRI parameters (3-parameter)
  EXPECT_DOUBLE_EQ(options->sri_A()[0], 0.54);
  EXPECT_DOUBLE_EQ(options->sri_B()[0], 201.0);
  EXPECT_DOUBLE_EQ(options->sri_C()[0], 1024.0);
  EXPECT_DOUBLE_EQ(options->sri_D()[0], 1.0);  // Default
  EXPECT_DOUBLE_EQ(options->sri_E()[0], 0.0);  // Default

  // Check reaction's falloff_type
  EXPECT_EQ(options->reactions()[0].falloff_type(), "SRI");
}

TEST(FalloffOptionsTest, falloff_sri_5param) {
  // Reaction 9 from sample_reaction_full.yaml (SRI 5-parameter)
  std::string yaml_str = R"(
- equation: H + HO2 (+ M) <=> H2 + O2 (+ M)
  type: falloff
  high-P-rate-constant: {A: 4.0e+15, b: -0.5, Ea_R: 0.0}
  low-P-rate-constant: {A: 7.0e+20, b: -1.0, Ea_R: 0.0}
  efficiencies: {AR: 0.7, H2: 2.0, H2O: 6.0}
  SRI: {A: 1.1, B: 700.0, C: 1234.0, D: 56.0, E: 0.7}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto options = FalloffOptionsImpl::from_yaml(root);

  ASSERT_EQ(options->reactions().size(), 1);

  // Falloff type should be SRI
  EXPECT_EQ(options->falloff_types()[0], static_cast<int>(FalloffType::SRI));

  // Check SRI parameters (5-parameter)
  EXPECT_DOUBLE_EQ(options->sri_A()[0], 1.1);
  EXPECT_DOUBLE_EQ(options->sri_B()[0], 700.0);
  EXPECT_DOUBLE_EQ(options->sri_C()[0], 1234.0);
  EXPECT_DOUBLE_EQ(options->sri_D()[0], 56.0);
  EXPECT_DOUBLE_EQ(options->sri_E()[0], 0.7);

  // Check efficiencies
  const auto& eff = options->efficiencies()[0];
  EXPECT_EQ(eff.size(), 3);

  // Check reaction's falloff_type
  EXPECT_EQ(options->reactions()[0].falloff_type(), "SRI");
}

TEST(FalloffOptionsTest, multiple_reactions) {
  // Parse multiple different reaction types
  std::string yaml_str = R"(
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
  efficiencies: {AR: 0.83}
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
  auto options = FalloffOptionsImpl::from_yaml(root);

  ASSERT_EQ(options->reactions().size(), 4);

  // Check is_three_body flags
  EXPECT_TRUE(options->is_three_body()[0]);   // three-body
  EXPECT_FALSE(options->is_three_body()[1]);  // Lindemann
  EXPECT_FALSE(options->is_three_body()[2]);  // Troe
  EXPECT_FALSE(options->is_three_body()[3]);  // SRI

  // Check falloff types
  EXPECT_EQ(options->falloff_types()[0], static_cast<int>(FalloffType::None));
  EXPECT_EQ(options->falloff_types()[1], static_cast<int>(FalloffType::None));
  EXPECT_EQ(options->falloff_types()[2], static_cast<int>(FalloffType::Troe));
  EXPECT_EQ(options->falloff_types()[3], static_cast<int>(FalloffType::SRI));
}

TEST(FalloffOptionsTest, no_efficiencies) {
  // Reaction without efficiencies should have empty map
  std::string yaml_str = R"(
- equation: 2 OH (+ M) <=> H2O2 (+ M)
  type: falloff
  low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: 0.0}
  high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto options = FalloffOptionsImpl::from_yaml(root);

  ASSERT_EQ(options->reactions().size(), 1);
  ASSERT_EQ(options->efficiencies().size(), 1);
  EXPECT_TRUE(options->efficiencies()[0].empty());
}

TEST(FalloffOptionsTest, skip_other_types) {
  // Parser should skip non-falloff/three-body types
  std::string yaml_str = R"(
- equation: O + H2 <=> H + OH
  type: arrhenius
  rate-constant: {A: 38.7, b: 2.7, Ea_R: 6260.0}
- equation: H2O2 + M <=> O + H2O + M
  type: three-body
  rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
- equation: HO2 <=> OH + O
  type: Chebyshev
  temperature-range: [290.0, 3000.0]
  pressure-range: [0.01, 100.0]
  data: [[8.2883]]
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto options = FalloffOptionsImpl::from_yaml(root);

  // Only the three-body reaction should be parsed
  ASSERT_EQ(options->reactions().size(), 1);
  EXPECT_TRUE(options->is_three_body()[0]);
}

TEST(FalloffOptionsTest, falloff_type_enum) {
  // Test enum conversion utilities
  EXPECT_EQ(falloff_type_to_string(FalloffType::None), "none");
  EXPECT_EQ(falloff_type_to_string(FalloffType::Troe), "Troe");
  EXPECT_EQ(falloff_type_to_string(FalloffType::SRI), "SRI");

  EXPECT_EQ(string_to_falloff_type("none"), FalloffType::None);
  EXPECT_EQ(string_to_falloff_type("Troe"), FalloffType::Troe);
  EXPECT_EQ(string_to_falloff_type("SRI"), FalloffType::SRI);
  EXPECT_EQ(string_to_falloff_type("unknown"), FalloffType::None);
}
