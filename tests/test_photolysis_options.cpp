// C/C++
#include <algorithm>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/kinetics/photolysis.hpp>
#include <kintera/utils/parse_comp_string.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;

class PhotolysisOptionsTest : public testing::Test {
 protected:
  void SetUp() override {
    // Initialize species names for testing
    extern std::vector<std::string> species_names;
    species_names = {"CH4", "CH3", "(1)CH2", "(3)CH2", "CH", "H2", "H", "N2"};
  }
};

TEST_F(PhotolysisOptionsTest, CreateDefaultOptions) {
  auto opts = PhotolysisOptionsImpl::create();
  ASSERT_NE(opts, nullptr);
  EXPECT_EQ(opts->name(), "photolysis");
  EXPECT_EQ(opts->reactions().size(), 0);
  EXPECT_EQ(opts->wavelength().size(), 0);
}

TEST_F(PhotolysisOptionsTest, ParseYAMLInlineData) {
  // Create inline YAML
  std::string yaml_str = R"(
- equation: N2 => N2
  type: photolysis
  cross-section:
    - format: YAML
      temperature-range: [0., 300.]
      data:
        - [20., 1.e-18]
        - [100., 2.e-18]
        - [180., 3.e-18]
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = PhotolysisOptionsImpl::from_yaml(root);

  ASSERT_NE(opts, nullptr);
  EXPECT_EQ(opts->reactions().size(), 1);
  EXPECT_EQ(opts->reactions()[0].equation(), "N2 => N2");
  EXPECT_EQ(opts->wavelength().size(), 3);
  EXPECT_DOUBLE_EQ(opts->wavelength()[0], 20.);
  EXPECT_DOUBLE_EQ(opts->wavelength()[1], 100.);
  EXPECT_DOUBLE_EQ(opts->wavelength()[2], 180.);
}

TEST_F(PhotolysisOptionsTest, ParseBranchReactions) {
  std::string yaml_str = R"(
- equation: CH4 => CH4 + CH3 + (1)CH2 + (3)CH2 + CH + H2 + H
  type: photolysis
  branches:
    - "CH4:1"
    - "CH3:1 H:1"
    - "(1)CH2:1 H2:1"
    - "(3)CH2:1 H:2"
    - "CH:1 H2:1 H:1"
  cross-section:
    - format: YAML
      temperature-range: [0., 300.]
      data:
        - [100., 1.e-18, 0.5e-18, 0.3e-18, 0.15e-18, 0.05e-18]
        - [150., 2.e-18, 1.0e-18, 0.6e-18, 0.30e-18, 0.10e-18]
)";

  YAML::Node root = YAML::Load(yaml_str);
  auto opts = PhotolysisOptionsImpl::from_yaml(root);

  ASSERT_NE(opts, nullptr);
  EXPECT_EQ(opts->reactions().size(), 1);
  // 5 branches from YAML + 1 photoabsorption = 6 total
  EXPECT_GE(opts->branch_names().size(), 1);
  EXPECT_GE(opts->branch_names()[0].size(), 5);
}

// Test module creation
class PhotolysisModuleTest : public DeviceTest {};

TEST_P(PhotolysisModuleTest, CreateModule) {
  auto opts = PhotolysisOptionsImpl::create();
  opts->wavelength() = {100., 150., 200.};
  opts->temperature() = {200., 300.};

  // Add a simple reaction
  opts->reactions().push_back(Reaction("N2 => N2"));
  opts->cross_section() = {1.e-18, 2.e-18, 3.e-18};  // one branch
  opts->branches().push_back({parse_comp_string("N2:1")});

  Photolysis module(opts);
  module->to(device, dtype);

  EXPECT_EQ(module->wavelength.size(0), 3);
}

TEST_P(PhotolysisModuleTest, ForwardWithSimpleFlux) {
  auto opts = PhotolysisOptionsImpl::create();
  opts->wavelength() = {100., 150., 200.};
  opts->temperature() = {200., 300.};

  // Add test reaction with one branch
  opts->reactions().push_back(Reaction("N2 => N2"));
  opts->cross_section() = {1.e-18, 2.e-18, 1.e-18};
  opts->branches().push_back({parse_comp_string("N2:1")});

  Photolysis module(opts);
  module->to(device, dtype);

  // Create test inputs
  auto temp = torch::tensor({250.}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({1.e5}, torch::device(device).dtype(dtype));
  auto conc = torch::zeros({1, 8}, torch::device(device).dtype(dtype));

  auto wave =
      torch::tensor({100., 150., 200.}, torch::device(device).dtype(dtype));
  auto aflux = torch::ones({3}, torch::device(device).dtype(dtype));

  std::map<std::string, torch::Tensor> other = {{"wavelength", wave},
                                                {"actinic_flux", aflux}};

  auto rate = module->forward(temp, pres, conc, other);

  EXPECT_EQ(rate.dim(), 2);
  EXPECT_EQ(rate.size(-1), 1);  // one reaction
  // Rate should be integral of cross-section * flux
  // For constant flux = 1, this is the trapezoid integral of cross-sections
  EXPECT_GT(rate[0][0].item<double>(), 0.);
}

TEST_P(PhotolysisModuleTest, InterpolateCrossSection) {
  auto opts = PhotolysisOptionsImpl::create();
  opts->wavelength() = {100., 200., 300.};
  opts->temperature() = {200., 400.};

  opts->reactions().push_back(Reaction("N2 => N2"));
  opts->cross_section() = {1.e-18, 2.e-18, 1.e-18};
  opts->branches().push_back({parse_comp_string("N2:1")});

  Photolysis module(opts);
  module->to(device, dtype);

  // Query wavelengths between stored values
  auto query_wave =
      torch::tensor({150., 250.}, torch::device(device).dtype(dtype));
  auto query_temp = torch::tensor({300.}, torch::device(device).dtype(dtype));

  auto xs = module->interp_cross_section(0, query_wave, query_temp);

  EXPECT_EQ(xs.size(0), 2);  // 2 query wavelengths
  // Interpolated values should be between neighboring stored values
  EXPECT_GT(xs[0][0].item<double>(), 1.e-18);
  EXPECT_LT(xs[0][0].item<double>(), 2.e-18);
}

INSTANTIATE_TEST_SUITE_P(
    DeviceTests, PhotolysisModuleTest,
    testing::Values(Parameters{torch::kCPU, torch::kFloat64},
                    Parameters{torch::kCUDA, torch::kFloat64}),
    [](const testing::TestParamInfo<PhotolysisModuleTest::ParamType>& info) {
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

