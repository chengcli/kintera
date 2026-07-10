#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include <kintera/equilibrium/equilibrium.hpp>

#define DEVICE_TESTING_SKIP_DEFAULT_INSTANTIATION
#include "device_testing.hpp"

using namespace kintera;

namespace {

EquilibriumOptions make_options() {
  auto op = EquilibriumOptionsImpl::create();
  op->components({"A", "B"})
      .phases({"gas"})
      .phase_ids({0, 0})
      .stoich({{-1.}, {1.}})
      .element_matrix({{1., 1.}})
      .gas_phase(0)
      .standard_pressure(1.e5)
      .max_iter(50)
      .ftol(1.e-7)
      .mole_floor(1.e-20);
  return op;
}

class EquilibriumDeviceTest : public DeviceTest {};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DeviceTest);

TEST_P(EquilibriumDeviceTest, IdealGasReaction) {
  if (device.type() == torch::kMPS)
    GTEST_SKIP();
  auto op = make_options();
  Equilibrium eq(op);
  eq->to(device, dtype);

  auto tensor_options = torch::device(device).dtype(dtype);
  auto temp = torch::tensor({1000., 1500.}, tensor_options);
  auto pres = torch::tensor(1.e5, tensor_options);
  auto moles = torch::tensor({{0.8, 0.2}}, tensor_options);
  auto log_k = torch::zeros({2, 1}, tensor_options);

  auto [result, gain, diag] = eq->forward(temp, pres, moles, log_k);
  EXPECT_TRUE(torch::allclose(result.select(-1, 0),
                              torch::full({2}, .5, tensor_options), 1.e-5,
                              1.e-6));
  EXPECT_TRUE(torch::allclose(result.sum(-1), torch::ones({2}, tensor_options),
                              1.e-6, 1.e-6));
  EXPECT_EQ(gain.sizes(), torch::IntArrayRef({2, 1, 1}));
  EXPECT_TRUE(torch::all(diag.select(-1, 0) == 0).item<bool>());
  EXPECT_LT(diag.select(-1, 2).max().item<double>(), 1.e-5);
  EXPECT_LT(diag.select(-1, 3).max().item<double>(), 1.e-6);
  EXPECT_TRUE(
      torch::allclose(moles, torch::tensor({{0.8, 0.2}}, tensor_options)));
}

TEST(EquilibriumOptions, RejectsUnbalancedReaction) {
  auto op = make_options();
  op->element_matrix({{1., 2.}});
  EXPECT_THROW(op->validate(), c10::Error);
}

TEST(EquilibriumOptions, ReadsSpeciesAndReactionsFromYaml) {
  auto path = std::filesystem::temp_directory_path() /
              "kintera_equilibrium_options_test.yaml";
  {
    std::ofstream yaml(path);
    yaml << "phases:\n"
         << "  - name: gas\n"
         << "    thermo: ideal-gas\n"
         << "    species: [A, B]\n"
         << "species:\n"
         << "  - {name: A, composition: {X: 1}}\n"
         << "  - {name: B, composition: {X: 1}}\n"
         << "reactions:\n"
         << "  - {type: equilibrium, equation: 'A <=> B'}\n"
         << "  - {type: arrhenius, equation: 'B => A'}\n"
         << "equilibrium: {standard-pressure: 200000, max-iter: 12, "
            "ftol: 1.e-7}\n";
  }

  auto op = EquilibriumOptionsImpl::from_yaml(path.string());
  EXPECT_EQ(op->components(), std::vector<std::string>({"A", "B"}));
  EXPECT_EQ(op->elements(), std::vector<std::string>({"X"}));
  EXPECT_EQ(op->reactions(), std::vector<std::string>({"A <=> B"}));
  EXPECT_EQ(op->phase_ids(), std::vector<int>({0, 0}));
  EXPECT_DOUBLE_EQ(op->stoich()[0][0], -1.);
  EXPECT_DOUBLE_EQ(op->stoich()[1][0], 1.);
  EXPECT_DOUBLE_EQ(op->standard_pressure(), 2.e5);
  EXPECT_EQ(op->max_iter(), 12);
  std::filesystem::remove(path);
}

INSTANTIATE_TEST_SUITE_P(
    DeviceAndDtype, EquilibriumDeviceTest,
    testing::Values(Parameters{torch::kCPU, torch::kFloat32},
                    Parameters{torch::kCPU, torch::kFloat64},
                    Parameters{torch::kCUDA, torch::kFloat32},
                    Parameters{torch::kCUDA, torch::kFloat64}),
    [](const testing::TestParamInfo<EquilibriumDeviceTest::ParamType> &info) {
      std::string name = torch::Device(info.param.device_type).str();
      name += "_";
      name += torch::toString(info.param.dtype);
      std::replace(name.begin(), name.end(), '.', '_');
      return name;
    });

} // namespace
