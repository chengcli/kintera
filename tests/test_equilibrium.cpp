#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <future>
#include <kintera/equilibrium/equilibrium.hpp>
#include <kintera/species.hpp>
#include <kintera/utils/molar_mass.hpp>

#define DEVICE_TESTING_SKIP_DEFAULT_INSTANTIATION
#include "device_testing.hpp"

using namespace kintera;

namespace {

EquilibriumOptions make_options() {
  auto op = EquilibriumOptionsImpl::create();
  op->components({"A", "B"})
      .phases({"gas"})
      .phase_ids({0, 0})
      .reactions({"A <=> B"})
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
  if (device.type() == torch::kMPS) GTEST_SKIP();
  auto op = make_options();
  EquilibriumTP eq(op);
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
  EXPECT_TRUE(
      torch::allclose(moles, torch::tensor({{0.8, 0.2}}, tensor_options)));
}

TEST_P(EquilibriumDeviceTest, EvaluatesLogReactionConstant) {
  if (device.type() == torch::kMPS) GTEST_SKIP();
  auto op = make_options();
  op->A({0.1}).B2({0.4}).B1({0.4}).C({0.3}).D1({0.1}).D2({0.05});
  EquilibriumTP eq(op);
  eq->to(device, dtype);

  auto tensor_options = torch::device(device).dtype(dtype);
  auto temp = torch::tensor({2., 4.}, tensor_options);
  auto pres = torch::tensor(1.e5, tensor_options);
  auto moles = torch::tensor({0.8, 0.2}, tensor_options);

  auto [result, gain, diag] = eq->forward(temp, pres, moles);
  auto expected_log_k = 0.1 + 0.4 / temp.square() + 0.4 / temp +
                        0.3 * temp.log() + 0.1 * temp + 0.05 * temp.square();
  auto expected_b = torch::sigmoid(expected_log_k);
  EXPECT_TRUE(torch::allclose(result.select(-1, 1), expected_b, 2.e-5, 1.e-6));
  EXPECT_TRUE(torch::all(diag.select(-1, 0) == 0).item<bool>());

  auto explicit_log_k = torch::zeros({2, 1}, tensor_options);
  auto [overridden, override_gain, override_diag] =
      eq->forward(temp, pres, moles, explicit_log_k);
  EXPECT_TRUE(torch::allclose(overridden.select(-1, 1),
                              torch::full({2}, .5, tensor_options), 1.e-5,
                              1.e-6));
}

TEST(EquilibriumOptions, RequiresCoefficientsWhenLogKIsOmitted) {
  auto op = make_options();
  EquilibriumTP equilibrium(op);
  auto tensor_options = torch::TensorOptions().dtype(torch::kFloat64);
  EXPECT_THROW(equilibrium->forward(torch::tensor(1000., tensor_options),
                                    torch::tensor(1.e5, tensor_options),
                                    torch::tensor({0.8, 0.2}, tensor_options)),
               c10::Error);
}

TEST(EquilibriumOptions, RejectsPartialLogReactionConstants) {
  auto op = EquilibriumOptionsImpl::create();
  op->components({"A", "B", "C"})
      .phases({"gas"})
      .phase_ids({0, 0, 0})
      .reactions({"A <=> B", "B <=> C"})
      .A({1.})
      .B2({0.})
      .B1({0.})
      .C({0.})
      .D1({0.})
      .D2({0.});
  EXPECT_THROW(op->validate(), c10::Error);
}

TEST(MolarMass, StandaloneUtilities) {
  EXPECT_NEAR(atomic_mass("H"), 1.008e-3, 1.e-8);
  EXPECT_NEAR(molar_mass({{"H", 2.}, {"O", 1.}}), 18.015e-3, 1.e-8);
}

TEST(Nasa9, ConcurrentFirstDatabaseAccess) {
  std::vector<std::future<torch::Tensor>> futures;
  for (int i = 0; i < 8; ++i) {
    futures.push_back(std::async(std::launch::async, [] {
      return nasa9_gibbs_rt(torch::tensor(1500., torch::kFloat64),
                            {"H2", "O2", "H2O"});
    }));
  }
  auto expected = futures.front().get();
  EXPECT_EQ(expected.sizes(), torch::IntArrayRef({3}));
  EXPECT_TRUE(torch::isfinite(expected).all().item<bool>());
  for (size_t i = 1; i < futures.size(); ++i) {
    EXPECT_TRUE(torch::allclose(futures[i].get(), expected));
  }
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
         << "  - {name: A, composition: {H: 1}}\n"
         << "  - {name: B, composition: {H: 1}}\n"
         << "reactions:\n"
         << "  - type: equilibrium\n"
         << "    equation: 'A <=> B'\n"
         << "    log-reaction-constant: {A: 1, B1: 2}\n"
         << "  - {type: arrhenius, equation: 'B => A'}\n"
         << "equilibrium: {standard-pressure: 200000, max-iter: 12, "
            "ftol: 1.e-7}\n";
  }

  auto op = EquilibriumOptionsImpl::from_yaml(path.string());
  EXPECT_EQ(op->components(), std::vector<std::string>({"A", "B"}));
  EXPECT_EQ(op->reactions(), std::vector<std::string>({"A <=> B"}));
  EXPECT_EQ(op->phase_ids(), std::vector<int>({0, 0}));
  EXPECT_EQ(op->A(), std::vector<double>({1.}));
  EXPECT_EQ(op->B1(), std::vector<double>({2.}));
  EXPECT_EQ(op->B2(), std::vector<double>({0.}));
  EXPECT_EQ(op->C(), std::vector<double>({0.}));
  EXPECT_EQ(op->D1(), std::vector<double>({0.}));
  EXPECT_EQ(op->D2(), std::vector<double>({0.}));
  EXPECT_DOUBLE_EQ(op->standard_pressure(), 2.e5);
  EXPECT_EQ(op->max_iter(), 12);
  EquilibriumTP equilibrium(op);
  EXPECT_DOUBLE_EQ(equilibrium->stoich[0][0].item<double>(), -1.);
  EXPECT_DOUBLE_EQ(equilibrium->stoich[1][0].item<double>(), 1.);
  auto masses = molar_masses_from_yaml(path.string());
  ASSERT_EQ(masses.size(), 2u);
  EXPECT_NEAR(masses[0], 1.008e-3, 1.e-8);
  EXPECT_NEAR(masses[1], 1.008e-3, 1.e-8);
  std::filesystem::remove(path);
}

TEST(EquilibriumOptions, RejectsUnbalancedYamlReaction) {
  auto path = std::filesystem::temp_directory_path() /
              "kintera_unbalanced_equilibrium_test.yaml";
  {
    std::ofstream yaml(path);
    yaml << "phases:\n"
         << "  - {name: gas, thermo: ideal-gas, species: [A, B]}\n"
         << "species:\n"
         << "  - {name: A, composition: {H: 1}}\n"
         << "  - {name: B, composition: {H: 2}}\n"
         << "reactions:\n"
         << "  - {type: equilibrium, equation: 'A <=> B'}\n";
  }
  EXPECT_THROW(EquilibriumOptionsImpl::from_yaml(path.string()), c10::Error);
  std::filesystem::remove(path);
}

INSTANTIATE_TEST_SUITE_P(
    DeviceAndDtype, EquilibriumDeviceTest,
    testing::Values(Parameters{torch::kCPU, torch::kFloat32},
                    Parameters{torch::kCPU, torch::kFloat64},
                    Parameters{torch::kCUDA, torch::kFloat32},
                    Parameters{torch::kCUDA, torch::kFloat64}),
    [](const testing::TestParamInfo<EquilibriumDeviceTest::ParamType>& info) {
      std::string name = torch::Device(info.param.device_type).str();
      name += "_";
      name += torch::toString(info.param.dtype);
      std::replace(name.begin(), name.end(), '.', '_');
      return name;
    });

}  // namespace
