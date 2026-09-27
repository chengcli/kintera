// C/C++
#include <functional>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/constants.h>

#include <kintera/kinetics/evolve_implicit.hpp>
#include <kintera/kinetics/kinetics.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>
#include <kintera/thermo/log_svp.hpp>
#include <kintera/thermo/relative_humidity.hpp>
#include <kintera/thermo/thermo.hpp>
#include <kintera/thermo/thermo_formatter.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;

TEST(EvolveImplicit, BatchedCpuWorkspaceAndSingularFallback) {
  auto options =
      torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  auto rate = torch::ones({128, 2}, options);
  auto stoich = torch::eye(2, options);
  auto jacobian = torch::zeros({128, 2, 2}, options);
  auto expected = rate.clone();

  for (int index : {0, 17, 127}) {
    jacobian.select(0, index).select(0, 1).select(0, 1).fill_(1.);
    expected.select(0, index).select(0, 1).fill_(0.);
  }

  auto delta = evolve_implicit(rate, stoich, jacobian, 1.);
  EXPECT_TRUE(torch::allclose(delta, expected, 1.e-12, 1.e-12));
}

TEST(KineticsFormatter, NoneSubOptionsPrintAsEmpty) {
  auto op = KineticsOptionsImpl::create();
  op->arrhenius(ArrheniusOptions());
  op->coagulation(CoagulationOptions());
  op->evaporation(EvaporationOptions());
  auto s = fmt::format("{}", op);
  EXPECT_NE(s.find("Arrhenius Reactions:\n--\n"), std::string::npos) << s;
  EXPECT_NE(s.find("Coagulation Reactions:\n--\n"), std::string::npos) << s;
  EXPECT_NE(s.find("Evaporation Reactions:\n--\n"), std::string::npos) << s;
}

// KineticsOptionsImpl::clone() must deep-copy every sub-option: the clone
// and the original must not share a sub-option object, and mutating one
// must not change the other.
struct SubOptionAccess {
  std::string name;
  std::function<void const*(KineticsOptions const&)> ptr;
  std::function<double&(KineticsOptions const&)> Tref;
};

template <typename T>
SubOptionAccess sub_option(std::string name, T& (KineticsOptionsImpl::*get)()) {
  return {name,
          [get](KineticsOptions const& op) {
            return static_cast<void const*>(((*op).*get)().get());
          },
          [get](KineticsOptions const& op) -> double& {
            return ((*op).*get)()->Tref();
          }};
}

std::vector<SubOptionAccess> all_sub_options() {
  return {
      sub_option("arrhenius", &KineticsOptionsImpl::arrhenius),
      sub_option("coagulation", &KineticsOptionsImpl::coagulation),
      sub_option("evaporation", &KineticsOptionsImpl::evaporation),
      sub_option("three_body", &KineticsOptionsImpl::three_body),
      sub_option("lindemann_falloff", &KineticsOptionsImpl::lindemann_falloff),
      sub_option("troe_falloff", &KineticsOptionsImpl::troe_falloff),
      sub_option("sri_falloff", &KineticsOptionsImpl::sri_falloff),
      sub_option("kb_falloff", &KineticsOptionsImpl::kb_falloff),
  };
}

class KineticsOptionsClone : public testing::TestWithParam<int> {};

TEST_P(KineticsOptionsClone, DeepCopiesSubOption) {
  auto sub = all_sub_options()[GetParam()];
  auto op = KineticsOptionsImpl::create();
  sub.Tref(op) = 123.;

  auto cl = op->clone();
  ASSERT_NE(sub.ptr(cl), nullptr);
  EXPECT_NE(sub.ptr(cl), sub.ptr(op)) << sub.name << " is shared";
  EXPECT_EQ(sub.Tref(cl), 123.);

  sub.Tref(cl) = 456.;
  EXPECT_EQ(sub.Tref(op), 123.) << "mutating the clone's " << sub.name
                                << " changed the original";
  sub.Tref(op) = 789.;
  EXPECT_EQ(sub.Tref(cl), 456.) << "mutating the original's " << sub.name
                                << " changed the clone";
}

TEST(KineticsOptionsClone, NullSubOptionStaysNull) {
  auto op = KineticsOptionsImpl::create();
  op->arrhenius(ArrheniusOptions());
  op->coagulation(CoagulationOptions());
  op->evaporation(EvaporationOptions());
  op->three_body(ThreeBodyOptions());
  op->lindemann_falloff(LindemannFalloffOptions());
  op->troe_falloff(TroeFalloffOptions());
  op->sri_falloff(SRIFalloffOptions());
  op->kb_falloff(KBFalloffOptions());
  auto cl = op->clone();
  for (auto const& sub : all_sub_options()) {
    EXPECT_EQ(sub.ptr(cl), nullptr) << sub.name;
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllSubOptions, KineticsOptionsClone, testing::Range(0, 8),
    [](testing::TestParamInfo<int> const& info) {
      return all_sub_options()[info.param].name;
    });

TEST_P(DeviceTest, kinetics) {
  auto op_kinet = KineticsOptionsImpl::from_yaml("jupiter.yaml");
  std::cout << fmt::format("{}", op_kinet) << std::endl;

  Kinetics kinet(op_kinet);
  kinet->to(device, dtype);
  std::cout << fmt::format("{}", kinet->options) << std::endl;
}

TEST_P(DeviceTest, merge) {
  auto op_thermo = ThermoOptionsImpl::from_yaml("jupiter.yaml");
  std::cout << fmt::format("{}", op_thermo) << std::endl;

  auto op_kinet = KineticsOptionsImpl::from_yaml("jupiter.yaml");
  std::cout << fmt::format("{}", op_kinet) << std::endl;

  populate_thermo(op_thermo);
  populate_thermo(op_kinet);
  auto op_all = merge_thermo(op_thermo, op_kinet);
  std::cout << fmt::format("{}", op_all) << std::endl;
}

TEST_P(DeviceTest, forward) {
  auto op_kinet = KineticsOptionsImpl::from_yaml("jupiter.yaml");
  Kinetics kinet(op_kinet);
  kinet->to(device, dtype);

  std::cout << fmt::format("{}", kinet->options) << std::endl;
  std::cout << "kinet stoich =\n" << kinet->stoich << std::endl;

  auto op_thermo = ThermoOptionsImpl::from_yaml("jupiter.yaml");
  op_thermo->max_iter(10);
  ThermoX thermo(op_thermo, op_kinet);
  thermo->to(device, dtype);

  std::cout << fmt::format("{}", thermo->options) << std::endl;
  std::cout << "thermo stoich =\n" << thermo->stoich << std::endl;

  auto species = thermo->options->species();
  int ny = species.size() - 1;
  std::cout << "Species = " << species << std::endl;

  auto xfrac =
      torch::zeros({1, 2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  auto temp = 200. * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));

  auto conc = thermo->compute("TPX->V", {temp, pres, xfrac});
  std::cout << "conc = " << conc << std::endl;

  auto conc_kinet = kinet->options->narrow_copy(conc, thermo->options);
  std::cout << "conc_kinet = " << conc_kinet << std::endl;

  auto [rate, rc_ddC, rc_ddT] = kinet->forward(temp, pres, conc_kinet);

  // Species tendencies: du = stoich^T @ rate (stoich is nspecies x nreaction)
  auto du = rate.matmul(kinet->stoich.t());
  std::cout << "rate: " << rate << std::endl;
  std::cout << "du: " << du << std::endl;
}

TEST_P(DeviceTest, evolve_implicit) {
  auto op_kinet = KineticsOptionsImpl::from_yaml("jupiter.yaml");
  Kinetics kinet(op_kinet);
  kinet->to(device, dtype);

  auto op_thermo = ThermoOptionsImpl::from_yaml("jupiter.yaml");
  op_thermo->max_iter(10);
  ThermoX thermo(op_thermo, op_kinet);
  thermo->to(device, dtype);

  auto species = thermo->options->species();
  int ny = species.size() - 1;

  auto xfrac =
      torch::zeros({1, 2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  auto temp = 300. * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));

  auto conc = thermo->compute("TPX->V", {temp, pres, xfrac});
  auto conc_kinet = kinet->options->narrow_copy(conc, thermo->options);

  auto [rate, rc_ddC, rc_ddT] = kinet->forward(temp, pres, conc_kinet);

  // Species tendencies
  auto du = rate.matmul(kinet->stoich.t());
  std::cout << "du: " << du << std::endl;

  // Compute Jacobian
  auto cvol = torch::ones_like(temp);
  auto jac = kinet->jacobian(temp, conc_kinet, cvol, rate, rc_ddC, rc_ddT);
  std::cout << "Jacobian shape: " << jac.sizes() << std::endl;

  // Implicit Euler step
  double dt = 1.e3;
  // evolve_implicit operates on single-point (nspecies,) tensors
  // For batch, take the first element
  auto rate_0 = rate[0][0][0];
  auto jac_0 = jac[0][0][0];
  auto delta = evolve_implicit(rate_0, kinet->stoich, jac_0, dt);
  std::cout << "Implicit Euler delta: " << delta << std::endl;

  std::cout << "Forward + implicit evolve completed successfully\n";
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
