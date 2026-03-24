// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/constants.h>

#include <kintera/kinetics/kinetics.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>
#include <kintera/kinetics/evolve_implicit.hpp>
#include <kintera/thermo/log_svp.hpp>
#include <kintera/thermo/relative_humidity.hpp>
#include <kintera/thermo/thermo.hpp>
#include <kintera/thermo/thermo_formatter.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;

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

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
