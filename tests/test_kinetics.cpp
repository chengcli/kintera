// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/constants.h>

#include <kintera/kinetics/kinetic_rate.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>
#include <kintera/thermo/thermo.hpp>
#include <kintera/thermo/thermo_formatter.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;

TEST_P(DeviceTest, kinetic_rate) {
  auto op_kinet = KineticRateOptions::from_yaml("jupiter.yaml");
  KineticRate kinet(op_kinet);
  kinet->to(device, dtype);
  std::cout << fmt::format("{}", kinet->options) << std::endl;
}

TEST_P(DeviceTest, merge) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");
  std::cout << fmt::format("{}", op_thermo) << std::endl;

  auto op_kinet = KineticRateOptions::from_yaml("jupiter.yaml");
  std::cout << fmt::format("{}", op_kinet) << std::endl;

  populate_thermo(op_thermo);
  populate_thermo(op_kinet);
  auto op_all = merge_thermo(op_thermo, op_kinet);
  std::cout << fmt::format("{}", op_all) << std::endl;
}

TEST_P(DeviceTest, forward) {
  auto op_kinet = KineticRateOptions::from_yaml("jupiter.yaml");
  KineticRate kinet(op_kinet);
  kinet->to(device, dtype);

  std::cout << fmt::format("{}", kinet->options) << std::endl;
  std::cout << "kinet stoich =\n" << kinet->stoich << std::endl;

  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml").max_iter(10);
  ThermoX thermo(op_thermo, op_kinet);
  thermo->to(device, dtype);

  std::cout << fmt::format("{}", thermo->options) << std::endl;
  std::cout << "thermo stoich =\n" << thermo->stoich << std::endl;

  auto species = thermo->options.species();
  int ny = species.size() - 1;  // exclude the reference species
  std::cout << "Species = " << species << std::endl;

  auto xfrac =
      torch::zeros({1, 2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  auto temp = 300. * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));

  std::cout << "xfrac = " << xfrac << std::endl;

  auto conc = thermo->compute("TPX->V", {temp, pres, xfrac});
  std::cout << "conc = " << conc << std::endl;

  auto conc_kinet = kinet->options.narrow_copy(conc, thermo->options);
  std::cout << "conc_kinet = " << conc_kinet << std::endl;

  kinet->options.accumulate(conc, conc_kinet, thermo->options);
  std::cout << "conc2 = " << conc << std::endl;

  auto [rate, rc_ddT] = kinet->forward(temp, pres, conc_kinet);
  std::cout << "rate: " << rate << std::endl;

  switch (rc_ddT.has_value()) {
    case true:
      std::cout << "rc_ddT: " << rc_ddT.value() << std::endl;
      break;
    case false:
      std::cout << "rc_ddT: None" << std::endl;
      break;
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
