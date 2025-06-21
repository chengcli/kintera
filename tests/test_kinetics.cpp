// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/constants.h>

#include <kintera/kinetics/kinetic_rate.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>
#include <kintera/thermo/thermo.hpp>

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
  auto op_kinet = KineticRateOptions::from_yaml("jupiter.yaml");
  auto op_all = merge_thermo(op_thermo, op_kinet);
  std::cout << fmt::format("{}", op_all) << std::endl;
}

/*TEST_P(DeviceTest, forward) {
  auto op_kinet = KineticRateOptions::from_yaml("jupiter.yaml");
  KineticRate kinet(op_kinet);
  kinet->to(device, dtype);

  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml").max_iter(10);
  ThermoX thermo(op_thermo);
  thermo->to(device, dtype);

  auto op_all = merge_thermo(op_kinet.options, op_thermo);

  int nx = op_all.s
  auto xfrac =
      torch::zeros({1, 2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  auto temp = 300. * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto conc = thermo->compute("TPX->V", {temp, pres, xfrac});

  auto [rate, rc_ddT] = kinet->forward(temp, pres, conc);
  std::cout << "rate: " << rate << std::endl;

  switch (rc_ddT.has_value()) {
    case true:
      std::cout << "rc_ddT: " << rc_ddT.value() << std::endl;
      break;
    case false:
      std::cout << "rc_ddT: None" << std::endl;
      break;
  }
}*/

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
