// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/constants.h>

#include <kintera/kinetics/kinetic_rate.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;

TEST_P(DeviceTest, kinetic_rate) {
  auto op_kinet = KineticRateOptions::from_yaml("jupiter.yaml");

  KineticRate kinet(op_kinet);
  kinet->to(device, dtype);

  std::cout << fmt::format("{}", kinet->options) << std::endl;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
