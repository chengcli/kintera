//! @file test_rt.cpp
//! @brief Tests for the Beer-Lambert RT module

#include <cmath>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <kintera/rt/rt.hpp>
#include "device_testing.hpp"

using namespace kintera;

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DeviceTest);

class RTTest : public DeviceTest {
 protected:
  int nz = 20;
  int nwave = 10;
  double dz_val = 1.e5;  // 1 km in cm
};

TEST_P(RTTest, OpticalDepthShape) {
  auto y = torch::ones({nz, 2}, torch::device(device).dtype(dtype));
  auto cross = torch::ones({1, nwave}, torch::device(device).dtype(dtype)) * 1e-18;
  auto dz = torch::full({nz}, dz_val, torch::device(device).dtype(dtype));

  auto tau = compute_optical_depth(y, cross, dz, {0});

  EXPECT_EQ(tau.size(0), nz + 1);
  EXPECT_EQ(tau.size(1), nwave);
  EXPECT_NEAR(tau[0][0].item<double>(), 0.0, 1e-30);
}

TEST_P(RTTest, OpticalDepthMonotonic) {
  auto y = torch::ones({nz, 2}, torch::device(device).dtype(dtype)) * 1e12;
  auto cross = torch::ones({1, nwave}, torch::device(device).dtype(dtype)) * 1e-18;
  auto dz = torch::full({nz}, dz_val, torch::device(device).dtype(dtype));

  auto tau = compute_optical_depth(y, cross, dz, {0});

  for (int k = 0; k < nz; k++) {
    EXPECT_GT(tau[k + 1][0].item<double>(), tau[k][0].item<double>())
        << "Optical depth must increase from TOA to surface";
  }
}

TEST_P(RTTest, ActinicFluxDecreases) {
  auto y = torch::ones({nz, 2}, torch::device(device).dtype(dtype)) * 1e12;
  auto cross = torch::ones({1, nwave}, torch::device(device).dtype(dtype)) * 1e-17;
  auto stellar = torch::ones({nwave}, torch::device(device).dtype(dtype)) * 1e13;
  auto dz = torch::full({nz}, dz_val, torch::device(device).dtype(dtype));
  double cos_zen = 1.0;

  auto aflux = compute_actinic_flux(y, cross, stellar, dz, cos_zen, {0});

  EXPECT_EQ(aflux.size(0), nz);
  EXPECT_EQ(aflux.size(1), nwave);

  // Top layer (nz-1) should have highest flux, bottom (0) lowest
  double flux_top = aflux[nz - 1][0].item<double>();
  double flux_bot = aflux[0][0].item<double>();
  EXPECT_GT(flux_top, flux_bot) << "Flux should decrease from top to bottom";

  std::cout << "Actinic flux: top=" << flux_top << " bot=" << flux_bot
            << " ratio=" << flux_top / flux_bot << "\n";
}

TEST_P(RTTest, TransparentAtmosphere) {
  // Zero cross-section => no attenuation => flux = stellar everywhere
  auto y = torch::ones({nz, 2}, torch::device(device).dtype(dtype)) * 1e12;
  auto cross = torch::zeros({1, nwave}, torch::device(device).dtype(dtype));
  auto stellar = torch::ones({nwave}, torch::device(device).dtype(dtype)) * 1e13;
  auto dz = torch::full({nz}, dz_val, torch::device(device).dtype(dtype));

  auto aflux = compute_actinic_flux(y, cross, stellar, dz, 1.0, {0});

  for (int j = 0; j < nz; j++) {
    EXPECT_NEAR(aflux[j][0].item<double>(), 1e13, 1e13 * 1e-10)
        << "Transparent atmosphere should have uniform flux";
  }
}

TEST_P(RTTest, BeerLambertAnalytical) {
  // Uniform atmosphere: tau = n * sigma * z
  // F(z) = F0 * exp(-n * sigma * z / cos_zen)
  double n_density = 1e12;  // cm^-3
  double sigma = 1e-17;     // cm^2
  double F0 = 1e13;
  double cos_zen = 0.5;

  auto y = torch::ones({nz, 1}, torch::device(device).dtype(dtype)) * n_density;
  auto cross = torch::ones({1, 1}, torch::device(device).dtype(dtype)) * sigma;
  auto stellar = torch::ones({1}, torch::device(device).dtype(dtype)) * F0;
  auto dz = torch::full({nz}, dz_val, torch::device(device).dtype(dtype));

  auto aflux = compute_actinic_flux(y, cross, stellar, dz, cos_zen, {0});

  // Check layers analytically using interface-averaged flux
  // Layer j (from bottom): top interface at level (nz-1-j), bottom at level (nz-j)
  // tau at level k (from TOA) = k * n * sigma * dz
  double dtau = n_density * sigma * dz_val;
  for (int j = 0; j < nz; j++) {
    int k_top = nz - 1 - j;
    int k_bot = nz - j;
    double F_top = F0 * std::exp(-k_top * dtau / cos_zen);
    double F_bot = F0 * std::exp(-k_bot * dtau / cos_zen);
    double F_expected = (F_top + F_bot) / 2.0;
    double F_computed = aflux[j][0].item<double>();

    double rel_err = std::abs(F_computed / F_expected - 1.0);
    EXPECT_LT(rel_err, 1e-10)
        << "Layer " << j << ": expected=" << F_expected
        << " computed=" << F_computed << " err=" << rel_err;
  }

  std::cout << "Beer-Lambert analytical test passed\n";
}

TEST_P(RTTest, ZenithAngleEffect) {
  auto y = torch::ones({nz, 1}, torch::device(device).dtype(dtype)) * 1e12;
  auto cross = torch::ones({1, 1}, torch::device(device).dtype(dtype)) * 1e-17;
  auto stellar = torch::ones({1}, torch::device(device).dtype(dtype)) * 1e13;
  auto dz = torch::full({nz}, dz_val, torch::device(device).dtype(dtype));

  auto aflux_overhead = compute_actinic_flux(y, cross, stellar, dz, 1.0, {0});
  auto aflux_slanted = compute_actinic_flux(y, cross, stellar, dz, 0.5, {0});

  // Slanted path (cos_zen=0.5) should give less flux at bottom
  double bot_overhead = aflux_overhead[0][0].item<double>();
  double bot_slanted = aflux_slanted[0][0].item<double>();
  EXPECT_LT(bot_slanted, bot_overhead)
      << "Slanted path should have more attenuation";

  std::cout << "Zenith angle: overhead bot=" << bot_overhead
            << " slanted bot=" << bot_slanted << "\n";
}

INSTANTIATE_TEST_SUITE_P(
    DeviceTests, RTTest,
    testing::Values(Parameters{torch::kCPU, torch::kFloat64},
                    Parameters{torch::kMPS, torch::kFloat32},
                    Parameters{torch::kCUDA, torch::kFloat64}),
    [](const testing::TestParamInfo<RTTest::ParamType>& info) {
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
