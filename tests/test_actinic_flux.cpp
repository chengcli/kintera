// C/C++
#include <algorithm>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/photolysis/actinic_flux.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;

class ActinicFluxTest : public DeviceTest {};

TEST_P(ActinicFluxTest, InterpolateFlux) {
  auto wavelength =
      torch::tensor({100., 200., 300.}, torch::device(device).dtype(dtype));
  auto flux_vals =
      torch::tensor({1.e14, 2.e14, 1.e14}, torch::device(device).dtype(dtype));

  auto query_wave =
      torch::tensor({150., 250.}, torch::device(device).dtype(dtype));
  auto interp_flux =
      interpolate_actinic_flux(wavelength, flux_vals, query_wave);

  EXPECT_EQ(interp_flux.size(0), 2);
  EXPECT_GT(interp_flux[0].item<double>(), 1.e14);
  EXPECT_LT(interp_flux[0].item<double>(), 2.e14);
}

TEST_P(ActinicFluxTest, CreateUniformFlux) {
  auto wavelength =
      torch::linspace(100., 300., 21, torch::device(device).dtype(dtype));
  auto flux = create_uniform_flux(wavelength, 1.e14);

  EXPECT_EQ(flux.size(0), 21);
  EXPECT_NEAR(flux.mean().item<double>(), 1.e14, 1e-6);
}

TEST_P(ActinicFluxTest, CreateSolarFlux) {
  auto wavelength =
      torch::linspace(100., 800., 71, torch::device(device).dtype(dtype));
  auto flux = create_solar_flux(wavelength, 1.e14);

  EXPECT_EQ(flux.size(0), 71);
  auto max_idx = flux.argmax().item<int>();
  auto peak_wave = wavelength[max_idx].item<double>();
  EXPECT_NEAR(peak_wave, 500., 20.);
}

TEST_P(ActinicFluxTest, FluxOptionsInterpolated) {
  auto opts = ActinicFluxOptionsImpl::create();
  opts->wavelength() = {100., 200., 300.};
  opts->default_flux() = {1.e14, 2.e14, 1.e14};

  auto target = torch::tensor({150., 250.}, torch::device(device).dtype(dtype));
  auto flux = create_actinic_flux(opts, target);

  EXPECT_EQ(flux.size(0), 2);
  EXPECT_GT(flux[0].item<double>(), 1.e14);
  EXPECT_LT(flux[0].item<double>(), 2.e14);
}

TEST_P(ActinicFluxTest, FluxOptionsDefaultOnTargetGrid) {
  auto opts = ActinicFluxOptionsImpl::create();
  opts->default_flux() = {1.e14, 2.e14, 1.e14};

  auto target =
      torch::tensor({100., 200., 300.}, torch::device(device).dtype(dtype));
  auto flux = create_actinic_flux(opts, target);

  EXPECT_EQ(flux.size(0), 3);
  EXPECT_NEAR(flux[1].item<double>(), 2.e14, 1e-6);
}

TEST_P(ActinicFluxTest, FluxOptionsUnitDefault) {
  auto opts = ActinicFluxOptionsImpl::create();
  auto target =
      torch::tensor({100., 200., 300.}, torch::device(device).dtype(dtype));
  auto flux = create_actinic_flux(opts, target);

  EXPECT_EQ(flux.size(0), 3);
  EXPECT_NEAR(flux.sum().item<double>(), 3.0, 1e-6);
}

TEST_P(ActinicFluxTest, MultiDimensionalFluxInterpolation) {
  auto wavelength =
      torch::tensor({100., 200., 300.}, torch::device(device).dtype(dtype));
  auto flux_vals =
      torch::ones({3, 2, 4}, torch::device(device).dtype(dtype)) * 1.e14;

  auto query_wave =
      torch::tensor({150., 250.}, torch::device(device).dtype(dtype));
  auto interp_flux =
      interpolate_actinic_flux(wavelength, flux_vals, query_wave);

  EXPECT_EQ(interp_flux.size(0), 2);
  EXPECT_EQ(interp_flux.size(1), 2);
  EXPECT_EQ(interp_flux.size(2), 4);
}

INSTANTIATE_TEST_SUITE_P(
    DeviceTests, ActinicFluxTest,
    testing::Values(Parameters{torch::kCPU, torch::kFloat64},
                    Parameters{torch::kCUDA, torch::kFloat64}),
    [](const testing::TestParamInfo<ActinicFluxTest::ParamType>& info) {
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
