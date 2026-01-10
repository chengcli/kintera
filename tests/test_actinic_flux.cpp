// C/C++
#include <algorithm>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/kinetics/actinic_flux.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;

class ActinicFluxTest : public DeviceTest {};

TEST_P(ActinicFluxTest, CreateEmptyFlux) {
  ActinicFluxData flux;
  EXPECT_FALSE(flux.is_valid());
  EXPECT_EQ(flux.nwave(), 0);
}

TEST_P(ActinicFluxTest, CreateFluxFromTensors) {
  auto wavelength =
      torch::tensor({100., 200., 300.}, torch::device(device).dtype(dtype));
  auto flux_vals =
      torch::tensor({1.e14, 2.e14, 1.e14}, torch::device(device).dtype(dtype));

  ActinicFluxData flux(wavelength, flux_vals);

  EXPECT_TRUE(flux.is_valid());
  EXPECT_EQ(flux.nwave(), 3);
}

TEST_P(ActinicFluxTest, CreateUniformFlux) {
  auto flux = create_uniform_flux(100., 300., 21, 1.e14, device, dtype);

  EXPECT_TRUE(flux.is_valid());
  EXPECT_EQ(flux.nwave(), 21);
  EXPECT_NEAR(flux.wavelength[0].item<double>(), 100., 1e-6);
  EXPECT_NEAR(flux.wavelength[20].item<double>(), 300., 1e-6);

  // All flux values should be equal
  auto mean_flux = flux.flux.mean().item<double>();
  EXPECT_NEAR(mean_flux, 1.e14, 1e-6);
}

TEST_P(ActinicFluxTest, CreateSolarFlux) {
  auto flux = create_solar_flux(100., 800., 71, 1.e14, device, dtype);

  EXPECT_TRUE(flux.is_valid());
  EXPECT_EQ(flux.nwave(), 71);

  // Peak should be around 500 nm
  auto max_idx = flux.flux.argmax().item<int>();
  auto peak_wave = flux.wavelength[max_idx].item<double>();
  EXPECT_NEAR(peak_wave, 500., 20.);  // Within 20 nm of peak
}

TEST_P(ActinicFluxTest, InterpolateFlux) {
  auto wavelength =
      torch::tensor({100., 200., 300.}, torch::device(device).dtype(dtype));
  auto flux_vals =
      torch::tensor({1.e14, 2.e14, 1.e14}, torch::device(device).dtype(dtype));

  ActinicFluxData flux(wavelength, flux_vals);

  // Interpolate to midpoint
  auto query_wave =
      torch::tensor({150., 250.}, torch::device(device).dtype(dtype));
  auto interp_flux = flux.interpolate_to(query_wave);

  EXPECT_EQ(interp_flux.size(0), 2);
  // At 150 nm, flux should be between 1e14 and 2e14
  EXPECT_GT(interp_flux[0].item<double>(), 1.e14);
  EXPECT_LT(interp_flux[0].item<double>(), 2.e14);
}

TEST_P(ActinicFluxTest, ToMap) {
  auto wavelength =
      torch::tensor({100., 200., 300.}, torch::device(device).dtype(dtype));
  auto flux_vals =
      torch::tensor({1.e14, 2.e14, 1.e14}, torch::device(device).dtype(dtype));

  ActinicFluxData flux(wavelength, flux_vals);
  auto map = flux.to_map();

  EXPECT_EQ(map.count("wavelength"), 1);
  EXPECT_EQ(map.count("actinic_flux"), 1);
  EXPECT_EQ(map.at("wavelength").size(0), 3);
  EXPECT_EQ(map.at("actinic_flux").size(0), 3);
}

TEST_P(ActinicFluxTest, FluxOptions) {
  auto opts = ActinicFluxOptionsImpl::create();
  opts->wavelength() = {100., 200., 300.};
  opts->default_flux() = {1.e14, 2.e14, 1.e14};
  opts->wave_min(50.);
  opts->wave_max(400.);

  auto flux = create_actinic_flux(opts, device, dtype);

  EXPECT_TRUE(flux.is_valid());
  EXPECT_EQ(flux.nwave(), 3);
}

TEST_P(ActinicFluxTest, FluxOptionsEmptyWavelength) {
  auto opts = ActinicFluxOptionsImpl::create();
  // Don't set wavelength

  auto flux = create_actinic_flux(opts, device, dtype);

  EXPECT_FALSE(flux.is_valid());
}

TEST_P(ActinicFluxTest, FluxOptionsDefaultFlux) {
  auto opts = ActinicFluxOptionsImpl::create();
  opts->wavelength() = {100., 200., 300.};
  // Don't set default_flux - should default to ones

  auto flux = create_actinic_flux(opts, device, dtype);

  EXPECT_TRUE(flux.is_valid());
  // Should be all ones
  auto sum = flux.flux.sum().item<double>();
  EXPECT_NEAR(sum, 3.0, 1e-6);
}

TEST_P(ActinicFluxTest, MultiDimensionalFlux) {
  // Test with flux varying across columns/layers
  auto wavelength =
      torch::tensor({100., 200., 300.}, torch::device(device).dtype(dtype));

  // Flux shape (nwave, ncol, nlyr) = (3, 2, 4)
  auto flux_vals =
      torch::ones({3, 2, 4}, torch::device(device).dtype(dtype)) * 1.e14;

  ActinicFluxData flux(wavelength, flux_vals);

  EXPECT_TRUE(flux.is_valid());
  EXPECT_EQ(flux.nwave(), 3);
  EXPECT_EQ(flux.flux.dim(), 3);
  EXPECT_EQ(flux.flux.size(1), 2);
  EXPECT_EQ(flux.flux.size(2), 4);
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

