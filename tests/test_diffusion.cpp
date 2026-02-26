//! @file test_diffusion.cpp
//! @brief Tests for the eddy diffusion (Kzz) transport module

#include <cmath>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <kintera/diffusion/diffusion.hpp>
#include "device_testing.hpp"

using namespace kintera;

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DeviceTest);

class DiffusionTest : public DeviceTest {
 protected:
  int nz = 20;
  int nspecies = 2;
  double Kzz_val = 1.e5;   // cm^2/s
  double dz_val = 1.e5;    // cm (1 km)
  double n_bg = 1.e12;     // background number density

  torch::Tensor make_uniform_grid() {
    return torch::full({nz - 1}, dz_val, torch::device(device).dtype(dtype));
  }

  torch::Tensor make_uniform_Kzz() {
    return torch::full({nz - 1}, Kzz_val, torch::device(device).dtype(dtype));
  }

  torch::Tensor make_gaussian_profile() {
    auto y = torch::zeros({nz, nspecies}, torch::device(device).dtype(dtype));
    double z_mid = nz / 2.0;
    double sigma = nz / 6.0;
    for (int j = 0; j < nz; j++) {
      double gauss = 0.01 * n_bg * std::exp(-std::pow(j - z_mid, 2) / (2 * sigma * sigma));
      y[j][0] = gauss;
      y[j][1] = n_bg - gauss;
    }
    return y;
  }
};

TEST_P(DiffusionTest, TendencyShape) {
  auto dzi = make_uniform_grid();
  auto Kzz = make_uniform_Kzz();
  auto y = make_gaussian_profile();

  auto tend = diffusion_tendency(y, Kzz, dzi);

  EXPECT_EQ(tend.size(0), nz);
  EXPECT_EQ(tend.size(1), nspecies);
}

TEST_P(DiffusionTest, CoefficientsShape) {
  auto dzi = make_uniform_grid();
  auto Kzz = make_uniform_Kzz();
  auto y = make_gaussian_profile();

  auto [A, B, C] = diffusion_coefficients(y, Kzz, dzi);

  EXPECT_EQ(A.size(0), nz);
  EXPECT_EQ(B.size(0), nz);
  EXPECT_EQ(C.size(0), nz);

  // Zero-flux BC: C[0] = 0, B[nz-1] = 0
  EXPECT_NEAR(C[0].item<double>(), 0.0, 1e-30);
  EXPECT_NEAR(B[nz - 1].item<double>(), 0.0, 1e-30);
}

TEST_P(DiffusionTest, MassConservation) {
  auto dzi = make_uniform_grid();
  auto Kzz = make_uniform_Kzz();
  auto y = make_gaussian_profile();

  auto tend = diffusion_tendency(y, Kzz, dzi);

  // Total of each species across all levels should be conserved
  // Sum of tendencies * dz should be zero
  // With uniform grid: sum(tend, dim=0) * dz should be ~0
  auto total_tend = tend.sum(0);

  for (int s = 0; s < nspecies; s++) {
    double net = total_tend[s].item<double>();
    double scale = y.select(1, s).abs().sum().item<double>() / nz;
    double rel = std::abs(net) / (scale + 1e-30);
    EXPECT_LT(rel, 1e-10)
        << "Species " << s << " net tendency = " << net << " (not conserved)";
  }
}

TEST_P(DiffusionTest, UniformProfileNoFlux) {
  auto dzi = make_uniform_grid();
  auto Kzz = make_uniform_Kzz();

  // Uniform mixing ratio -> zero diffusion tendency
  auto y = torch::full({nz, nspecies}, n_bg / nspecies,
                        torch::device(device).dtype(dtype));

  auto tend = diffusion_tendency(y, Kzz, dzi);

  double max_tend = tend.abs().max().item<double>();
  EXPECT_LT(max_tend, 1e-10 * n_bg)
      << "Uniform profile should have zero diffusion tendency";
}

TEST_P(DiffusionTest, GaussianSmooths) {
  auto dzi = make_uniform_grid();
  auto Kzz = make_uniform_Kzz();
  auto y = make_gaussian_profile();

  double initial_peak = y.select(1, 0).max().item<double>();

  // Explicit Euler time stepping
  double dt = 0.1 * dz_val * dz_val / Kzz_val;  // CFL-like condition
  int nsteps = 100;

  for (int step = 0; step < nsteps; step++) {
    auto tend = diffusion_tendency(y, Kzz, dzi);
    y = (y + tend * dt).clamp_min(0.0);
  }

  double final_peak = y.select(1, 0).max().item<double>();

  // Peak should decrease (spreading out)
  EXPECT_LT(final_peak, initial_peak * 0.95)
      << "Gaussian should smooth out over time";

  // Total mass should be approximately conserved
  double initial_total = make_gaussian_profile().select(1, 0).sum().item<double>();
  double final_total = y.select(1, 0).sum().item<double>();
  double conservation_err = std::abs(final_total - initial_total) / initial_total;
  EXPECT_LT(conservation_err, 0.01)
      << "Mass conservation error: " << conservation_err * 100 << "%";

  std::cout << "Gaussian smoothing: peak " << initial_peak << " -> " << final_peak
            << " (ratio " << final_peak / initial_peak << ")\n";
  std::cout << "Mass conservation error: " << conservation_err * 100 << "%\n";
}

TEST_P(DiffusionTest, TendencySignCorrect) {
  auto dzi = make_uniform_grid();
  auto Kzz = make_uniform_Kzz();
  auto y = make_gaussian_profile();

  auto tend = diffusion_tendency(y, Kzz, dzi);

  // At the peak of the Gaussian (center), tendency should be negative
  // (diffusion moves material away from the peak)
  int mid = nz / 2;
  double tend_at_peak = tend[mid][0].item<double>();
  EXPECT_LT(tend_at_peak, 0.0)
      << "Diffusion should remove material from the peak";

  // At the edges (far from peak), tendency should be positive
  // (material diffuses outward)
  double tend_at_edge = tend[1][0].item<double>();
  EXPECT_GT(tend_at_edge, 0.0)
      << "Diffusion should add material at the edges";
}

INSTANTIATE_TEST_SUITE_P(
    DeviceTests, DiffusionTest,
    testing::Values(Parameters{torch::kCPU, torch::kFloat64},
                    Parameters{torch::kMPS, torch::kFloat32},
                    Parameters{torch::kCUDA, torch::kFloat64}),
    [](const testing::TestParamInfo<DiffusionTest::ParamType>& info) {
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
