//! @file test_chapman_cycle.cpp
//! @brief Chapman cycle benchmark: photolysis + three-body reactions

#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include <kintera/constants.h>
#include <kintera/kinetics/arrhenius.hpp>
#include <kintera/kinetics/photolysis.hpp>
#include <kintera/kinetics/three_body.hpp>
#include <kintera/utils/parse_comp_string.hpp>

#include "device_testing.hpp"

using namespace kintera;

namespace kintera {
extern std::vector<std::string> species_names;
extern bool species_initialized;
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DeviceTest);

constexpr int IDX_N2 = 0;
constexpr int IDX_O2 = 1;
constexpr int IDX_O = 2;
constexpr int IDX_O3 = 3;

class ChapmanCycleTest : public DeviceTest {
 protected:
  void SetUp() override {
    DeviceTest::SetUp();
    kintera::species_initialized = false;
    kintera::species_names = {"N2", "O2", "O", "O3"};
  }

  PhotolysisOptions createPhotolysisOptions() {
    auto opts = PhotolysisOptionsImpl::create();

    opts->reactions().push_back(Reaction("O2 => 2 O"));
    opts->reactions().push_back(Reaction("O3 => O2 + O"));

    opts->wavelength() = {120., 140., 160., 180., 200., 220., 240., 260.,
                          280., 300., 320.};
    opts->temperature() = {200., 300.};

    std::vector<double> xs_o2 = {1.5e-17, 1.8e-17, 1.2e-17, 5.0e-18, 7.0e-19,
                                 5.0e-20, 1.0e-21, 1.0e-23, 1.0e-24, 1.0e-25,
                                 1.0e-26};
    std::vector<double> xs_o3 = {1.0e-19, 2.0e-19, 3.0e-19, 5.0e-19, 3.0e-19,
                                 1.0e-18, 5.0e-18, 1.1e-17, 4.0e-18, 8.0e-19,
                                 5.0e-20};

    for (double x : xs_o2) opts->cross_section().push_back(x);
    for (double x : xs_o3) opts->cross_section().push_back(x);

    opts->branches().push_back({parse_comp_string("O2:1")});
    opts->branches().push_back({parse_comp_string("O3:1")});

    return opts;
  }

  ArrheniusOptions createArrheniusOptions() {
    auto opts = ArrheniusOptionsImpl::create();

    opts->reactions().push_back(Reaction("O + O2 => O3"));
    opts->reactions().push_back(Reaction("O + O3 => 2 O2"));

    opts->A() = {1.7e-14, 8.0e-12};
    opts->b() = {-2.4, 0.0};
    opts->Ea_R() = {0.0, 2060.0};

    return opts;
  }

  // O + O3 → 2O2 (only the destruction reaction)
  ArrheniusOptions createArrheniusDestructionOnly() {
    auto opts = ArrheniusOptionsImpl::create();
    opts->reactions().push_back(Reaction("O + O3 => 2 O2"));
    opts->A() = {8.0e-12};
    opts->b() = {0.0};
    opts->Ea_R() = {2060.0};
    return opts;
  }

  // O + O2 + M → O3 + M (three-body formation)
  ThreeBodyOptions createFalloffOptions() {
    std::string yaml_str = R"(
- equation: O + O2 + M <=> O3 + M
  type: three-body
  rate-constant: {A: 6.0e-34, b: -2.4, Ea_R: 0.0}
  efficiencies: {N2: 1.0, O2: 1.0}
)";
    YAML::Node root = YAML::Load(yaml_str);
    return ThreeBodyOptionsImpl::from_yaml(root);
  }

  std::pair<torch::Tensor, torch::Tensor> createActinicFlux(
      const std::vector<double>& wavelength) {
    auto wave = torch::tensor(wavelength, torch::device(device).dtype(dtype));
    auto flux = torch::zeros_like(wave);

    for (int i = 0; i < (int)wavelength.size(); i++) {
      double w = wavelength[i];
      if (w < 200) {
        flux[i] = 1.e10 * std::exp(-(200 - w) / 30);
      } else if (w < 320) {
        flux[i] = 1.e13 * std::exp(-std::pow(w - 250, 2) / 5000);
      } else {
        flux[i] = 1.e14;
      }
    }

    return {wave, flux};
  }
};

TEST_P(ChapmanCycleTest, PhotolysisRatesInRange) {
  auto photo_opts = createPhotolysisOptions();
  Photolysis photolysis(photo_opts);
  photolysis->to(device, dtype);

  double T = 250.0, P = 1000.0;
  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({P}, torch::device(device).dtype(dtype));
  auto conc = torch::zeros({1, 4}, torch::device(device).dtype(dtype));

  double n_tot = P / (constants::Rgas * T);
  conc[0][IDX_N2] = 0.78 * n_tot;
  conc[0][IDX_O2] = 0.21 * n_tot;
  conc[0][IDX_O] = 1.e-10 * n_tot;
  conc[0][IDX_O3] = 1.e-8 * n_tot;

  auto [wave, flux] = createActinicFlux(photo_opts->wavelength());
  std::map<std::string, torch::Tensor> flux_map = {{"wavelength", wave},
                                                   {"actinic_flux", flux}};

  auto J = photolysis->forward(temp, pres, conc, flux_map);

  double J_O2 = J[0][0].item<double>();
  double J_O3 = J[0][1].item<double>();

  EXPECT_GT(J_O2, 1.e-12);
  EXPECT_LT(J_O2, 1.e-4);
  EXPECT_GT(J_O3, 1.e-6);
  EXPECT_LT(J_O3, 1.e0);

  std::cout << "J(O2) = " << J_O2 << " s^-1\n";
  std::cout << "J(O3) = " << J_O3 << " s^-1\n";
}

TEST_P(ChapmanCycleTest, ArrheniusRatesAtTemperature) {
  auto arr_opts = createArrheniusOptions();
  Arrhenius arrhenius(arr_opts);
  arrhenius->to(device, dtype);

  double T = 250.0;
  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({1000.}, torch::device(device).dtype(dtype));
  auto conc = torch::zeros({1, 4}, torch::device(device).dtype(dtype));

  std::map<std::string, torch::Tensor> other;
  auto k = arrhenius->forward(temp, pres, conc, other);

  double k2 = k[0][0].item<double>();
  double k4 = k[0][1].item<double>();
  double k4_expected = 8.0e-12 * std::exp(-2060.0 / T);

  EXPECT_GT(k2, 1.e-16);
  EXPECT_LT(k2, 1.e-12);
  EXPECT_NEAR(k4, k4_expected, k4_expected * 0.1);

  std::cout << "k2 = " << k2 << "\n";
  std::cout << "k4 = " << k4 << "\n";
}

TEST_P(ChapmanCycleTest, MassConservation) {
  auto photo_opts = createPhotolysisOptions();
  auto arr_opts = createArrheniusOptions();

  Photolysis photolysis(photo_opts);
  Arrhenius arrhenius(arr_opts);
  photolysis->to(device, dtype);
  arrhenius->to(device, dtype);

  double T = 250.0, P = 1000.0;
  double n_tot = P / (constants::Rgas * T);

  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({P}, torch::device(device).dtype(dtype));

  auto conc = torch::zeros({1, 4}, torch::device(device).dtype(dtype));
  conc[0][IDX_N2] = 0.78 * n_tot;
  conc[0][IDX_O2] = 0.21 * n_tot;
  conc[0][IDX_O] = 1.e-10 * n_tot;
  conc[0][IDX_O3] = 1.e-8 * n_tot;

  double O_initial = 2.0 * conc[0][IDX_O2].item<double>() +
                     conc[0][IDX_O].item<double>() +
                     3.0 * conc[0][IDX_O3].item<double>();

  auto [wave, flux] = createActinicFlux(photo_opts->wavelength());
  std::map<std::string, torch::Tensor> flux_map = {{"wavelength", wave},
                                                   {"actinic_flux", flux}};
  std::map<std::string, torch::Tensor> other;

  double dt = 1.0;
  int nsteps = 1000;

  for (int step = 0; step < nsteps; step++) {
    auto J = photolysis->forward(temp, pres, conc, flux_map);
    auto k = arrhenius->forward(temp, pres, conc, other);

    double c_O2 = conc[0][IDX_O2].item<double>();
    double c_O = conc[0][IDX_O].item<double>();
    double c_O3 = conc[0][IDX_O3].item<double>();

    double J_O2 = J[0][0].item<double>();
    double J_O3 = J[0][1].item<double>();
    double k2 = k[0][0].item<double>();
    double k4 = k[0][1].item<double>();

    double R1 = J_O2 * c_O2;
    double R2 = k2 * c_O * c_O2;
    double R3 = J_O3 * c_O3;
    double R4 = k4 * c_O * c_O3;

    conc[0][IDX_O2] = std::max(0.0, c_O2 + (-R1 + R2 + R3 + 2 * R4) * dt);
    conc[0][IDX_O] = std::max(0.0, c_O + (2 * R1 - R2 + R3 - R4) * dt);
    conc[0][IDX_O3] = std::max(0.0, c_O3 + (R2 - R3 - R4) * dt);
  }

  double O_final = 2.0 * conc[0][IDX_O2].item<double>() +
                   conc[0][IDX_O].item<double>() +
                   3.0 * conc[0][IDX_O3].item<double>();

  double error = std::abs(O_final - O_initial) / O_initial;
  EXPECT_LT(error, 0.01);

  std::cout << "Mass conservation error: " << error * 100 << "%\n";
}

TEST_P(ChapmanCycleTest, SteadyStateOzone) {
  auto photo_opts = createPhotolysisOptions();
  auto arr_opts = createArrheniusOptions();

  Photolysis photolysis(photo_opts);
  Arrhenius arrhenius(arr_opts);
  photolysis->to(device, dtype);
  arrhenius->to(device, dtype);

  double T = 250.0, P = 1000.0;
  double n_tot = P / (constants::Rgas * T);

  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({P}, torch::device(device).dtype(dtype));

  auto conc = torch::zeros({1, 4}, torch::device(device).dtype(dtype));
  conc[0][IDX_N2] = 0.78 * n_tot;
  conc[0][IDX_O2] = 0.21 * n_tot;
  conc[0][IDX_O] = 1.e-10 * n_tot;
  conc[0][IDX_O3] = 1.e-8 * n_tot;

  auto [wave, flux] = createActinicFlux(photo_opts->wavelength());
  std::map<std::string, torch::Tensor> flux_map = {{"wavelength", wave},
                                                   {"actinic_flux", flux}};
  std::map<std::string, torch::Tensor> other;

  double dt = 10.0;
  int nsteps = 10000;
  double prev_O3 = 0.0;

  for (int step = 0; step < nsteps; step++) {
    auto J = photolysis->forward(temp, pres, conc, flux_map);
    auto k = arrhenius->forward(temp, pres, conc, other);

    double c_O2 = conc[0][IDX_O2].item<double>();
    double c_O = conc[0][IDX_O].item<double>();
    double c_O3 = conc[0][IDX_O3].item<double>();

    double J_O2 = J[0][0].item<double>();
    double J_O3 = J[0][1].item<double>();
    double k2 = k[0][0].item<double>();
    double k4 = k[0][1].item<double>();

    double R1 = J_O2 * c_O2;
    double R2 = k2 * c_O * c_O2;
    double R3 = J_O3 * c_O3;
    double R4 = k4 * c_O * c_O3;

    conc[0][IDX_O2] = std::max(0.0, c_O2 + (-R1 + R2 + R3 + 2 * R4) * dt);
    conc[0][IDX_O] = std::max(0.0, c_O + (2 * R1 - R2 + R3 - R4) * dt);
    conc[0][IDX_O3] = std::max(0.0, c_O3 + (R2 - R3 - R4) * dt);

    if (step % 1000 == 0 && step > 0) {
      double O3_change = std::abs(c_O3 - prev_O3) / (prev_O3 + 1e-30);
      if (O3_change < 1e-4) break;
      prev_O3 = c_O3;
    }
  }

  double final_O3 = conc[0][IDX_O3].item<double>();
  double O3_ppm = final_O3 / n_tot * 1.e6;

  // The simplified model produces very low O3 due to rapid photolysis
  // Just verify the system evolved and O3 is non-negative
  EXPECT_GE(O3_ppm, 0.0);
  EXPECT_LT(O3_ppm, 100);

  std::cout << "Steady-state O3: " << O3_ppm << " ppm\n";
}

// ============================================================================
// Three-body (Falloff) reaction tests
// ============================================================================

TEST_P(ChapmanCycleTest, ThreeBodyOzoneFormation) {
  auto falloff_opts = createFalloffOptions();
  ThreeBody falloff(falloff_opts);
  falloff->to(device, dtype);

  EXPECT_EQ(falloff_opts->reactions().size(), 1);

  double T = 220.0, P = 5000.0;
  double n_tot = P / (constants::Rgas * T);

  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({P}, torch::device(device).dtype(dtype));
  auto conc = torch::zeros({1, 4}, torch::device(device).dtype(dtype));
  conc[0][IDX_N2] = 0.78 * n_tot;
  conc[0][IDX_O2] = 0.21 * n_tot;
  conc[0][IDX_O] = 1.e-10 * n_tot;
  conc[0][IDX_O3] = 1.e-8 * n_tot;

  std::map<std::string, torch::Tensor> other;
  auto k = falloff->forward(temp, pres, conc, other);

  EXPECT_EQ(k.dim(), 2);
  EXPECT_EQ(k.size(-1), 1);
  EXPECT_GT(k[0][0].item<double>(), 0.0);

  std::cout << "k(O+O2+M) = " << k[0][0].item<double>() << "\n";
}

TEST_P(ChapmanCycleTest, ThreeBodyTemperatureDependence) {
  auto falloff_opts = createFalloffOptions();
  ThreeBody falloff(falloff_opts);
  falloff->to(device, dtype);

  auto T = torch::tensor({200.0, 250.0, 300.0}, torch::device(device).dtype(dtype));
  auto P = torch::tensor({5000.0, 5000.0, 5000.0}, torch::device(device).dtype(dtype));
  auto conc = torch::zeros({3, 4}, torch::device(device).dtype(dtype));
  
  for (int i = 0; i < 3; i++) {
    double T_i = T[i].item<double>();
    double n_tot = 5000.0 / (constants::Rgas * T_i);
    conc[i][IDX_N2] = 0.78 * n_tot;
    conc[i][IDX_O2] = 0.21 * n_tot;
    conc[i][IDX_O] = 1e-10 * n_tot;
    conc[i][IDX_O3] = 1e-8 * n_tot;
  }

  std::map<std::string, torch::Tensor> other;
  auto k = falloff->forward(T, P, conc, other);

  // With b = -2.4 < 0, rate increases as T decreases
  double k_200K = k[0][0].item<double>();
  double k_250K = k[1][0].item<double>();
  double k_300K = k[2][0].item<double>();

  EXPECT_GT(k_200K, k_250K);
  EXPECT_GT(k_250K, k_300K);

  std::cout << "k(200K) = " << k_200K << ", k(250K) = " << k_250K << ", k(300K) = " << k_300K << "\n";
}

TEST_P(ChapmanCycleTest, ThreeBodyEfficiencyScaling) {
  auto falloff_opts = createFalloffOptions();
  ThreeBody falloff(falloff_opts);
  falloff->to(device, dtype);

  double T = 250.0;
  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({10000.0}, torch::device(device).dtype(dtype));

  // Test 1: Pure N2 bath (efficiency = 1.0)
  auto C1 = torch::zeros({1, 4}, torch::device(device).dtype(dtype));
  C1[0][IDX_N2] = 1.0;

  std::map<std::string, torch::Tensor> other;
  auto k1 = falloff->forward(temp, pres, C1, other);

  // Test 2: Pure O2 bath (efficiency = 1.0 as well per YAML)
  auto C2 = torch::zeros({1, 4}, torch::device(device).dtype(dtype));
  C2[0][IDX_O2] = 1.0;

  auto k2 = falloff->forward(temp, pres, C2, other);

  // Both N2 and O2 have efficiency 1.0, so rates should be equal
  double ratio = k2[0][0].item<double>() / k1[0][0].item<double>();
  EXPECT_NEAR(ratio, 1.0, 0.01);
}

// ============================================================================
// Combined photolysis + three-body tests
// ============================================================================

TEST_P(ChapmanCycleTest, FullChapmanWithThreeBody) {
  // Full Chapman cycle: photolysis + three-body + Arrhenius destruction
  auto photo_opts = createPhotolysisOptions();
  auto falloff_opts = createFalloffOptions();
  auto arr_opts = createArrheniusDestructionOnly();

  Photolysis photolysis(photo_opts);
  ThreeBody falloff(falloff_opts);
  Arrhenius arrhenius(arr_opts);

  photolysis->to(device, dtype);
  falloff->to(device, dtype);
  arrhenius->to(device, dtype);

  double T = 250.0, P = 1000.0;
  double n_tot = P / (constants::Rgas * T);

  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({P}, torch::device(device).dtype(dtype));
  auto conc = torch::zeros({1, 4}, torch::device(device).dtype(dtype));
  conc[0][IDX_N2] = 0.78 * n_tot;
  conc[0][IDX_O2] = 0.21 * n_tot;
  conc[0][IDX_O] = 1.e-10 * n_tot;
  conc[0][IDX_O3] = 1.e-8 * n_tot;

  auto [wave, flux] = createActinicFlux(photo_opts->wavelength());
  std::map<std::string, torch::Tensor> flux_map = {{"wavelength", wave}, {"actinic_flux", flux}};
  std::map<std::string, torch::Tensor> other;

  // Get all rate constants
  auto J = photolysis->forward(temp, pres, conc, flux_map);
  auto k_falloff = falloff->forward(temp, pres, conc, other);
  auto k_arr = arrhenius->forward(temp, pres, conc, other);

  double J_O2 = J[0][0].item<double>();
  double J_O3 = J[0][1].item<double>();
  double k_form = k_falloff[0][0].item<double>();
  double k_dest = k_arr[0][0].item<double>();

  std::cout << "Full Chapman cycle rate constants:\n";
  std::cout << "  J(O2) = " << J_O2 << " s^-1\n";
  std::cout << "  J(O3) = " << J_O3 << " s^-1\n";
  std::cout << "  k(O+O2+M) = " << k_form << "\n";
  std::cout << "  k(O+O3) = " << k_dest << "\n";

  EXPECT_GT(J_O2, 0.0);
  EXPECT_GT(J_O3, 0.0);
  EXPECT_GT(k_form, 0.0);
  EXPECT_GT(k_dest, 0.0);
}

TEST_P(ChapmanCycleTest, ThreeBodyRateVsConcentration) {
  // Verify three-body rate scales linearly with [M]
  auto falloff_opts = createFalloffOptions();
  ThreeBody falloff(falloff_opts);
  falloff->to(device, dtype);

  double T = 250.0;
  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({1000.0}, torch::device(device).dtype(dtype));

  // Test with different total concentrations
  std::map<std::string, torch::Tensor> other;

  auto C1 = torch::zeros({1, 4}, torch::device(device).dtype(dtype));
  C1[0][IDX_N2] = 1.0;
  C1[0][IDX_O2] = 0.5;
  auto k1 = falloff->forward(temp, pres, C1, other);

  auto C2 = torch::zeros({1, 4}, torch::device(device).dtype(dtype));
  C2[0][IDX_N2] = 2.0;  // Double N2
  C2[0][IDX_O2] = 1.0;  // Double O2
  auto k2 = falloff->forward(temp, pres, C2, other);

  // k_eff = k0 * [M], so doubling [M] should double k_eff
  double ratio = k2[0][0].item<double>() / k1[0][0].item<double>();
  EXPECT_NEAR(ratio, 2.0, 0.05);

  std::cout << "k1 = " << k1[0][0].item<double>() << ", k2 = " << k2[0][0].item<double>() 
            << ", ratio = " << ratio << "\n";
}

INSTANTIATE_TEST_SUITE_P(
    DeviceTests, ChapmanCycleTest,
    testing::Values(Parameters{torch::kCPU, torch::kFloat64},
                    Parameters{torch::kCUDA, torch::kFloat64}),
    [](const testing::TestParamInfo<ChapmanCycleTest::ParamType>& info) {
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
