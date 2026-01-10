//! @file test_chapman_cycle.cpp
//! @brief Chapman cycle benchmark for photochemistry validation

#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <kintera/constants.h>
#include <kintera/kinetics/arrhenius.hpp>
#include <kintera/kinetics/photolysis.hpp>
#include <kintera/utils/parse_comp_string.hpp>

#include "device_testing.hpp"

using namespace kintera;

constexpr int IDX_N2 = 0;
constexpr int IDX_O2 = 1;
constexpr int IDX_O = 2;
constexpr int IDX_O3 = 3;

class ChapmanCycleTest : public DeviceTest {
 protected:
  void SetUp() override {
    DeviceTest::SetUp();
    extern std::vector<std::string> species_names;
    species_names = {"N2", "O2", "O", "O3"};
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

  EXPECT_GT(O3_ppm, 0.01);
  EXPECT_LT(O3_ppm, 100);

  std::cout << "Steady-state O3: " << O3_ppm << " ppm\n";
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
