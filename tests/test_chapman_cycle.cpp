//! @file test_chapman_cycle.cpp
//! @brief Chapman cycle benchmark: photolysis + three-body reactions

#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include <kintera/constants.h>
#include <kintera/kinetics/arrhenius.hpp>
#include <kintera/kinetics/kinetics.hpp>
#include <kintera/kinetics/evolve_implicit.hpp>
#include <kintera/kinetics/photolysis.hpp>
#include <kintera/kinetics/three_body.hpp>
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

static torch::Tensor build_stoich(const std::vector<Reaction>& rxns,
                                   int nspecies) {
  int nreaction = rxns.size();
  auto stoich = torch::zeros({nspecies, nreaction}, torch::kFloat64);
  for (int j = 0; j < nreaction; j++) {
    for (int i = 0; i < nspecies; i++) {
      auto it = rxns[j].reactants().find(kintera::species_names[i]);
      if (it != rxns[j].reactants().end()) stoich[i][j] = -it->second;
      it = rxns[j].products().find(kintera::species_names[i]);
      if (it != rxns[j].products().end()) stoich[i][j] = it->second;
    }
  }
  return stoich;
}

// Compute species tendencies from rate constants via mass-action kinetics.
// rc: (..., nreaction), conc: (..., nspecies), stoich: (nspecies, nreaction)
static torch::Tensor apply_mass_action(torch::Tensor rc,
                                        torch::Tensor conc,
                                        torch::Tensor stoich) {
  auto react_order = stoich.clamp_max(0.0).abs();
  auto conc_prod = conc.unsqueeze(-1).pow(react_order).prod(-2);
  auto rates = rc * conc_prod;
  return torch::matmul(rates, stoich.t());
}

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

    int nwave = opts->wavelength().size();

    // O2 cross-section (2 branches: absorption + dissociation)
    // Layout: [w0_abs, w0_diss, w1_abs, w1_diss, ...] — nwave * nbranch values
    std::vector<double> xs_O2;
    for (int w = 0; w < nwave; w++) {
      double wl = opts->wavelength()[w];
      double xs_abs = 7.e-18 * std::exp(-std::pow(wl - 145, 2) / 400);
      xs_O2.push_back(xs_abs);
      double xs_diss =
          wl < 240 ? 1.e-17 * std::exp(-std::pow(wl - 160, 2) / 800) : 0.0;
      xs_O2.push_back(xs_diss);
    }
    opts->cross_section() = xs_O2;

    opts->branches().push_back({
        {{"O2", 1.0}},
        {{"O", 2.0}},
    });
    opts->branch_names().push_back({"absorption", "dissociation"});

    // O3 cross-section (2 branches)
    std::vector<double> xs_O3;
    for (int w = 0; w < nwave; w++) {
      double wl = opts->wavelength()[w];
      double xs_abs = 1.e-17 * std::exp(-std::pow(wl - 255, 2) / 800);
      xs_O3.push_back(xs_abs);
      double xs_diss =
          wl < 320 ? 1.e-17 * std::exp(-std::pow(wl - 300, 2) / 1200) : 0.0;
      xs_O3.push_back(xs_diss);
    }
    opts->cross_section().insert(opts->cross_section().end(), xs_O3.begin(),
                                 xs_O3.end());

    opts->branches().push_back({
        {{"O3", 1.0}},
        {{"O2", 1.0}, {"O", 1.0}},
    });
    opts->branch_names().push_back({"absorption", "dissociation"});

    return opts;
  }

  // Arrhenius options with A in molecule,cm,s units
  ArrheniusOptions createArrheniusOptions() {
    auto opts = ArrheniusOptionsImpl::create();

    opts->reactions().push_back(Reaction("O + O2 => O3"));
    opts->reactions().push_back(Reaction("O + O3 => 2 O2"));

    opts->A() = {1.7e-14, 8.0e-12};
    opts->b() = {-2.4, 0.0};
    opts->Ea_R() = {0.0, 2060.0};

    return opts;
  }

  ArrheniusOptions createArrheniusDestructionOnly() {
    auto opts = ArrheniusOptionsImpl::create();
    opts->reactions().push_back(Reaction("O + O3 => 2 O2"));
    opts->A() = {8.0e-12};
    opts->b() = {0.0};
    opts->Ea_R() = {2060.0};
    return opts;
  }

  // Three-body: O + O2 + M -> O3 + M, A in molecule,cm,s units (no YAML conversion)
  ThreeBodyOptions createThreeBodyOptions() {
    auto opts = ThreeBodyOptionsImpl::create();
    auto rxn = Reaction("O + O2 + M <=> O3 + M");
    rxn.efficiencies({{"N2", 1.0}, {"O2", 1.0}});
    opts->reactions().push_back(rxn);
    opts->k0_A() = {6.0e-34};
    opts->k0_b() = {-2.4};
    opts->k0_Ea_R() = {0.0};
    opts->efficiencies().push_back({{"N2", 1.0}, {"O2", 1.0}});
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

  // Number density in molecule/cm^3
  double number_density(double P_Pa, double T_K) {
    return P_Pa / (constants::KBoltz * T_K) * 1.e-6;
  }
};

TEST_P(ChapmanCycleTest, PhotolysisRatesInRange) {
  auto photo_opts = createPhotolysisOptions();
  Photolysis photolysis(photo_opts);
  int nspecies = kintera::species_names.size();
  auto stoich = build_stoich(photo_opts->reactions(), nspecies).to(device, dtype);
  photolysis->to(device, dtype);

  auto [wave, flux] = createActinicFlux(photo_opts->wavelength());

  double T = 250.0, P = 1000.0;
  double n_tot = number_density(P, T);

  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({P}, torch::device(device).dtype(dtype));
  auto conc = torch::zeros({1, nspecies}, torch::device(device).dtype(dtype));
  conc[0][IDX_N2] = 0.78 * n_tot;
  conc[0][IDX_O2] = 0.21 * n_tot;
  conc[0][IDX_O] = 1.e-10 * n_tot;
  conc[0][IDX_O3] = 1.e-8 * n_tot;

  std::map<std::string, torch::Tensor> other;
  other["wavelength"] = wave;
  other["actinic_flux"] = flux;

  auto rc = photolysis->forward(temp, pres, conc, other);
  auto du = apply_mass_action(rc, conc, stoich);

  EXPECT_LT(du[0][IDX_O2].item<double>(), 0.0);
  EXPECT_GT(du[0][IDX_O].item<double>(), 0.0);

  std::cout << "Photolysis J_O2 = " << rc[0][0].item<double>()
            << ", J_O3 = " << rc[0][1].item<double>()
            << ", du[O2] = " << du[0][IDX_O2].item<double>()
            << ", du[O] = " << du[0][IDX_O].item<double>() << "\n";
}

TEST_P(ChapmanCycleTest, ArrheniusRatesAtTemperature) {
  auto arr_opts = createArrheniusOptions();
  Arrhenius arrhenius(arr_opts);
  int nspecies = kintera::species_names.size();
  auto stoich = build_stoich(arr_opts->reactions(), nspecies).to(device, dtype);
  arrhenius->to(device, dtype);

  double T = 250.0;
  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({1000.0}, torch::device(device).dtype(dtype));
  auto conc = torch::ones({1, nspecies}, torch::device(device).dtype(dtype));

  auto rc = arrhenius->forward(temp, pres, conc, {});
  auto du = apply_mass_action(rc, conc, stoich);

  EXPECT_LT(du[0][IDX_O].item<double>(), 0.0);

  std::cout << "Arrhenius rc: " << rc << "\n";
  std::cout << "Arrhenius du: " << du << "\n";
}

TEST_P(ChapmanCycleTest, MassConservation) {
  auto photo_opts = createPhotolysisOptions();
  auto arr_opts = createArrheniusOptions();

  Photolysis photolysis(photo_opts);
  Arrhenius arrhenius(arr_opts);
  int nspecies = kintera::species_names.size();
  auto stoich_photo = build_stoich(photo_opts->reactions(), nspecies).to(device, dtype);
  auto stoich_arr = build_stoich(arr_opts->reactions(), nspecies).to(device, dtype);

  auto [wave, flux] = createActinicFlux(photo_opts->wavelength());
  photolysis->to(device, dtype);
  arrhenius->to(device, dtype);

  double T = 250.0, P = 1000.0;
  double n_tot = number_density(P, T);

  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({P}, torch::device(device).dtype(dtype));

  auto conc = torch::zeros({1, nspecies}, torch::device(device).dtype(dtype));
  conc[0][IDX_N2] = 0.78 * n_tot;
  conc[0][IDX_O2] = 0.21 * n_tot;
  conc[0][IDX_O] = 1.e-10 * n_tot;
  conc[0][IDX_O3] = 1.e-8 * n_tot;

  double O_initial = 2.0 * conc[0][IDX_O2].item<double>() +
                     conc[0][IDX_O].item<double>() +
                     3.0 * conc[0][IDX_O3].item<double>();

  std::map<std::string, torch::Tensor> photo_other;
  photo_other["wavelength"] = wave;
  photo_other["actinic_flux"] = flux;

  // Small dt required for stability: O atom equilibrium timescale ~ J/k ~ 1e-3 s
  double dt = 1.e-5;
  int nsteps = 100000;

  for (int step = 0; step < nsteps; step++) {
    auto rc_photo = photolysis->forward(temp, pres, conc, photo_other);
    auto du_photo = apply_mass_action(rc_photo, conc, stoich_photo);

    auto rc_arr = arrhenius->forward(temp, pres, conc, {});
    auto du_arr = apply_mass_action(rc_arr, conc, stoich_arr);

    conc = (conc + (du_photo + du_arr) * dt).clamp_min(0.0);
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
  int nspecies = kintera::species_names.size();
  auto stoich_photo = build_stoich(photo_opts->reactions(), nspecies).to(device, dtype);
  auto stoich_arr = build_stoich(arr_opts->reactions(), nspecies).to(device, dtype);

  auto [wave, flux] = createActinicFlux(photo_opts->wavelength());
  photolysis->to(device, dtype);
  arrhenius->to(device, dtype);

  double T = 250.0, P = 1000.0;
  double n_tot = number_density(P, T);

  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({P}, torch::device(device).dtype(dtype));

  auto conc = torch::zeros({1, nspecies}, torch::device(device).dtype(dtype));
  conc[0][IDX_N2] = 0.78 * n_tot;
  conc[0][IDX_O2] = 0.21 * n_tot;
  conc[0][IDX_O] = 1.e-10 * n_tot;
  conc[0][IDX_O3] = 1.e-8 * n_tot;

  std::map<std::string, torch::Tensor> photo_other;
  photo_other["wavelength"] = wave;
  photo_other["actinic_flux"] = flux;

  double dt = 1.e-4;
  int nsteps = 100000;
  double prev_O3 = 0.0;

  for (int step = 0; step < nsteps; step++) {
    auto rc_photo = photolysis->forward(temp, pres, conc, photo_other);
    auto du_photo = apply_mass_action(rc_photo, conc, stoich_photo);

    auto rc_arr = arrhenius->forward(temp, pres, conc, {});
    auto du_arr = apply_mass_action(rc_arr, conc, stoich_arr);

    conc = (conc + (du_photo + du_arr) * dt).clamp_min(0.0);

    if (step % 10000 == 0 && step > 0) {
      double c_O3 = conc[0][IDX_O3].item<double>();
      double O3_change = std::abs(c_O3 - prev_O3) / (prev_O3 + 1e-30);
      if (O3_change < 1e-4) break;
      prev_O3 = c_O3;
    }
  }

  double final_O3 = conc[0][IDX_O3].item<double>();
  double O3_ppm = final_O3 / n_tot * 1.e6;

  EXPECT_GE(O3_ppm, 0.0);
  EXPECT_LT(O3_ppm, 1000);

  std::cout << "Steady-state O3: " << O3_ppm << " ppm\n";
}

// ============================================================================
// Three-body reaction tests
// ============================================================================

TEST_P(ChapmanCycleTest, ThreeBodyOzoneFormation) {
  auto tb_opts = createThreeBodyOptions();
  ThreeBody three_body(tb_opts);
  int nspecies = kintera::species_names.size();
  auto stoich = build_stoich(tb_opts->reactions(), nspecies).to(device, dtype);
  three_body->to(device, dtype);

  double T = 220.0, P = 5000.0;
  double n_tot = number_density(P, T);

  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({P}, torch::device(device).dtype(dtype));
  auto conc = torch::zeros({1, nspecies}, torch::device(device).dtype(dtype));
  conc[0][IDX_N2] = 0.78 * n_tot;
  conc[0][IDX_O2] = 0.21 * n_tot;
  conc[0][IDX_O] = 1.e-10 * n_tot;
  conc[0][IDX_O3] = 1.e-8 * n_tot;

  auto rc = three_body->forward(temp, pres, conc, {});
  auto du = apply_mass_action(rc, conc, stoich);

  EXPECT_GT(du[0][IDX_O3].item<double>(), 0.0);
  EXPECT_LT(du[0][IDX_O].item<double>(), 0.0);

  std::cout << "Three-body du[O3] = " << du[0][IDX_O3].item<double>() << "\n";
}

TEST_P(ChapmanCycleTest, ThreeBodyTemperatureDependence) {
  auto tb_opts = createThreeBodyOptions();
  ThreeBody three_body(tb_opts);
  int nspecies = kintera::species_names.size();
  three_body->to(device, dtype);

  double n_tot = number_density(5000.0, 250.0);
  auto conc = torch::zeros({1, nspecies}, torch::device(device).dtype(dtype));
  conc[0][IDX_N2] = 0.78 * n_tot;
  conc[0][IDX_O2] = 0.21 * n_tot;
  conc[0][IDX_O] = 1e-10 * n_tot;
  conc[0][IDX_O3] = 1e-8 * n_tot;

  auto stoich = build_stoich(tb_opts->reactions(), nspecies).to(device, dtype);
  auto pres = torch::tensor({5000.0}, torch::device(device).dtype(dtype));

  auto t200 = torch::tensor({200.0}, torch::device(device).dtype(dtype));
  auto t250 = torch::tensor({250.0}, torch::device(device).dtype(dtype));
  auto t300 = torch::tensor({300.0}, torch::device(device).dtype(dtype));

  auto du200 = apply_mass_action(three_body->forward(t200, pres, conc, {}), conc, stoich);
  auto du250 = apply_mass_action(three_body->forward(t250, pres, conc, {}), conc, stoich);
  auto du300 = apply_mass_action(three_body->forward(t300, pres, conc, {}), conc, stoich);

  // With b = -2.4 < 0, rate increases as T decreases
  double du_O3_200 = du200[0][IDX_O3].item<double>();
  double du_O3_250 = du250[0][IDX_O3].item<double>();
  double du_O3_300 = du300[0][IDX_O3].item<double>();

  EXPECT_GT(du_O3_200, du_O3_250);
  EXPECT_GT(du_O3_250, du_O3_300);

  std::cout << "du[O3](200K) = " << du_O3_200 << ", du[O3](250K) = " << du_O3_250
            << ", du[O3](300K) = " << du_O3_300 << "\n";
}

TEST_P(ChapmanCycleTest, ThreeBodyEfficiencyScaling) {
  auto tb_opts = createThreeBodyOptions();
  ThreeBody three_body(tb_opts);
  int nspecies = kintera::species_names.size();
  auto stoich = build_stoich(tb_opts->reactions(), nspecies).to(device, dtype);
  three_body->to(device, dtype);

  double T = 250.0;
  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({1000.0}, torch::device(device).dtype(dtype));

  auto C1 = torch::zeros({1, nspecies}, torch::device(device).dtype(dtype));
  C1[0][IDX_N2] = 1.0;
  C1[0][IDX_O2] = 0.1;
  C1[0][IDX_O] = 0.1;

  auto du1 = apply_mass_action(three_body->forward(temp, pres, C1, {}), C1, stoich);

  auto C2 = torch::zeros({1, nspecies}, torch::device(device).dtype(dtype));
  C2[0][IDX_O2] = 1.1;
  C2[0][IDX_O] = 0.1;

  auto du2 = apply_mass_action(three_body->forward(temp, pres, C2, {}), C2, stoich);

  double du1_O3 = du1[0][IDX_O3].item<double>();
  double du2_O3 = du2[0][IDX_O3].item<double>();
  EXPECT_GT(du1_O3, 0.0);
  EXPECT_GT(du2_O3, 0.0);
}

// ============================================================================
// Combined photolysis + three-body tests
// ============================================================================

TEST_P(ChapmanCycleTest, FullChapmanWithThreeBody) {
  auto photo_opts = createPhotolysisOptions();
  auto tb_opts = createThreeBodyOptions();
  auto arr_opts = createArrheniusDestructionOnly();

  Photolysis photolysis(photo_opts);
  ThreeBody three_body(tb_opts);
  Arrhenius arrhenius(arr_opts);

  int nspecies = kintera::species_names.size();
  auto stoich_photo = build_stoich(photo_opts->reactions(), nspecies).to(device, dtype);
  auto stoich_tb = build_stoich(tb_opts->reactions(), nspecies).to(device, dtype);
  auto stoich_arr = build_stoich(arr_opts->reactions(), nspecies).to(device, dtype);

  auto [wave, flux] = createActinicFlux(photo_opts->wavelength());

  photolysis->to(device, dtype);
  three_body->to(device, dtype);
  arrhenius->to(device, dtype);

  double T = 250.0, P = 1000.0;
  double n_tot = number_density(P, T);

  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({P}, torch::device(device).dtype(dtype));
  auto conc = torch::zeros({1, nspecies}, torch::device(device).dtype(dtype));
  conc[0][IDX_N2] = 0.78 * n_tot;
  conc[0][IDX_O2] = 0.21 * n_tot;
  conc[0][IDX_O] = 1.e-10 * n_tot;
  conc[0][IDX_O3] = 1.e-8 * n_tot;

  std::map<std::string, torch::Tensor> photo_other;
  photo_other["wavelength"] = wave;
  photo_other["actinic_flux"] = flux;

  auto du_photo = apply_mass_action(
      photolysis->forward(temp, pres, conc, photo_other), conc, stoich_photo);
  auto du_tb = apply_mass_action(
      three_body->forward(temp, pres, conc, {}), conc, stoich_tb);
  auto du_arr = apply_mass_action(
      arrhenius->forward(temp, pres, conc, {}), conc, stoich_arr);

  auto du = du_photo + du_tb + du_arr;

  EXPECT_GT(torch::sum(torch::abs(du)).item<double>(), 0.0);

  std::cout << "Full Chapman du: " << du << "\n";
}

TEST_P(ChapmanCycleTest, ThreeBodyRateVsConcentration) {
  auto tb_opts = createThreeBodyOptions();
  ThreeBody three_body(tb_opts);
  int nspecies = kintera::species_names.size();
  auto stoich = build_stoich(tb_opts->reactions(), nspecies).to(device, dtype);
  three_body->to(device, dtype);

  double T = 250.0;
  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({1000.0}, torch::device(device).dtype(dtype));

  auto C1 = torch::zeros({1, nspecies}, torch::device(device).dtype(dtype));
  C1[0][IDX_N2] = 1.0;
  C1[0][IDX_O2] = 0.5;
  C1[0][IDX_O] = 0.1;

  auto C2 = torch::zeros({1, nspecies}, torch::device(device).dtype(dtype));
  C2[0][IDX_N2] = 2.0;
  C2[0][IDX_O2] = 0.5;
  C2[0][IDX_O] = 0.1;

  auto du1 = apply_mass_action(three_body->forward(temp, pres, C1, {}), C1, stoich);
  auto du2 = apply_mass_action(three_body->forward(temp, pres, C2, {}), C2, stoich);

  // k_eff = k0 * [M], [M]_1 = 1.0 + 0.5 = 1.5, [M]_2 = 2.0 + 0.5 = 2.5
  double du1_O3 = du1[0][IDX_O3].item<double>();
  double du2_O3 = du2[0][IDX_O3].item<double>();
  double ratio = du2_O3 / du1_O3;
  double expected_ratio = 2.5 / 1.5;
  EXPECT_NEAR(ratio, expected_ratio, expected_ratio * 0.05);

  std::cout << "du1[O3] = " << du1_O3 << ", du2[O3] = " << du2_O3
            << ", ratio = " << ratio << " (expected " << expected_ratio << ")\n";
}

TEST_P(ChapmanCycleTest, TimeMarching) {
  kintera::species_initialized = false;
  kintera::species_names = {"N2", "O2", "O", "O3"};

  auto op_kinet = KineticsOptionsImpl::from_yaml("chapman_cycle.yaml");
  ASSERT_NE(op_kinet, nullptr) << "Failed to load chapman_cycle.yaml";
  Kinetics kinet(op_kinet);
  kinet->to(device, dtype);

  auto species = op_kinet->species();
  int nspecies = species.size();

  std::cout << "Species: ";
  for (auto& s : species) std::cout << s << " ";
  std::cout << "\nStoich:\n" << kinet->stoich << "\n";

  int nreaction = kinet->stoich.size(1);
  std::cout << "Reactions: " << nreaction << "\n";

  // Wavelength grid matching the YAML cross-section data
  std::vector<double> wl_vec;
  for (int w = 100; w <= 320; w += 10) wl_vec.push_back(w);
  auto wavelength = torch::tensor(wl_vec, torch::device(device).dtype(dtype));

  // Actinic flux at those wavelengths
  auto actinic_flux = torch::zeros_like(wavelength);
  for (int i = 0; i < (int)wl_vec.size(); i++) {
    double w = wl_vec[i];
    if (w < 200)
      actinic_flux[i] = 1.e10 * std::exp(-(200 - w) / 30);
    else if (w < 320)
      actinic_flux[i] = 1.e13 * std::exp(-std::pow(w - 250, 2) / 5000);
    else
      actinic_flux[i] = 1.e14;
  }

  std::map<std::string, torch::Tensor> extra;
  extra["wavelength"] = wavelength;
  extra["actinic_flux"] = actinic_flux;

  // Initial conditions in mol/m^3
  double T = 250.0, P = 0.01;  // Pa
  double n_tot = P / (constants::Rgas * T);  // mol/m^3
  auto temp = torch::tensor({T}, torch::device(device).dtype(dtype));
  auto pres = torch::tensor({P}, torch::device(device).dtype(dtype));

  // Scalar (0D) tensors for single-point computation
  auto conc = torch::zeros({nspecies}, torch::device(device).dtype(dtype));
  for (int i = 0; i < nspecies; i++) {
    if (species[i] == "N2")  conc[i] = 0.79 * n_tot;
    else if (species[i] == "O2")  conc[i] = 0.21 * n_tot;
    else if (species[i] == "O")   conc[i] = 1.e-10 * n_tot;
    else if (species[i] == "O3")  conc[i] = 1.e-8 * n_tot;
  }

  int idx_O = -1, idx_O2 = -1, idx_O3 = -1;
  for (int i = 0; i < nspecies; i++) {
    if (species[i] == "O") idx_O = i;
    if (species[i] == "O2") idx_O2 = i;
    if (species[i] == "O3") idx_O3 = i;
  }

  double dt = 1.0;
  std::cout << "\nTime marching with Kinetics + evolve_implicit:\n";

  for (int step = 0; step < 500; step++) {
    // Kinetics::forward expects temp shape to define batch dims;
    // for 0D, use scalars and unsqueeze conc to match
    auto [rate, rc_ddC, rc_ddT] = kinet->forward(temp, pres, conc.unsqueeze(0), extra);

    // Extract single-point results (remove batch dim)
    auto rate0 = rate.squeeze(0);
    auto cvol = torch::ones({1}, torch::device(device).dtype(dtype));
    auto jac = kinet->jacobian(temp, conc.unsqueeze(0), cvol, rate, rc_ddC, rc_ddT);
    auto jac0 = jac.squeeze(0);

    auto delta = evolve_implicit(rate0, kinet->stoich, jac0, dt);
    conc = (conc + delta).clamp_min(0.0);

    double rel_change = (delta.abs() / (conc.abs() + 1e-30)).max().item<double>();
    if (rel_change < 0.5)
      dt = std::min(dt * 1.5, 1.e6);
    else if (rel_change > 2.0)
      dt = std::max(dt * 0.5, 1.e-14);

    if (step % 50 == 0 || step == 499) {
      double total = conc.sum().item<double>();
      std::cout << "  step " << step << " dt=" << dt
                << " O=" << conc[idx_O].item<double>() / total
                << " O2=" << conc[idx_O2].item<double>() / total
                << " O3=" << conc[idx_O3].item<double>() / total << "\n";
    }

    if (step > 100 && rel_change < 1e-10) {
      std::cout << "  Converged at step " << step << "\n";
      break;
    }
  }

  double total = conc.sum().item<double>();
  double mix_O  = conc[idx_O].item<double>() / total;
  double mix_O2 = conc[idx_O2].item<double>() / total;
  double mix_O3 = conc[idx_O3].item<double>() / total;

  EXPECT_GT(mix_O, 1e-4) << "O should build up from photolysis";
  EXPECT_GT(mix_O3, 1e-4) << "O3 should build up from O + O2 reactions";
  EXPECT_LT(mix_O2, 0.21) << "O2 should be partially consumed";

  std::cout << "Final: O=" << mix_O << " O2=" << mix_O2 << " O3=" << mix_O3 << "\n";
}

INSTANTIATE_TEST_SUITE_P(
    DeviceTests, ChapmanCycleTest,
    testing::Values(Parameters{torch::kCPU, torch::kFloat64},
                    Parameters{torch::kMPS, torch::kFloat32},
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
