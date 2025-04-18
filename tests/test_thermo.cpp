// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/constants.h>

#include <kintera/eos/equation_of_state.hpp>
#include <kintera/thermo/thermo.hpp>
#include <kintera/thermo/thermo_formatter.hpp>
#include <kintera/vapors/water_vapors.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;

TEST_P(DeviceTest, feps) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");
  std::cout << fmt::format("{}", op_thermo) << std::endl;

  ThermoY thermo(op_thermo);
  thermo->to(device, dtype);

  int nspecies = thermo->options.size();
  auto yfrac =
      torch::zeros({nspecies, 1, 2, 3}, torch::device(device).dtype(dtype));

  for (int i = 0; i < nspecies; ++i) {
    yfrac[i] = 0.01 * (i + 1);
  }

  auto feps = thermo->f_eps(yfrac);
  std::cout << "feps = " << feps << std::endl;

  auto fsig = thermo->f_sig(yfrac);
  std::cout << "fsig = " << fsig << std::endl;

  auto fpsi = thermo->f_psi(yfrac);
  std::cout << "fpsi = " << fpsi << std::endl;
}

TEST_P(DeviceTest, thermo_y) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");
  std::cout << fmt::format("{}", op_thermo) << std::endl;

  ThermoY thermo(op_thermo);
  thermo->to(device, dtype);

  int nspecies = thermo->options.size();
  auto yfrac =
      torch::zeros({nspecies, 1, 2, 3}, torch::device(device).dtype(dtype));

  for (int i = 0; i < nspecies; ++i) {
    yfrac[i] = 0.01 * (i + 1);
  }

  auto mu = thermo->get_mu();
  std::cout << "mu = " << mu << std::endl;

  auto cv = thermo->get_cv();
  std::cout << "cv = " << cv << std::endl;

  auto cp = thermo->get_cp();
  std::cout << "cp = " << cp << std::endl;

  auto xfrac = thermo->get_mole_fraction(yfrac);
  std::cout << "xfrac = " << xfrac << std::endl;

  EXPECT_EQ(torch::allclose(
                xfrac.sum(-1),
                torch::ones({1, 2, 3}, torch::device(device).dtype(dtype)),
                /*rtol=*/1e-4, /*atol=*/1e-4),
            true);

  auto rho = torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto conc = thermo->get_concentration(rho, yfrac);

  std::cout << "conc = " << conc << std::endl;
}

TEST_P(DeviceTest, thermo_x) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");
  std::cout << fmt::format("{}", op_thermo) << std::endl;

  ThermoX thermo(op_thermo);
  thermo->to(device, dtype);

  int nspecies = thermo->options.size();
  auto xfrac =
      torch::zeros({1, 2, 3, 1 + nspecies}, torch::device(device).dtype(dtype));

  for (int i = 0; i < nspecies; ++i) {
    xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  }
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, nspecies).sum(-1);

  auto mu = thermo->get_mu();
  std::cout << "mu = " << mu << std::endl;

  auto cv = thermo->get_cv();
  std::cout << "cv = " << cv << std::endl;

  auto cp = thermo->get_cp();
  std::cout << "cp = " << cp << std::endl;

  auto yfrac = thermo->get_mass_fraction(xfrac);
  std::cout << "yfrac = " << yfrac << std::endl;
}

/*TEST_P(DeviceTest, forward) {
  auto op_thermo = ThermodynamicsOptions();
  int nvapor = 1;
  int ncloud = 1;

  op_thermo.nvapor(nvapor).ncloud(ncloud).max_iter(200);
  op_thermo.species().push_back("H2O");
  op_thermo.species().push_back("H2O(l)");

  auto mu = 18.e-3;
  auto Rd = 287.;
  auto gammad = 1.4;
  auto mud = constants::Rgas / Rd;
  auto cvd = Rd / (gammad - 1.);
  auto cpd = cvd + Rd;

  auto cvv = (cvd * mud) / mu;
  auto cpv = cvv + constants::Rgas / mu;
  auto cc = 4.18e3;

  op_thermo.Rd(Rd).gammad_ref(gammad);
  op_thermo.mu_ratio_m1() = std::vector<double>{mud / mu - 1., mud / mu - 1.};
  op_thermo.cv_ratio_m1() = std::vector<double>{cvv / cvd - 1., cc / cvd - 1.};
  op_thermo.cp_ratio_m1() = std::vector<double>{cpv / cpd - 1., cc / cpd - 1.};
  op_thermo.h0() = std::vector<double>{0., 0., -2.5e6};

  Thermodynamics thermo(op_thermo);

  auto w = torch::zeros({5 + nvapor + ncloud, 1, 2, 5}, torch::kFloat64);

  w[index::IDN] = torch::randn({1, 2, 5}, torch::kFloat64).abs();
  w[index::IPR] = torch::randn({1, 2, 5}, torch::kFloat64).abs() * 1.e5;

  int ivapor = thermo->species_index("H2O");
  int icloud = thermo->species_index("H2O(l)");

  w[ivapor] = torch::randn({1, 2, 5}, torch::kFloat64).abs() * 0.2;
  w[icloud] = torch::randn({1, 2, 5}, torch::kFloat64).abs() * 0.2;
  w.narrow(0, index::IVX, 3) = torch::randn({3, 1, 2, 5}, torch::kFloat64);

  std::cout << "w = " << w << std::endl;

  auto temp = thermo->get_temp(w);

  std::cout << "temp = " << temp << std::endl;

  auto op_eos = EquationOfStateOptions().thermo(thermo->options);
  auto eos = IdealMoist(op_eos);

  auto u = torch::empty_like(w);
  eos->prim2cons(u, w);
  std::cout << "u = " << u << std::endl;

  auto w2 = eos->forward(u);
  std::cout << "w2 = " << w2 << std::endl;

  auto yfrac = w.slice(0, index::ICY, w.size(0));
  std::cout << "yfrac1 = " << yfrac << std::endl;

  auto xfrac = thermo->get_mole_fraction(yfrac);
  std::cout << "xfrac = " << xfrac << std::endl;
  std::cout << "yfrac2 = " << thermo->get_mass_fraction(xfrac) << std::endl;

  auto dw = thermo->equilibrate_tp(temp, w[index::IPR],
                                   w.slice(0, index::ICY, w.size(0)));
  std::cout << "rate = " << dw << std::endl;
}

TEST_P(DeviceTest, earth) {
  auto op_cond = CondensationOptions();

  auto r = Nucleation(
      "H2O => H2O(l)", "ideal",
      {{"T3", 273.15}, {"P3", 611.7}, {"beta", 24.8}, {"delta", 0.0}});
  op_cond.react().push_back(r);

  auto op_thermo = ThermodynamicsOptions().cond(op_cond);
  int nvapor = 1;
  int ncloud = 1;

  op_thermo.nvapor(nvapor).ncloud(ncloud).max_iter(200);
  op_thermo.species().push_back("H2O");
  op_thermo.species().push_back("H2O(l)");

  auto mu = 18.e-3;
  auto Rd = 287.;
  auto gammad = 1.4;
  auto mud = constants::Rgas / Rd;
  auto cvd = Rd / (gammad - 1.);
  auto cpd = cvd + Rd;

  auto cvv = (cvd * mud) / mu;
  auto cpv = cvv + constants::Rgas / mu;
  auto cc = 4.18e3;

  std::cout << "mud = " << mud << std::endl;
  std::cout << "cvd = " << cvd << std::endl;
  std::cout << "cpd = " << cpd << std::endl;

  op_thermo.Rd(Rd).gammad_ref(gammad);
  op_thermo.mu_ratio_m1() = std::vector<double>{mud / mu - 1., mud / mu - 1.};
  op_thermo.cv_ratio_m1() = std::vector<double>{cvv / cvd - 1., cc / cvd - 1.};
  op_thermo.cp_ratio_m1() = std::vector<double>{cpv / cpd - 1., cc / cpd - 1.};
  op_thermo.h0() = std::vector<double>{0., 0., -2.5e6};

  Thermodynamics thermo(op_thermo);

  std::cout << "mu_ratio_m1 = " << thermo->mu_ratio_m1 << std::endl;
  std::cout << "cv_ratio_m1 = " << thermo->cv_ratio_m1 << std::endl;
  std::cout << "cp_ratio_m1 = " << thermo->cp_ratio_m1 << std::endl;
  std::cout << "h0 = " << thermo->h0 << std::endl;

  double temp0 = 300.;
  double pres = 1.e5;
  double xvapor = 0.1;
  auto u = torch::zeros({5 + nvapor + ncloud, 1, 1, 2}, torch::kFloat64);

  u[0] = (1. - xvapor) * pres / (constants::Rgas * temp0 / mud);

  int ivapor = thermo->species_index("H2O");
  int icloud = thermo->species_index("H2O(l)");

  u[ivapor] = xvapor * pres / (constants::Rgas * temp0 / mu);
  u[index::IPR] = (u[0] * cvd + u[ivapor] * cvv + u[icloud] * cc) * temp0;

  std::cout << "u before = " << u << std::endl;

  auto op_eos = EquationOfStateOptions().thermo(thermo->options);
  auto eos = IdealMoist(op_eos);

  auto w = eos->forward(u);
  std::cout << "w before = " << w << std::endl;

  auto du = thermo->forward(u);
  std::cout << "du = " << du << std::endl;
  std::cout << "u after = " << u << std::endl;

  auto temp = thermo->get_temp(w);
  auto dw = thermo->equilibrate_tp(temp, w[index::IPR],
                                   w.slice(0, index::ICY, w.size(0)));
  w.slice(0, index::ICY, w.size(0)) += dw;
  std::cout << "dw = " << dw << std::endl;
  std::cout << "w after = " << w << std::endl;
}*/

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
