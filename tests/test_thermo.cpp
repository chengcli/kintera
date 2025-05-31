// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/constants.h>

#include <kintera/eos/equation_of_state.hpp>
#include <kintera/thermo/thermo.hpp>
#include <kintera/thermo/thermo_formatter.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;

TEST_P(DeviceTest, feps) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");

  ThermoY thermo(op_thermo);
  thermo->to(device, dtype);

  int ny = thermo->options.vapor_ids().size() + thermo->options.cloud_ids().size();
  auto yfrac =
      torch::zeros({ny, 1, 2, 3}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

  auto feps = thermo->f_eps(yfrac);
  std::cout << "feps = " << feps << std::endl;

  auto fsig = thermo->f_sig(yfrac);
  std::cout << "fsig = " << fsig << std::endl;
}

TEST_P(DeviceTest, thermo_y) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");

  ThermoY thermo(op_thermo);
  thermo->to(device, dtype);

  int ny = thermo->options.vapor_ids().size() + thermo->options.cloud_ids().size();
  auto yfrac =
      torch::zeros({ny, 1, 2, 3}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

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

  ThermoX thermo(op_thermo);
  thermo->to(device, dtype);

  int ny = thermo->options.vapor_ids().size() + thermo->options.cloud_ids().size();
  auto xfrac =
      torch::zeros({1, 2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  auto yfrac = thermo->get_mass_fraction(xfrac);
  std::cout << "yfrac = " << yfrac << std::endl;
}

TEST_P(DeviceTest, thermo_xy) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");
  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  int ny = op_thermo.vapor_ids().size() + op_thermo.cloud_ids().size();
  auto xfrac =
      torch::zeros({1, 2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  auto yfrac = thermo_x->get_mass_fraction(xfrac);
  auto xfrac2 = thermo_y->get_mole_fraction(yfrac);

  EXPECT_EQ(torch::allclose(xfrac, xfrac2, /*rtol=*/1e-4, /*atol=*/1e-4), true);
}

TEST_P(DeviceTest, thermo_yx) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");
  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  int ny = op_thermo.vapor_ids().size() + op_thermo.cloud_ids().size();
  auto yfrac =
      torch::zeros({ny, 1, 2, 3}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

  auto xfrac = thermo_y->get_mole_fraction(yfrac);
  auto yfrac2 = thermo_x->get_mass_fraction(xfrac);

  EXPECT_EQ(torch::allclose(yfrac, yfrac2, /*rtol=*/1e-4, /*atol=*/1e-4), true);
}

TEST_P(DeviceTest, eng_pres) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");
  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  int ny = thermo_y->options.vapor_ids().size() + thermo_y->options.cloud_ids().size();
  auto yfrac = torch::zeros({ny, 1}, torch::device(device).dtype(dtype));
  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

  auto temp = 200.0 * torch::ones({1}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({1}, torch::device(device).dtype(dtype));

  auto xfrac = thermo_y->get_mole_fraction(yfrac);
  auto rho = thermo_x->get_density(temp, pres, xfrac);

  auto intEng = thermo_y->get_intEng(rho, pres, yfrac);
  auto pres2 = thermo_y->get_pres(rho, intEng, yfrac);
  EXPECT_EQ(torch::allclose(pres, pres2, /*rtol=*/1e-4, /*atol=*/1e-4), true);

  auto temp2 = thermo_y->get_temp(rho, pres, yfrac);
  EXPECT_EQ(torch::allclose(temp, temp2, /*rtol=*/1e-4, /*atol=*/1e-4), true);
}

TEST_P(DeviceTest, equilibrate_tp) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");
  op_thermo.max_iter(10);
  std::cout << fmt::format("{}", op_thermo) << std::endl;

  int ny = op_thermo.vapor_ids().size() + op_thermo.cloud_ids().size();

  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  auto xfrac =
      torch::zeros({1, 2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  auto temp = 200.0 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));

  std::cout << "xfrac before = " << xfrac[0][0][0] << std::endl;

  thermo_x->forward(temp, pres, xfrac);

  std::cout << "xfrac after = " << xfrac[0][0][0] << std::endl;
}

/*TEST_P(DeviceTest, equilibrate_uv) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");
  std::cout << fmt::format("{}", op_thermo) << std::endl;

  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  int ny = thermo_y->options.vapor_ids().size() + thermo_y->options.cloud_ids().size();
  auto yfrac =
      torch::zeros({ny, 1, 2, 3}, torch::device(device).dtype(dtype));
  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

  auto rho = torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto intEng = thermo_y->get_intEng(rho, pres, yfrac);
  std::cout << "intEng = " << intEng << std::endl;

  auto result = thermo_y->forward(rho, intEng, yfrac);
  auto yfrac2 = yfrac + result;

  auto pres2 = thermo_y->get_pres(rho, intEng, yfrac2);
  std::cout << "pres2 = " << pres2 << std::endl;

  auto intEng2 = thermo_y->get_intEng(rho, pres2, yfrac2);
  std::cout << "intEng2 = " << intEng2 << std::endl;

  auto result2 = thermo_y->forward(rho, intEng, yfrac2);
  auto yfrac3 = yfrac2 + result;

  auto pres3 = thermo_y->get_pres(rho, intEng, yfrac3);
  std::cout << "pres3 = " << pres3 << std::endl;

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
}*/

/*TEST_P(DeviceTest, earth) {
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
