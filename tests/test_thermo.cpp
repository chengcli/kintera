// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/constants.h>

#include <kintera/thermo/eval_uhs.hpp>
#include <kintera/thermo/relative_humidity.hpp>
#include <kintera/thermo/thermo.hpp>
#include <kintera/thermo/thermo_formatter.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;
using namespace torch::indexing;

TEST_P(DeviceTest, thermo_y) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");

  std::cout << fmt::format("{}", op_thermo) << std::endl;

  ThermoY thermo(op_thermo);
  thermo->to(device, dtype);

  int ny = thermo->options.vapor_ids().size() +
           thermo->options.cloud_ids().size() - 1;
  auto yfrac = torch::zeros({ny, 1, 2, 3}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

  ////////// Testing Y->X conversion //////////
  auto xfrac = thermo->compute("Y->X", {yfrac});
  EXPECT_EQ(torch::allclose(
                xfrac.sum(-1),
                torch::ones({1, 2, 3}, torch::device(device).dtype(dtype)),
                /*rtol=*/1e-4, /*atol=*/1e-4),
            true);

  ////////// Testing DY->V conversion //////////
  auto rho = torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto ivol = thermo->compute("DY->V", {rho, yfrac});
  EXPECT_EQ(torch::allclose(rho, ivol.sum(-1),
                            /*rtol=*/1e-4, /*atol=*/1e-4),
            true);

  ////////// Testing V->Y conversion //////////
  auto yfrac2 = thermo->compute("V->Y", {ivol});
  EXPECT_EQ(torch::allclose(yfrac, yfrac2, /*rtol=*/1e-4, /*atol=*/1e-4), true);

  ////////// Testing VT->P conversion //////////
  auto temp = 300. * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto pres = thermo->compute("VT->P", {ivol, temp});

  ////////// Testing PV->T conversion //////////
  auto temp2 = thermo->compute("PV->T", {pres, ivol});
  EXPECT_EQ(torch::allclose(temp, temp2, /*rtol=*/1e-4, /*atol=*/1e-4), true);

  ////////// Testing VT->cv conversion //////////
  auto cv = thermo->compute("VT->cv", {ivol, temp});

  ////////// Testing VT->U conversion //////////
  auto intEng = thermo->compute("VT->U", {ivol, temp});

  ////////// Testing VU->T conversion //////////
  auto temp3 = thermo->compute("VU->T", {ivol, intEng});
  EXPECT_EQ(torch::allclose(temp, temp3, /*rtol=*/1e-4, /*atol=*/1e-4), true);

  ////////// Testing PVT->S conversion //////////
  auto entropy = thermo->compute("PVT->S", {pres, ivol, temp});
  // std::cout << "entropy = " << entropy << std::endl;
}

TEST_P(DeviceTest, thermo_x) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml").max_iter(10);

  ThermoX thermo(op_thermo);
  thermo->to(device, dtype);

  int ny = thermo->options.vapor_ids().size() +
           thermo->options.cloud_ids().size() - 1;
  auto xfrac =
      torch::zeros({1, 2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  /////////// Testing X->Y conversion //////////
  auto yfrac = thermo->compute("X->Y", {xfrac});

  /////////// Testing TDX->V conversion //////////
  auto temp = 300. * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto conc = thermo->compute("TPX->V", {temp, pres, xfrac});

  //////////// Testing V->D conversion //////////
  auto rho = thermo->compute("V->D", {conc});
  EXPECT_EQ(torch::allclose(rho, (conc * thermo->mu).sum(-1),
                            /*rtol=*/1e-4, /*atol=*/1e-4),
            true);

  //////////// Testing VT->cp conversion //////////
  auto cp = thermo->compute("TV->cp", {temp, conc});

  //////////// Testing VT->H conversion //////////
  auto enthalpy = thermo->compute("TV->H", {temp, conc});

  //////////// Testing TPV->S conversion //////////
  auto entropy = thermo->compute("TPV->S", {temp, pres, conc});

  //////////// Testing PVS->T conversion //////////
  thermo->forward(temp, pres, xfrac);
  auto conc2 = thermo->compute("TPX->V", {temp, pres, xfrac});
  auto entropy2 = thermo->compute("TPV->S", {temp, pres, conc2});
  auto temp2 = thermo->compute("PXS->T", {pres, xfrac, entropy2});
  EXPECT_EQ(torch::allclose(temp, temp2, /*rtol=*/1e-4, /*atol=*/1e-4), true);
}

TEST_P(DeviceTest, thermo_xy) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");
  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  int ny = op_thermo.vapor_ids().size() + op_thermo.cloud_ids().size() - 1;
  auto xfrac =
      torch::zeros({1, 2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  auto yfrac = thermo_x->compute("X->Y", {xfrac});
  auto xfrac2 = thermo_y->compute("Y->X", {yfrac});

  EXPECT_EQ(torch::allclose(xfrac, xfrac2, /*rtol=*/1e-4, /*atol=*/1e-4), true);
}

TEST_P(DeviceTest, thermo_yx) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");
  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  int ny = op_thermo.vapor_ids().size() + op_thermo.cloud_ids().size() - 1;
  auto yfrac = torch::zeros({ny, 1, 2, 3}, torch::device(device).dtype(dtype));
  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

  auto xfrac = thermo_y->compute("Y->X", {yfrac});
  auto yfrac2 = thermo_x->compute("X->Y", {xfrac});

  EXPECT_EQ(torch::allclose(yfrac, yfrac2, /*rtol=*/1e-4, /*atol=*/1e-4), true);
}

TEST_P(DeviceTest, eng_pres) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");
  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  int ny = thermo_y->options.vapor_ids().size() +
           thermo_y->options.cloud_ids().size() - 1;
  auto yfrac = torch::zeros({ny, 1}, torch::device(device).dtype(dtype));
  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

  auto temp = 200.0 * torch::ones({1}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({1}, torch::device(device).dtype(dtype));

  auto xfrac = thermo_y->compute("Y->X", {yfrac});
  auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
  auto rho = thermo_x->compute("V->D", {conc});

  auto ivol = thermo_y->compute("DY->V", {rho, yfrac});
  auto intEng = thermo_y->compute("VT->U", {ivol, temp});
  auto pres2 = thermo_y->compute("VT->P", {ivol, temp});
  EXPECT_EQ(torch::allclose(pres, pres2, 1e-4, 1e-4), true);

  auto temp2 = thermo_y->compute("VU->T", {ivol, intEng});
  EXPECT_EQ(torch::allclose(temp, temp2, 1e-4, 1e-4), true);
}

TEST_P(DeviceTest, equilibrate_tp) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml").max_iter(15);

  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  int ny = op_thermo.vapor_ids().size() + op_thermo.cloud_ids().size() - 1;
  auto xfrac =
      torch::zeros({1, 2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  auto temp =
      200.0 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));

  std::cout << "xfrac before = " << xfrac[0][0][0] << std::endl;
  thermo_x->forward(temp, pres, xfrac);
  std::cout << "xfrac after = " << xfrac[0][0][0] << std::endl;

  EXPECT_EQ(torch::allclose(
                xfrac.sum(-1),
                torch::ones({1, 2, 3}, torch::device(device).dtype(dtype)),
                /*rtol=*/1e-4, /*atol=*/1e-4),
            true);
}

TEST_P(DeviceTest, equilibrate_tp_large) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml").max_iter(15);

  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  int ny = op_thermo.vapor_ids().size() + op_thermo.cloud_ids().size() - 1;
  auto xfrac =
      torch::zeros({100, 200, 200, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  auto temp =
      200.0 * torch::ones({100, 200, 200}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({100, 200, 200}, torch::device(device).dtype(dtype));

  std::cout << "xfrac before = " << xfrac[0][0][0] << std::endl;
  thermo_x->forward(temp, pres, xfrac);
  std::cout << "xfrac after = " << xfrac[0][0][0] << std::endl;

  EXPECT_EQ(torch::allclose(
                xfrac.sum(-1),
                torch::ones({100, 200, 200}, torch::device(device).dtype(dtype)),
                /*rtol=*/1e-4, /*atol=*/1e-4),
            true);
}

TEST_P(DeviceTest, equilibrate_uv) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml").max_iter(10);

  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  int ny = thermo_y->options.vapor_ids().size() +
           thermo_y->options.cloud_ids().size() - 1;
  auto yfrac = torch::zeros({ny, 1, 2, 3}, torch::device(device).dtype(dtype));
  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

  auto rho = 0.1 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));

  auto ivol = thermo_y->compute("DY->V", {rho, yfrac});
  auto temp = thermo_y->compute("PV->T", {pres, ivol});
  auto intEng = thermo_y->compute("VT->U", {ivol, temp});

  std::cout << "intEng = " << intEng << std::endl;
  std::cout << "pres before = " << pres << std::endl;
  std::cout << "yfrac before = " << yfrac.index({Slice(), 0, 0, 0})
            << std::endl;
  std::cout << "temp before = " << temp << std::endl;

  thermo_y->forward(rho, intEng, yfrac);

  std::cout << "yfrac after = " << yfrac.index({Slice(), 0, 0, 0}) << std::endl;

  auto ivol2 = thermo_y->compute("DY->V", {rho, yfrac});
  auto temp2 = thermo_y->compute("VU->T", {ivol2, intEng});
  auto pres2 = thermo_y->compute("VT->P", {ivol2, temp2});
  auto intEng2 = thermo_y->compute("VT->U", {ivol2, temp2});

  std::cout << "pres after = " << pres2 << std::endl;
  std::cout << "temp after = " << temp2 << std::endl;
  std::cout << "intEng after = " << intEng2 << std::endl;

  EXPECT_EQ(torch::allclose(intEng, intEng2, 1e-4, 1e-4), true);
}

TEST_P(DeviceTest, equilibrate_uv_large) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml").max_iter(10);

  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  int ny = thermo_y->options.vapor_ids().size() +
           thermo_y->options.cloud_ids().size() - 1;
  auto yfrac = torch::zeros({ny, 100, 200, 200}, torch::device(device).dtype(dtype));
  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

  auto rho = 0.1 * torch::ones({100, 200, 200}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({100, 200, 200}, torch::device(device).dtype(dtype));

  auto ivol = thermo_y->compute("DY->V", {rho, yfrac});
  auto temp = thermo_y->compute("PV->T", {pres, ivol});
  auto intEng = thermo_y->compute("VT->U", {ivol, temp});

  thermo_y->forward(rho, intEng, yfrac);

  auto ivol2 = thermo_y->compute("DY->V", {rho, yfrac});
  auto temp2 = thermo_y->compute("VU->T", {ivol2, intEng});
  auto pres2 = thermo_y->compute("VT->P", {ivol2, temp2});
  auto intEng2 = thermo_y->compute("VT->U", {ivol2, temp2});

  EXPECT_EQ(torch::allclose(intEng, intEng2, 1e-4, 1e-4), true);
}

TEST_P(DeviceTest, extrapolate_ad) {
  if (dtype == torch::kFloat) {
    GTEST_SKIP() << "Skipping float test";
  }

  auto op_thermo =
      ThermoOptions::from_yaml("jupiter.yaml").max_iter(15).ftol(1e-8);

  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  auto temp = 200.0 * torch::ones({2, 3}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({2, 3}, torch::device(device).dtype(dtype));

  int ny = op_thermo.vapor_ids().size() + op_thermo.cloud_ids().size() - 1;
  auto xfrac = torch::zeros({2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  thermo_x->forward(temp, pres, xfrac);
  auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
  auto entropy_vol = thermo_x->compute("TPV->S", {temp, pres, conc});
  auto entropy_mole0 = entropy_vol / conc.sum(-1);

  std::cout << "entropy before = " << entropy_mole0 << std::endl;
  std::cout << "temp before = " << temp << std::endl;

  thermo_x->extrapolate_ad(temp, pres, xfrac, -0.1);

  conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
  entropy_vol = thermo_x->compute("TPV->S", {temp, pres, conc});
  auto entropy_mole1 = entropy_vol / conc.sum(-1);

  std::cout << "entropy after = " << entropy_mole1 << std::endl;
  std::cout << "temp after = " << temp << std::endl;

  EXPECT_EQ(torch::allclose(entropy_mole0, entropy_mole1, 1e-3, 1e-3), true);
}

TEST_P(DeviceTest, relative_humidity) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml").max_iter(15);

  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  int ny = thermo_x->options.vapor_ids().size() +
           thermo_x->options.cloud_ids().size() - 1;

  auto xfrac =
      torch::zeros({1, 2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  auto temp =
      200.0 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));

  thermo_x->forward(temp, pres, xfrac);

  auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
  auto rh = relative_humidity(temp, conc, thermo_x->stoich,
                              thermo_x->options.nucleation());
  std::cout << "rh = " << rh << std::endl;
  EXPECT_LE(rh.min().item<float>(), 1.0);
  EXPECT_GE(rh.max().item<float>(), 0.0);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
