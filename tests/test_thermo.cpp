// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <cmath>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/constants.h>
#include <kintera/math/constrained_newton.h>
#include <kintera/math/lubksb.h>
#include <kintera/math/ludcmp.h>
#include <kintera/vapors/vapor_functions.h>

#include <kintera/thermo/eval_uhs.hpp>
#include <kintera/thermo/relative_humidity.hpp>
#include <kintera/thermo/thermo.hpp>
#include <kintera/thermo/thermo_formatter.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;
using namespace torch::indexing;

TEST(LeastSquaresKkt, ScalesTraceSpeciesConstraints) {
  double matrix[] = {-4.369930087445457e19, -4790.297762242115,
                     -4802.781240192355, -8.048578465493849e35};
  double constraints[] = {0., 0., 1., 0., 0.,  1., -1.,
                          0., 0., 0., 0., -1., 0., 0.};
  double bounds[] = {0.4162445055546258,     2.2883661294100312e-20,
                     1.2424554277345191e-36, 6.965266854905543e-23,
                     1.209489121141968e-19,  4.57170555240497e-38,
                     2.406554145054398e-38};
  for (int repeat = 0; repeat < 2; ++repeat) {
    double rhs[] = {14.340180275068828, -0.08596112159418112};
    int max_iter = 30;
    int status =
        leastsq_kkt(rhs, matrix, constraints, bounds, 2, 2, 7, 0, &max_iter);

    EXPECT_EQ(status, 0);
    EXPECT_LT(max_iter, 30);
    for (int row = 0; row < 7; ++row) {
      double projection =
          constraints[2 * row] * rhs[0] + constraints[2 * row + 1] * rhs[1];
      EXPECT_LE(projection, bounds[row] * (1. + 1.e-10)) << row;
    }
  }
}

TEST(LeastSquaresKkt, ConvergesOnLastAllowedIteration) {
  double matrix[] = {1., 0., 0., 1.};
  double rhs[] = {2., 3.};
  int max_iter = 1;

  int status =
      leastsq_kkt<double>(rhs, matrix, nullptr, nullptr, 2, 2, 0, 0, &max_iter);

  EXPECT_EQ(status, 0);
  EXPECT_EQ(max_iter, 1);
  EXPECT_DOUBLE_EQ(rhs[0], 2.);
  EXPECT_DOUBLE_EQ(rhs[1], 3.);
}

TEST(LeastSquaresKkt, HandlesMoreThan64Constraints) {
  double matrix[] = {1.};
  double rhs[] = {2.};
  double constraints[65] = {};
  double bounds[65];
  for (double& bound : bounds) bound = 1.;
  constraints[64] = 1.;
  int max_iter = 10;

  int status =
      leastsq_kkt(rhs, matrix, constraints, bounds, 1, 1, 65, 0, &max_iter);

  EXPECT_EQ(status, 0);
  EXPECT_LT(max_iter, 10);
  EXPECT_NEAR(rhs[0], 1., 1.e-10);
}

TEST(LeastSquaresKkt, RegularizesRedundantActiveConstraints) {
  double matrix[] = {1., 0., 0., 1.};
  double constraints[] = {1., 0., 1., 0.};
  double bounds[] = {1., 1.};
  for (int repeat = 0; repeat < 2; ++repeat) {
    double rhs[] = {2., 0.};
    int max_iter = 10;
    int status =
        leastsq_kkt(rhs, matrix, constraints, bounds, 2, 2, 2, 2, &max_iter);

    EXPECT_EQ(status, 0);
    EXPECT_NEAR(rhs[0], 1., 1.e-8);
    EXPECT_NEAR(rhs[1], 0., 1.e-8);
  }
}

TEST(LeastSquaresKkt, RejectsInconsistentEqualities) {
  double matrix[] = {1.};
  for (double scale : {1., 1.e-30}) {
    double rhs[] = {0.};
    double constraints[] = {scale, scale};
    double bounds[] = {scale, 2. * scale};
    int max_iter = 10;

    int status =
        leastsq_kkt(rhs, matrix, constraints, bounds, 1, 1, 2, 2, &max_iter);

    EXPECT_EQ(status, 3);
    EXPECT_DOUBLE_EQ(rhs[0], 0.);
  }
}

TEST(LeastSquaresKkt, RejectsContradictoryInequalities) {
  double matrix[] = {1.};
  for (double scale : {1., 1.e-30}) {
    double rhs[] = {0.};
    double constraints[] = {scale, -scale};
    double bounds[] = {scale, -2. * scale};
    int max_iter = 10;

    int status =
        leastsq_kkt(rhs, matrix, constraints, bounds, 1, 1, 2, 0, &max_iter);

    EXPECT_EQ(status, 3);
    EXPECT_DOUBLE_EQ(rhs[0], 0.);
  }
}

TEST(LeastSquaresKkt, RegularizesRankDeficientObjective) {
  double matrix[] = {1., 1., 2., 2.};
  for (int repeat = 0; repeat < 2; ++repeat) {
    double rhs[] = {1., 2.};
    int max_iter = 10;
    int status = leastsq_kkt<double>(rhs, matrix, nullptr, nullptr, 2, 2, 0, 0,
                                     &max_iter);

    EXPECT_EQ(status, 0);
    EXPECT_NEAR(rhs[0], 0.5, 1.e-6);
    EXPECT_NEAR(rhs[1], 0.5, 1.e-6);
  }
}

TEST(LeastSquaresKkt, RejectsInconsistentZeroConstraint) {
  double matrix[] = {1.};
  double rhs[] = {2.};
  double constraint[] = {0.};
  double bound[] = {1.};
  int max_iter = 10;

  int status =
      leastsq_kkt(rhs, matrix, constraint, bound, 1, 1, 1, 1, &max_iter);

  EXPECT_EQ(status, 3);
  EXPECT_EQ(rhs[0], 2.);
}

TEST(LeastSquaresKkt, PreservesEqualityConstraint) {
  double matrix[] = {1., 0., 0., 1.};
  double rhs[] = {2., 0.};
  double constraint[] = {1., 1.};
  double bound[] = {1.};
  int max_iter = 10;

  int status =
      leastsq_kkt(rhs, matrix, constraint, bound, 2, 2, 1, 1, &max_iter);

  EXPECT_EQ(status, 0);
  EXPECT_NEAR(rhs[0], 1.5, 1.e-12);
  EXPECT_NEAR(rhs[1], -0.5, 1.e-12);
}

TEST(LeastSquaresKkt, DirectlySolvesIllConditionedSquareSystem) {
  double matrix[] = {1., 1., 1., 1. + 1.e-10};
  double rhs[] = {0., -1.e-10};
  int max_iter = 10;

  int status = constrained_newton_step(
      rhs, matrix, static_cast<double const*>(nullptr),
      static_cast<double const*>(nullptr), 2, 0, &max_iter);

  EXPECT_EQ(status, 0);
  EXPECT_NEAR(rhs[0], 1., 1.e-6);
  EXPECT_NEAR(rhs[1], -1., 1.e-6);
  EXPECT_EQ(max_iter, 1);
}

TEST(LeastSquaresKkt, LineSearchCanScaleNewtonDirectionToBound) {
  double matrix[] = {1.};
  double constraint[] = {1.};
  double bound[] = {1.};
  double rhs[] = {2.};
  int max_iter = 10;

  int status =
      constrained_newton_step(rhs, matrix, constraint, bound, 1, 1, &max_iter);

  EXPECT_EQ(status, 0);
  EXPECT_DOUBLE_EQ(rhs[0], 2.);
  EXPECT_EQ(max_iter, 1);

  double state[] = {1.};
  double trial[] = {0.};
  EXPECT_FALSE(
      constrained_newton_trial(trial, state, constraint, rhs, 1, 1, 1, 0, 1.));
  EXPECT_TRUE(constrained_newton_trial(trial, state, constraint, rhs, 1, 1, 1,
                                       0, 0.25));
  EXPECT_DOUBLE_EQ(trial[0], 0.5);
}

// #110 contract, stated directly: x <= 1 and x >= 2 must not come back as x = 2
TEST(LeastSquaresKkt, ContradictoryBoundsFailClosed) {
  double matrix[] = {1.};
  double rhs[] = {3.};
  double constraints[] = {1., -1.};
  double bounds[] = {1., -2.};
  int max_iter = 10;

  int status =
      leastsq_kkt(rhs, matrix, constraints, bounds, 1, 1, 2, 0, &max_iter);

  EXPECT_NE(status, 0);
  EXPECT_NE(rhs[0], 2.);
  EXPECT_DOUBLE_EQ(rhs[0], 3.);
}

// Two extents sharing one reactant (NH3 -> NH3(s), NH3 + H2S -> NH4SH(s)):
// four rows in 2-D, three of them violated by the unconstrained step.
TEST(LeastSquaresKktFeasibleOrigin, SolvesSharedReactantStep) {
  double matrix[] = {1., 0., 0., 1.};
  // NH3, H2S, NH3(s), NH4SH(s): consumed stock <= concentration
  double constraints[] = {1., 1., 0., 1., -1., 0., 0., -1.};
  double bounds[] = {1., 0.5, 0., 0.};
  for (int repeat = 0; repeat < 2; ++repeat) {
    double rhs[] = {-1., 5.};
    int max_iter = 5;
    int status = leastsq_kkt_feasible_origin(rhs, matrix, constraints, bounds,
                                             2, 2, 4, 0, &max_iter);

    EXPECT_EQ(status, 0);
    EXPECT_NEAR(rhs[0], 0., 1.e-12);
    EXPECT_NEAR(rhs[1], 0.5, 1.e-12);
  }
}

// A row within sqrt(eps) of the active block is not activated; one well
// clear of it is, whatever the magnitude of C and d. Once x0 <= 0.5 is
// active, x0 + delta x1 <= 0.5 + delta / 2 is violated by 0.75 delta.
TEST(LeastSquaresKktFeasibleOrigin, SkipsNearlyDependentRow) {
  double matrix[] = {1., 1., 0., 1.};
  for (double scale : {1., 1.e-30, 1.e30}) {
    for (double delta : {1.e-3, 1.e-12}) {
      double constraints[] = {scale, 0., scale, scale * delta};
      double bounds[] = {scale * 0.5, scale * (0.5 + 0.5 * delta)};
      double rhs[] = {3., 0.};
      int max_iter = 5;
      int status = leastsq_kkt_feasible_origin(rhs, matrix, constraints, bounds,
                                               2, 2, 2, 0, &max_iter);

      EXPECT_EQ(status, 0) << delta << " " << scale;
      if (delta > 1.e-6) {
        // activated, then x0 <= 0.5 drops out (negative multiplier): the
        // exact optimum has only the second row active
        EXPECT_EQ(max_iter, 4) << scale;
        double c = 0.5 + 0.5 * delta;
        double x1 =
            (1. - delta) * (3. - c) / ((1. - delta) * (1. - delta) + 1.);
        EXPECT_NEAR(rhs[1], x1, 1.e-12) << scale;
        EXPECT_NEAR(rhs[0], c - delta * x1, 1.e-12) << scale;
      } else {
        // never activated: x0 <= 0.5 alone, the other row off by 0.75 delta
        EXPECT_EQ(max_iter, 2) << scale;
        EXPECT_NEAR(rhs[0], 0.5, 1.e-12) << scale;
        EXPECT_NEAR(rhs[1], 1.25, 1.e-12) << scale;
      }
    }
  }
}

TEST(LeastSquaresKktFeasibleOrigin, RejectsInfeasibleOrigin) {
  double matrix[] = {1.};
  double constraints[] = {1., -1.};
  for (double lower : {2., 1.e-6}) {
    double rhs[] = {3.};
    double bounds[] = {1., -lower};  // x <= 1 and x >= lower
    int max_iter = 10;
    int status = leastsq_kkt_feasible_origin(rhs, matrix, constraints, bounds,
                                             1, 1, 2, 0, &max_iter);

    EXPECT_EQ(status, 1) << lower;
    EXPECT_DOUBLE_EQ(rhs[0], 3.) << lower;
  }

  // an equality x = 1 excludes x = 0 as well
  double rhs[] = {3.};
  double equality[] = {1.};
  double target[] = {1.};
  int max_iter = 10;
  EXPECT_EQ(leastsq_kkt_feasible_origin(rhs, matrix, equality, target, 1, 1, 1,
                                        1, &max_iter),
            1);
  EXPECT_DOUBLE_EQ(rhs[0], 3.);
}

TEST(LeastSquaresKktFeasibleOrigin, AcceptsRoundOffNegativeBound) {
  double matrix[] = {1., 0., 0., 1.};
  // x0 >= 1e-19: a used-up stock of 1e-3 after round-off; x1 <= 1e-3
  double constraints[] = {-1., 0., 0., 1.};
  double bounds[] = {-1.e-19, 1.e-3};
  double rhs[] = {-3., 3.};
  int max_iter = 10;
  int status = leastsq_kkt_feasible_origin(rhs, matrix, constraints, bounds, 2,
                                           2, 2, 0, &max_iter);

  EXPECT_EQ(status, 0);
  EXPECT_NEAR(rhs[0], 0., 1.e-18);
  EXPECT_NEAR(rhs[1], 1.e-3, 1.e-18);
}

TEST_P(DeviceTest, thermo_y) {
  auto op_thermo = ThermoOptionsImpl::from_yaml("jupiter.yaml");

  std::cout << fmt::format("{}", op_thermo) << std::endl;

  ThermoY thermo(op_thermo);
  thermo->to(device, dtype);

  int ny = thermo->options->vapor_ids().size() +
           thermo->options->cloud_ids().size() - 1;
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
  auto op_thermo = ThermoOptionsImpl::from_yaml("jupiter.yaml");
  op_thermo->max_iter(10);

  ThermoX thermo(op_thermo);
  thermo->to(device, dtype);

  int ny = thermo->options->vapor_ids().size() +
           thermo->options->cloud_ids().size() - 1;
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
  auto op_thermo = ThermoOptionsImpl::from_yaml("jupiter.yaml");
  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  int ny = op_thermo->vapor_ids().size() + op_thermo->cloud_ids().size() - 1;
  auto xfrac =
      torch::zeros({1, 2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  auto yfrac = thermo_x->compute("X->Y", {xfrac});
  auto xfrac2 = thermo_y->compute("Y->X", {yfrac});

  EXPECT_EQ(torch::allclose(xfrac, xfrac2, /*rtol=*/1e-4, /*atol=*/1e-4), true);
}

TEST_P(DeviceTest, thermo_yx) {
  auto op_thermo = ThermoOptionsImpl::from_yaml("jupiter.yaml");
  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  int ny = op_thermo->vapor_ids().size() + op_thermo->cloud_ids().size() - 1;
  auto yfrac = torch::zeros({ny, 1, 2, 3}, torch::device(device).dtype(dtype));
  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

  auto xfrac = thermo_y->compute("Y->X", {yfrac});
  auto yfrac2 = thermo_x->compute("X->Y", {xfrac});

  EXPECT_EQ(torch::allclose(yfrac, yfrac2, /*rtol=*/1e-4, /*atol=*/1e-4), true);
}

TEST_P(DeviceTest, eng_pres) {
  auto op_thermo = ThermoOptionsImpl::from_yaml("jupiter.yaml");
  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  int ny = thermo_y->options->vapor_ids().size() +
           thermo_y->options->cloud_ids().size() - 1;
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
  if (device.type() == torch::kMPS) {
    GTEST_SKIP() << "equilibrate_tp has no MPS backend.";
  }

  auto op_thermo = ThermoOptionsImpl::from_yaml("jupiter.yaml");
  op_thermo->max_iter(15);

  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  int ny = op_thermo->vapor_ids().size() + op_thermo->cloud_ids().size() - 1;
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
  if (device.type() == torch::kMPS) {
    GTEST_SKIP() << "equilibrate_tp has no MPS backend.";
  }

  auto op_thermo = ThermoOptionsImpl::from_yaml("earth.yaml");
  op_thermo->max_iter(15);

  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  int ny = op_thermo->vapor_ids().size() + op_thermo->cloud_ids().size() - 1;
  auto xfrac =
      torch::zeros({100, 200, 200, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  auto temp =
      200.0 * torch::ones({100, 200, 200}, torch::device(device).dtype(dtype));
  auto pres =
      1.e5 * torch::ones({100, 200, 200}, torch::device(device).dtype(dtype));

  std::cout << "xfrac before = " << xfrac[0][0][0] << std::endl;
  thermo_x->forward(temp, pres, xfrac);
  std::cout << "xfrac after = " << xfrac[0][0][0] << std::endl;

  EXPECT_EQ(torch::allclose(xfrac.sum(-1),
                            torch::ones({100, 200, 200},
                                        torch::device(device).dtype(dtype)),
                            /*rtol=*/1e-4, /*atol=*/1e-4),
            true);
}

TEST_P(DeviceTest, equilibrate_uv) {
  if (device.type() == torch::kMPS) {
    GTEST_SKIP() << "equilibrate_uv has no MPS backend.";
  }

  auto op_thermo = ThermoOptionsImpl::from_yaml("jupiter.yaml");
  op_thermo->max_iter(10);

  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  int ny = thermo_y->options->vapor_ids().size() +
           thermo_y->options->cloud_ids().size() - 1;
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

TEST_P(DeviceTest, equilibrate_uv_depleted_cloud) {
  if (dtype != torch::kFloat64 || device.type() == torch::kMPS) {
    GTEST_SKIP();
  }

  auto config = YAML::Load(R"(
reference-state: {Tref: 0.0, Pref: 1.e5}
species:
  - {name: dry, composition: {H: 1.5, He: 0.15}, cv_R: 2.5}
  - {name: CH4, composition: {C: 1, H: 4}, cv_R: 3.5, u0_R: 0.0}
  - {name: H2S, composition: {H: 2, S: 1}, cv_R: 3.5, u0_R: 0.0}
  - {name: CH4(s), composition: {C: 1, H: 4}, cv_R: 4.5, u0_R: -980.0}
  - {name: 'CH4(s,p)', composition: {C: 1, H: 4}, cv_R: 4.5, u0_R: -980.0}
  - {name: H2S(s), composition: {H: 2, S: 1}, cv_R: 4.5, u0_R: -2250.0}
  - {name: 'H2S(s,p)', composition: {H: 2, S: 1}, cv_R: 4.5, u0_R: -2250.0}
reactions:
  - {equation: 'CH4 => CH4(s)', type: nucleation, rate-constant: {formula: ch4_ideal}}
  - {equation: 'CH4(s) => CH4(s,p)', type: coagulation, rate-constant: {A: 0.0001, b: 0.0, Ea_R: 0.0}}
  - {equation: 'CH4(s,p) => CH4', type: evaporation, rate-constant: {formula: ch4_ideal, diff_c: 2.e-5, diff_T: 0.0, diff_P: 0.0, vm: 1.6e-5, diameter: 0.001}}
  - {equation: 'H2S => H2S(s)', type: nucleation, rate-constant: {formula: h2s_ideal}}
  - {equation: 'H2S(s) => H2S(s,p)', type: coagulation, rate-constant: {A: 0.0001, b: 0.0, Ea_R: 0.0}}
  - {equation: 'H2S(s,p) => H2S', type: evaporation, rate-constant: {formula: h2s_ideal, diff_c: 2.e-5, diff_T: 0.0, diff_P: 0.0, vm: 3.4e-5, diameter: 0.001}}
dynamics:
  equation-of-state: {max-iter: 30, ftol: 1.e-6}
)");
  init_species_from_yaml(config);
  auto op_thermo = ThermoOptionsImpl::from_yaml(config);
  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  auto tensor_options = torch::device(device).dtype(dtype);
  auto rho = torch::tensor({0.6120118390199387}, tensor_options);
  auto intEng = torch::tensor({434819.1169266227}, tensor_options);
  auto yfrac = torch::tensor({{0.1250846769992131},
                              {1.5203920607688577e-5},
                              {0.0006193545911286463},
                              {0.000688122185010699},
                              {1.884825971316318e-7},
                              {1.5557173919300405e-7}},
                             tensor_options);
  auto initial = yfrac.clone();
  auto diag = torch::zeros({1, 1}, tensor_options);

  thermo_y->forward(rho, intEng, yfrac, false, diag);

  EXPECT_LE(diag.item<double>(), op_thermo->max_iter());
  EXPECT_TRUE(torch::all(yfrac >= 0.).item<bool>());
  EXPECT_DOUBLE_EQ(yfrac[2][0].item<double>(), 0.);
  EXPECT_NEAR(
      (yfrac[0] + yfrac[2] + yfrac[3] - initial[0] - initial[2] - initial[3])
          .item<double>(),
      0., 1.e-12);
  EXPECT_NEAR(
      (yfrac[1] + yfrac[4] + yfrac[5] - initial[1] - initial[4] - initial[5])
          .item<double>(),
      0., 1.e-12);
  auto ivol = thermo_y->compute("DY->V", {rho, yfrac});
  auto temp = thermo_y->compute("VU->T", {ivol, intEng});
  auto intEng_after = thermo_y->compute("VT->U", {ivol, temp});
  EXPECT_TRUE(torch::allclose(intEng, intEng_after, 1.e-12, 1.e-8));

  op_thermo->max_iter(1);
  auto final_diag = torch::zeros({1, 1}, tensor_options);
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  auto final_gain = thermo_y->forward(rho, intEng, yfrac, false, final_diag);
  auto output = testing::internal::GetCapturedStdout() +
                testing::internal::GetCapturedStderr();
  EXPECT_EQ(output.find("equilibrate_uv did not converge"), std::string::npos);
  EXPECT_DOUBLE_EQ(final_diag.item<double>(), 1.);
  EXPECT_TRUE(torch::all(final_gain == 0.).item<bool>());

  op_thermo->max_iter(30);
  auto trace_conc = torch::tensor(
      {325.96348426613355, 8.1247303078597515, std::exp(-6.4175548244587679),
       1.e-20, 6.7784957553955697e-5, 0., 4.1234878195558463e-6},
      tensor_options);
  auto trace_mass = trace_conc * torch::tensor(species_weights, tensor_options);
  auto trace_rho = trace_mass.sum().reshape({1});
  auto trace_yfrac = (trace_mass.slice(0, 1) / trace_rho).reshape({6, 1});
  auto trace_ivol = thermo_y->compute("DY->V", {trace_rho, trace_yfrac});
  auto trace_temp = torch::tensor({86.95274487484208}, tensor_options);
  auto trace_intEng = thermo_y->compute("VT->U", {trace_ivol, trace_temp});
  auto trace_diag = torch::zeros({1, 1}, tensor_options);

  thermo_y->forward(trace_rho, trace_intEng, trace_yfrac, false, trace_diag);

  EXPECT_DOUBLE_EQ(trace_diag.item<double>(), 1.);
  EXPECT_DOUBLE_EQ(trace_yfrac[2][0].item<double>(), 0.);

  op_thermo->uv_solver("partition");
  EXPECT_TRUE(thermo_y->uv_partitionable);
  auto cold_conc =
      torch::tensor({0.2493891969, 6.330521766e-26, 0., 1.9957527307e-16,
                     1.224047439e-16, 7.291174753e-34, 4.651872658e-34},
                    tensor_options);
  auto cold_mass = cold_conc * torch::tensor(species_weights, tensor_options);
  auto cold_rho = cold_mass.sum().reshape({1});
  auto cold_yfrac = (cold_mass.slice(0, 1) / cold_rho).reshape({6, 1});
  auto cold_initial = cold_yfrac.clone();
  auto cold_ivol = thermo_y->compute("DY->V", {cold_rho, cold_yfrac});
  auto cold_temp = torch::tensor({12.597949986163009}, tensor_options);
  auto cold_intEng = thermo_y->compute("VT->U", {cold_ivol, cold_temp});
  auto cold_diag = torch::zeros({1, 1}, tensor_options);

  thermo_y->forward(cold_rho, cold_intEng, cold_yfrac, false, cold_diag);

  EXPECT_LT(cold_diag.item<double>(), op_thermo->max_iter());
  EXPECT_GT(cold_yfrac[1][0].item<double>(), 0.);
  EXPECT_TRUE(torch::all(cold_yfrac >= 0.).item<bool>());
  auto cold_ivol_after = thermo_y->compute("DY->V", {cold_rho, cold_yfrac});
  auto cold_conc_after = (cold_ivol_after * thermo_y->inv_mu).flatten();
  auto cold_temp_after =
      thermo_y->compute("VU->T", {cold_ivol_after, cold_intEng});
  auto cold_energy_after =
      thermo_y->compute("VT->U", {cold_ivol_after, cold_temp_after});
  EXPECT_TRUE(torch::allclose(cold_intEng, cold_energy_after, 1.e-12, 1.e-8));
  EXPECT_NEAR(cold_temp_after.item<double>(), cold_temp.item<double>(), 1.e-6);
  EXPECT_DOUBLE_EQ((cold_conc_after[2] + cold_conc_after[5]).item<double>(),
                   (cold_conc[2] + cold_conc[5]).item<double>());
  double adjusted_temperature = cold_temp_after.item<double>();
  double log_saturation = h2s_ideal(adjusted_temperature) -
                          std::log(constants::Rgas * adjusted_temperature);
  EXPECT_NEAR(std::log(cold_conc_after[2].item<double>()), log_saturation,
              1.e-6);

  op_thermo->max_iter(1);
  auto failed_yfrac = initial.clone();
  auto failed_diag = torch::full({1, 1}, 7., tensor_options);
  thermo_y->nactive.fill_(2);
  auto failed_gain =
      thermo_y->forward(rho, intEng, failed_yfrac, true, failed_diag);
  EXPECT_TRUE(torch::all(failed_gain == 0.).item<bool>());
  EXPECT_DOUBLE_EQ(failed_diag.item<double>(), -1.);
  EXPECT_EQ(thermo_y->nactive.item<int>(), 0);
  EXPECT_TRUE(torch::allclose(failed_yfrac, initial, 1.e-12, 0.));
  op_thermo->max_iter(30);

  op_thermo->uv_solver("auto");
  auto cold_auto = cold_initial.clone();
  auto auto_diag = torch::zeros({1, 1}, tensor_options);
  thermo_y->forward(cold_rho, cold_intEng, cold_auto, false, auto_diag);
  EXPECT_LT(auto_diag.item<double>(), op_thermo->max_iter());
  EXPECT_TRUE(torch::allclose(cold_auto, cold_yfrac, 1.e-12, 0.));

  op_thermo->uv_solver("kkt");
  auto kkt_diag = torch::zeros({1, 1}, tensor_options);
  thermo_y->forward(trace_rho, trace_intEng, trace_yfrac, false, kkt_diag);
  EXPECT_LT(kkt_diag.item<double>(), op_thermo->max_iter());

  init_species_from_yaml("jupiter.yaml");
}

TEST_P(DeviceTest, equilibrate_uv_partition_rejects_coupled_reactions) {
  if (device.type() == torch::kMPS) {
    GTEST_SKIP() << "equilibrate_uv has no MPS backend.";
  }

  auto options = ThermoOptionsImpl::from_yaml("jupiter.yaml");
  options->uv_solver("partition");
  ThermoY thermo_y(options);
  EXPECT_FALSE(thermo_y->uv_partitionable);
  auto density = torch::ones({1}, torch::device(device).dtype(dtype));
  auto energy = torch::ones({1}, torch::device(device).dtype(dtype));
  auto fractions =
      torch::zeros({static_cast<int>(options->species().size()) - 1, 1},
                   torch::device(device).dtype(dtype));
  EXPECT_THROW(thermo_y->forward(density, energy, fractions), c10::Error);
}

TEST_P(DeviceTest, equilibrate_uv_large) {
  if (device.type() == torch::kMPS) {
    GTEST_SKIP() << "equilibrate_uv has no MPS backend.";
  }

  auto op_thermo = ThermoOptionsImpl::from_yaml("jupiter.yaml");
  op_thermo->max_iter(10);

  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  int ny = thermo_y->options->vapor_ids().size() +
           thermo_y->options->cloud_ids().size() - 1;
  auto yfrac =
      torch::zeros({ny, 100, 200, 200}, torch::device(device).dtype(dtype));
  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

  auto rho =
      0.1 * torch::ones({100, 200, 200}, torch::device(device).dtype(dtype));
  auto pres =
      1.e5 * torch::ones({100, 200, 200}, torch::device(device).dtype(dtype));

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

  auto op_thermo = ThermoOptionsImpl::from_yaml("jupiter.yaml");
  (*op_thermo).max_iter(15).ftol(1e-8);

  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  auto temp = 200.0 * torch::ones({2, 3}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({2, 3}, torch::device(device).dtype(dtype));

  int ny = op_thermo->vapor_ids().size() + op_thermo->cloud_ids().size() - 1;
  auto xfrac = torch::zeros({2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  thermo_x->forward(temp, pres, xfrac);
  auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
  auto entropy_vol = thermo_x->compute("TPV->S", {temp, pres, conc});
  auto entropy_mole0 = entropy_vol / conc.sum(-1);

  std::cout << "entropy before = " << entropy_mole0 << std::endl;
  std::cout << "temp before = " << temp << std::endl;

  thermo_x->extrapolate_dlnp(temp, pres, xfrac, ExtrapOptions().dlnp(-0.1));

  conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
  entropy_vol = thermo_x->compute("TPV->S", {temp, pres, conc});
  auto entropy_mole1 = entropy_vol / conc.sum(-1);

  std::cout << "entropy after = " << entropy_mole1 << std::endl;
  std::cout << "temp after = " << temp << std::endl;

  EXPECT_EQ(torch::allclose(entropy_mole0, entropy_mole1, 1e-3, 1e-3), true);
}

TEST_P(DeviceTest, relative_humidity) {
  auto op_thermo = ThermoOptionsImpl::from_yaml("jupiter.yaml");
  op_thermo->max_iter(15);

  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  int ny = thermo_x->options->vapor_ids().size() +
           thermo_x->options->cloud_ids().size() - 1;

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
                              thermo_x->options->nucleation());
  std::cout << "rh = " << rh << std::endl;
  EXPECT_LE(rh.min().item<float>(), 1.0);
  EXPECT_GE(rh.max().item<float>(), 0.0);
}

void test_ludcmp_skip() {
  double array[9] = {3, 0, 0, 0, 0, 0, 0, 0, 1};
  double rhs[3] = {1, 2, 3};
  int indx[3];
  int skip_row[3] = {0, 1, 0};

  ludcmp(array, indx, 3, skip_row);
  lubksb(rhs, array, indx, 3, skip_row);

  printf("rhs = \n");
  for (int i = 0; i < 3; ++i) {
    printf("%f\n", rhs[i]);
  }
}

int main(int argc, char** argv) {
  // torch::set_num_threads(1);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();

  // test_ludcmp_skip();
}
