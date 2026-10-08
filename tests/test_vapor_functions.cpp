// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/vapors/vapor_functions.h>

#include <kintera/thermo/log_svp.hpp>

using namespace kintera;

namespace {

double h2o_bryan_expected(double T) {
  double beta = 24.845;
  double delta = 4.986009;
  double tr = 273.16;
  double pr = 611.7;
  return (1. - tr / T) * beta - delta * std::log(T / tr) + std::log(pr);
}

double h2o_bryan_ddT_expected(double T) {
  double beta = 24.845;
  double delta = 4.986009;
  double tr = 273.16;
  double t = T / tr;
  return (beta / (t * t) - delta / t) / tr;
}

}  // namespace

TEST(VaporFunctions, kcl_lodders_converts_bar_fit_to_pascals) {
  for (double temp : {500.0, 800.0, 1000.0, 1500.0}) {
    double expected_pa = 1.e5 * std::pow(10., 7.611 - 11382. / temp);
    EXPECT_NEAR(std::exp(kcl_lodders(temp)) / expected_pa, 1., 1.e-12);
  }
}

TEST(VaporFunctions, kcl_lodders_derivative_matches_finite_difference) {
  constexpr double step = 1.e-3;
  for (double temp : {500.0, 800.0, 1000.0, 1500.0}) {
    double finite_difference =
        (kcl_lodders(temp + step) - kcl_lodders(temp - step)) / (2. * step);
    EXPECT_NEAR(kcl_lodders_ddT(temp), finite_difference, 1.e-10);
  }
}

TEST(VaporFunctions, kcl_lodders_dispatches_through_log_svp) {
  auto nucleation = NucleationOptionsImpl::create();
  nucleation->logsvp({"kcl_lodders"});
  LogSVPFunc::init(nucleation);

  auto temp = torch::tensor({800.0, 1000.0}, torch::kFloat64);
  auto logsvp = LogSVPFunc::call(temp).squeeze(-1);
  auto grad = LogSVPFunc::grad(temp).squeeze(-1);

  for (int i = 0; i < 2; ++i) {
    double t = temp[i].item<double>();
    double expected_pa = 1.e5 * std::pow(10., 7.611 - 11382. / t);
    EXPECT_NEAR(logsvp[i].item<double>(), std::log(expected_pa), 1.e-12);
    EXPECT_NEAR(grad[i].item<double>(), 11382. * std::log(10.) / (t * t),
                1.e-12);
  }
}

TEST(VaporFunctions, h2o_bryan_matches_athena_liquid_branch) {
  for (double temp : {250.0, 273.16, 289.85, 300.0}) {
    EXPECT_NEAR(h2o_bryan(temp), h2o_bryan_expected(temp), 1.e-12);
    EXPECT_NEAR(h2o_bryan_ddT(temp), h2o_bryan_ddT_expected(temp), 1.e-12);
  }
}

TEST(VaporFunctions, h2o_bryan_keeps_liquid_branch_below_triple_point) {
  double temp = 250.0;
  EXPECT_NEAR(h2o_bryan(temp), h2o_bryan_expected(temp), 1.e-12);
  EXPECT_GT(std::abs(h2o_bryan(temp) - h2o_ideal(temp)), 1.e-3);
}

TEST(VaporFunctions, h2o_bryan_dispatches_through_log_svp) {
  auto nucleation = NucleationOptionsImpl::create();
  nucleation->logsvp({"h2o_bryan"});
  LogSVPFunc::init(nucleation);

  auto temp = torch::tensor({250.0, 289.85}, torch::kFloat64);
  auto logsvp = LogSVPFunc::call(temp).squeeze(-1);
  auto grad = LogSVPFunc::grad(temp).squeeze(-1);

  EXPECT_NEAR(logsvp[0].item<double>(), h2o_bryan_expected(250.0), 1.e-12);
  EXPECT_NEAR(logsvp[1].item<double>(), h2o_bryan_expected(289.85), 1.e-12);
  EXPECT_NEAR(grad[0].item<double>(), h2o_bryan_ddT_expected(250.0), 1.e-12);
  EXPECT_NEAR(grad[1].item<double>(), h2o_bryan_ddT_expected(289.85), 1.e-12);
}
