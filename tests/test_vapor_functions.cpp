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
