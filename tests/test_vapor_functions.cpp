// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <string>
#include <vector>

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

struct CondensateFit {
  const char *name;
  double (*value)(double);
  double (*derivative)(double);
};

const CondensateFit condensate_fits[] = {
    {"h2o_ideal", h2o_ideal, h2o_ideal_ddT},
    {"h2o_bryan", h2o_bryan, h2o_bryan_ddT},
    {"nh3_ideal", nh3_ideal, nh3_ideal_ddT},
    {"nh3_h2s_lewis", nh3_h2s_lewis, nh3_h2s_lewis_ddT},
    {"h2s_ideal", h2s_ideal, h2s_ideal_ddT},
    {"h2s_antoine", h2s_antoine, h2s_antoine_ddT},
    {"ch4_ideal", ch4_ideal, ch4_ideal_ddT},
    {"so2_antoine", so2_antoine, so2_antoine_ddT},
    {"co2_antoine", co2_antoine, co2_antoine_ddT},
    {"kcl_lodders", kcl_lodders, kcl_lodders_ddT},
    {"na_h2s_visscher", na_h2s_visscher, na_h2s_visscher_ddT}};

void check_condensate_dispatch(torch::Device device) {
  std::vector<std::string> names;
  for (const auto &fit : condensate_fits) names.emplace_back(fit.name);
  auto nucleation = NucleationOptionsImpl::create();
  nucleation->logsvp(names);
  LogSVPFunc::init(nucleation);

  auto temperatures = torch::tensor({500.0, 1000.0, 1500.0}, torch::kFloat64);
  auto temp = temperatures.to(device);
  auto values = LogSVPFunc::call(temp).cpu();
  auto derivatives = LogSVPFunc::grad(temp).cpu();
  for (int i = 0; i < temperatures.numel(); ++i) {
    double t = temperatures[i].item<double>();
    for (size_t j = 0; j < names.size(); ++j) {
      const auto &fit = condensate_fits[j];
      SCOPED_TRACE(fit.name);
      EXPECT_NEAR(values[i][j].item<double>(), fit.value(t), 1.e-10);
      EXPECT_NEAR(derivatives[i][j].item<double>(), fit.derivative(t), 1.e-10);
    }
  }
}

}  // namespace

TEST(VaporFunctions, condensate_derivatives_match_finite_differences) {
  constexpr double step = 1.e-3;
  for (const auto &fit : condensate_fits) {
    SCOPED_TRACE(fit.name);
    for (double t : {150.0, 180.0, 250.0, 500.0, 800.0, 1000.0, 1500.0}) {
      double expected =
          (fit.value(t + step) - fit.value(t - step)) / (2. * step);
      EXPECT_NEAR(fit.derivative(t), expected, 1.e-9);
    }
  }
}

TEST(VaporFunctions, condensates_dispatch_on_cpu) {
  check_condensate_dispatch(torch::kCPU);
}

TEST(VaporFunctions, condensates_dispatch_on_cuda) {
#ifdef ENABLE_CUDA
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA is not available";
  check_condensate_dispatch(torch::kCUDA);
#else
  GTEST_SKIP() << "Kintera was built without CUDA";
#endif
}

TEST(VaporFunctions, kcl_lodders_converts_bar_fit_to_pascals) {
  for (double temp : {150.0, 180.0, 250.0, 500.0, 800.0, 1000.0, 1500.0}) {
    double expected_pa = 1.e5 * std::pow(10., 7.611 - 11382. / temp);
    EXPECT_NEAR(std::exp(kcl_lodders(temp)) / expected_pa, 1., 1.e-12);
  }
}

TEST(VaporFunctions, na2s_quotient_converts_bar_squared_to_pascals) {
  for (double temp : {800.0, 1100.0, 1400.0}) {
    // PR108: log10 Q(bar) = 12.48 - 27778/T. The signed gas
    // exponents sum to two, so converting to Pa multiplies Q by 1e10.
    double q_bar = std::pow(10., 12.48 - 27778. / temp);
    EXPECT_NEAR(std::exp(na_h2s_visscher(temp)) / (1.e10 * q_bar), 1., 1.e-12);
    EXPECT_NEAR(na_h2s_visscher_ddT(temp),
                27778. * std::log(10.) / (temp * temp), 1.e-12);
  }
}

TEST(VaporFunctions, kcl_lodders_derivative_matches_finite_difference) {
  constexpr double step = 1.e-3;
  for (double temp : {150.0, 180.0, 250.0, 500.0, 800.0, 1000.0, 1500.0}) {
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

// NIST WebBook, Antoine parameters: H2S (Stull 1947), CO2 (Giauque 1937).
TEST(VaporFunctions, nist_antoine_coefficients) {
  EXPECT_NEAR(
      h2s_antoine(180.),
      std::log(1.e5) + (4.43681 - 829.439 / (180. - 25.412)) * std::log(10.),
      1.e-12);
  EXPECT_NEAR(
      h2s_antoine(250.),
      std::log(1.e5) + (4.52887 - 958.587 / (250. - .539)) * std::log(10.),
      1.e-12);
  EXPECT_NEAR(
      co2_antoine(180.),
      std::log(1.e5) + (6.81228 - 1301.679 / (180. - 3.494)) * std::log(10.),
      1.e-12);
  for (double t : {180., 250.}) {
    EXPECT_NEAR(h2s_antoine_ddT(t),
                (h2s_antoine(t + .001) - h2s_antoine(t - .001)) / .002, 1.e-9);
  }
  EXPECT_NEAR(co2_antoine_ddT(180.),
              (co2_antoine(180.001) - co2_antoine(179.999)) / .002, 1.e-9);
}

TEST(VaporFunctions, ideal_branches_meet_at_reference_pressure) {
  EXPECT_NEAR(h2o_ideal(273.16), std::log(611.7), 1.e-12);
  EXPECT_NEAR(nh3_ideal(195.4), std::log(6060.), 1.e-12);
  EXPECT_NEAR(h2s_ideal(187.63), std::log(23300.), 1.e-12);
  EXPECT_NEAR(ch4_ideal(90.67), std::log(11690.), 1.e-12);
}
