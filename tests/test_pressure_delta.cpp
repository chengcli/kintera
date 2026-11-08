#include <gtest/gtest.h>
#include <torch/torch.h>

#include <kintera/thermo/eval_uhs.hpp>

TEST(PressureDelta, RetainsSubUlpHeating) {
  auto e0 = torch::full({3}, 2.5e7, torch::kFloat32);
  auto gas0 = torch::full_like(e0, 1.e4);
  auto cv0 = torch::full_like(e0, 2.5e4);
  auto de = torch::tensor({0.01f, -0.01f, 0.f});
  auto zero = torch::zeros_like(de);
  ASSERT_TRUE(torch::equal(e0 + de, e0));
  auto dp =
      kintera::eval_const_cv_pressure_delta(e0, gas0, cv0, de, zero, zero);
  EXPECT_EQ(dp.scalar_type(), torch::kFloat32);
  EXPECT_TRUE(torch::allclose(dp, de * 0.4, 2.e-7, 1.e-10));
}

TEST(PressureDelta, CompositionAndEnergyMatchDoubleTotals) {
  auto e0 = torch::full({1}, 2.5e7, torch::kFloat64);
  auto gas0 = torch::full_like(e0, 1.e4);
  auto cv0 = torch::full_like(e0, 2.5e4);
  auto de = torch::tensor({-2.e4, 0., 5.e4}, torch::kFloat64);
  auto dg = torch::tensor({10., -3., 0.}, torch::kFloat64);
  auto dc = torch::tensor({-5., 20., 7.}, torch::kFloat64);
  auto expected = (e0 + de) * (gas0 + dg) / (cv0 + dc) - e0 * gas0 / cv0;
  auto dp = kintera::eval_const_cv_pressure_delta(e0, gas0, cv0, de, dg, dc);
  EXPECT_TRUE(torch::allclose(dp, expected, 1.e-11, 1.e-8));
  auto dp32 = kintera::eval_const_cv_pressure_delta(
      e0.to(torch::kFloat32), gas0.to(torch::kFloat32), cv0.to(torch::kFloat32),
      de.to(torch::kFloat32), dg.to(torch::kFloat32), dc.to(torch::kFloat32));
  EXPECT_TRUE(
      torch::allclose(dp32.to(torch::kFloat64), expected, 2.e-6, 1.e-5));
}
