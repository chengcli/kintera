#include <gtest/gtest.h>
#include <kintera/thermo/equilibrate_tp.h>

#include <cmath>
#include <cstring>

namespace {

// One dry gas, one vapor, one condensate. log(svp) is chosen so the
// equilibrium vapor fraction of the gas is 0.2. The initial state is far
// from that, and a wide cap converges on a known iteration.
double log_psat(double) { return std::log(0.2 * 1.e5); }

struct Call {
  int rc;
  int iter;
  double diag;
  double x[3];
};

Call solve(int cap) {
  constexpr int ns = 3;
  double stoich[ns] = {0., -1., 1.};
  double params[6] = {};
  int kind = 0;
  kintera::user_func1 func = log_psat;
  double x[ns] = {0.5, 0.5, 0.1};
  double gain[1] = {};
  double diag[1] = {};
  int reaction_set[1] = {0};
  int nactive = 0;
  int max_iter = cap;
  int rc = kintera::equilibrate_tp<double>(
      gain, diag, x, 300., 1.e5, stoich, ns, 1, 2, &func, &kind, params, 1.e-6f,
      &max_iter, reaction_set, &nactive, nullptr);
  Call out;
  out.rc = rc;
  out.diag = diag[0];
  out.iter = max_iter;
  std::memcpy(out.x, x, sizeof x);
  return out;
}

}  // namespace

// A solve that first succeeds with the cap above the iteration it needs must
// also succeed when the cap is exactly that iteration. Meeting the tolerance
// on that pass is success, even though iter has reached the cap.
TEST(equilibrate_tp, converges_on_its_last_allowed_iteration) {
  auto wide = solve(20);
  ASSERT_EQ(wide.rc, 0);
  ASSERT_GE(wide.iter, 2);

  auto exact = solve(wide.iter);
  EXPECT_EQ(exact.rc, 0);
  EXPECT_EQ(exact.iter, wide.iter);
  for (int i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(exact.x[i], wide.x[i]);
}

// One iteration does not reach that equilibrium, so the failure code stays.
TEST(equilibrate_tp, returns_failure_when_it_does_not_converge) {
  auto wide = solve(20);
  ASSERT_EQ(wide.rc, 0);

  auto stopped = solve(1);
  EXPECT_GE(stopped.rc, 20);
  EXPECT_LT(stopped.diag, 0.);
  EXPECT_GT(std::fabs(stopped.x[1] - wide.x[1]), 1e-3);
}
