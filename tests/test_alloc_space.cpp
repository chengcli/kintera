// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

// kintera
#include <kintera/constants.h>
#include <kintera/math/leastsq_kkt.h>
#include <kintera/thermo/equilibrate_uv.h>
#include <kintera/utils/alloc.h>
#include <kintera/vapors/vapor_functions.h>

using namespace kintera;

namespace {

constexpr size_t kGuard = 4096;
constexpr unsigned char kCanary = 0xA5;

// A buffer of exactly `budget` bytes followed by a canary zone. The solvers run
// on the HostBump pool, so any write past the advertised budget hits the zone.
struct GuardedPool {
  explicit GuardedPool(size_t budget)
      : budget(budget), bytes(budget + kGuard, kCanary) {}
  char* work() { return reinterpret_cast<char*>(bytes.data()); }
  bool guard_intact() const {
    for (size_t i = budget; i < bytes.size(); ++i)
      if (bytes[i] != kCanary) return false;
    return true;
  }
  size_t budget;
  std::vector<unsigned char> bytes;
};

template <typename T>
class AllocSpaceTest : public ::testing::Test {};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(AllocSpaceTest, ScalarTypes);

TYPED_TEST(AllocSpaceTest, leastsq_kkt_stays_in_budget) {
  using T = TypeParam;
  for (int n2 = 1; n2 <= 5; ++n2) {
    for (int n3 = 2; n3 <= 11; ++n3) {
      std::vector<T> a(n2 * n2, 0.), b(n2), c(n3 * n2), d(n3);
      for (int i = 0; i < n2; ++i) {
        a[i * n2 + i] = 2. + i;
        b[i] = 1. + 0.5 * i;
      }
      // alternate-sign rows with tight bounds so constraints activate
      for (int k = 0; k < n3; ++k) {
        for (int j = 0; j < n2; ++j) c[k * n2 + j] = ((k + j) % 2 ? -1. : 1.);
        d[k] = 0.1;
      }
      GuardedPool pool(leastsq_kkt_space<T>(n2, n3));
      int max_iter = n3 + 1;
      leastsq_kkt<T, PoolBackend::HostBump>(b.data(), a.data(), c.data(),
                                            d.data(), n2, n2, n3, 0, &max_iter,
                                            0., pool.work());
      EXPECT_TRUE(pool.guard_intact()) << "n2=" << n2 << " n3=" << n3;
    }
  }
}

// leastsq_kkt's budget plus the direct-solve hook buffers. Bounds are
// non-negative (x = 0 feasible) and rows repeat, so dependent rows come up.
TYPED_TEST(AllocSpaceTest, leastsq_kkt_feasible_origin_stays_in_budget) {
  using T = TypeParam;
  for (int n2 = 1; n2 <= 5; ++n2) {
    for (int n3 = 2; n3 <= 11; ++n3) {
      std::vector<T> a(n2 * n2, 0.), b(n2), c(n3 * n2), d(n3);
      for (int i = 0; i < n2; ++i) {
        a[i * n2 + i] = 2. + i;
        b[i] = 1. + 0.5 * i;
      }
      for (int k = 0; k < n3; ++k) {
        for (int j = 0; j < n2; ++j) c[k * n2 + j] = ((k + j) % 2 ? -1. : 1.);
        d[k] = 0.1;
      }
      GuardedPool pool(leastsq_kkt_feasible_origin_space<T>(n2, n3));
      int max_iter = n3 + 1;
      leastsq_kkt_feasible_origin<T, PoolBackend::HostBump>(
          b.data(), a.data(), c.data(), d.data(), n2, n2, n3, 0, &max_iter, 0.,
          pool.work());
      EXPECT_TRUE(pool.guard_intact()) << "n2=" << n2 << " n3=" << n3;
    }
  }
}

// dry + nr vapours + nr clouds, every vapour supersaturated, so the KKT path
// (the one that uses the pool) runs and forms cloud.
TYPED_TEST(AllocSpaceTest, equilibrate_uv_stays_in_budget) {
  using T = TypeParam;
  for (int nr = 1; nr <= 5; ++nr) {
    for (int uv_solver = 0; uv_solver <= 1; ++uv_solver) {
      int ngas = 1 + nr, ns = 1 + 2 * nr;
      std::vector<T> stoich(ns * nr, 0.), u0(ns, 0.), cv(ns), conc(ns, 0.);
      std::vector<T> gain(nr * nr), diag(1);
      std::vector<int> kind(nr, 0), rset(nr);
      std::vector<double> params(nr * KSVP_NPARAM, 0.);
      std::vector<user_func1> lsvp(nr, h2o_ideal);
      std::vector<user_func1> lsvp_ddT(nr, h2o_ideal_ddT);
      std::vector<user_func2> extra(ns, nullptr);
      T temp = 300.;
      cv[0] = 2.5 * constants::Rgas;
      conc[0] = 40.;
      for (int j = 0; j < nr; ++j) {
        stoich[(1 + j) * nr + j] = -1.;
        stoich[(1 + nr + j) * nr + j] = 1.;
        cv[1 + j] = 4. * constants::Rgas;
        cv[1 + nr + j] = 9. * constants::Rgas;
        u0[1 + nr + j] = -4.5e4;
        conc[1 + j] = 3. * exp(h2o_ideal(temp)) / (constants::Rgas * temp);
        rset[j] = j;
      }
      T h0 = 0.;
      for (int i = 0; i < ns; ++i) h0 += conc[i] * (u0[i] + cv[i] * temp);
      int nactive = 0, max_iter = 20;
      GuardedPool pool(equilibrate_uv_space<T>(ns, nr));
      equilibrate_uv<T, PoolBackend::HostBump>(
          gain.data(), diag.data(), &temp, conc.data(), h0, stoich.data(), ns,
          nr, ngas, u0.data(), cv.data(), lsvp.data(), lsvp_ddT.data(),
          kind.data(), params.data(), extra.data(), extra.data(), 1.e-6,
          &max_iter, rset.data(), &nactive, uv_solver, pool.work());
      EXPECT_TRUE(pool.guard_intact())
          << "nr=" << nr << " solver=" << uv_solver;
      EXPECT_TRUE(std::isfinite(temp));
      EXPECT_GT(conc[1 + nr], 0.) << "no cloud: the pool path did not run";
    }
  }
}

}  // namespace
