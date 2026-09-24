// Transcription guard: the scalar per-cell h2diss EOS must match the torch
// h2diss::eval on a (T,c) grid; both read the same (2,3,9) NASA-9 block.

// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <cstdio>
#include <vector>

// kintera
#include <kintera/species.hpp>

#include "../src/thermo/h2_dissociation.hpp"
#include "../src/thermo/h2_dissociation_scalar.hpp"

using namespace kintera;

namespace {
constexpr double kNH = 1.6667, kNHe = 0.16667;  // a solar-like H/He mixture

std::vector<double> logspace(double lo, double hi, int n) {
  std::vector<double> v(n);
  double a = std::log(lo), b = std::log(hi);
  for (int i = 0; i < n; ++i) v[i] = std::exp(a + (b - a) * i / (n - 1));
  return v;
}

// running max of |a-b| and of |a-b|/(|b| rtol + atol); <= 1 means within tol
struct Acc {
  const char* name;
  double maxabs = 0, maxrel = 0;
  double Tworst = 0, cworst = 0;
  void add(double a, double b, double rtol, double atol, double T, double c) {
    double e = std::fabs(a - b);
    double r = e / (std::fabs(b) * rtol + atol);
    if (e > maxabs) maxabs = e;
    if (r > maxrel) {
      maxrel = r;
      Tworst = T;
      cworst = c;
    }
  }
};

torch::Tensor coeffs() {
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  return nasa9_coeffs_by_name({"H2", "H", "He"}, opt).contiguous();
}
}  // namespace

TEST(H2DissScalar, MatchesTorchOnGrid) {
  auto ab_t = coeffs();
  ASSERT_EQ(ab_t.numel(), 2 * 3 * 9);
  const double* ab = ab_t.data_ptr<double>();

  std::vector<double> Ts = logspace(200., 4500., 71);
  std::vector<double> cs = logspace(1e-2, 1e4, 41);
  int N = (int)(Ts.size() * cs.size());

  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  auto Tf = torch::empty({N}, opt), cf = torch::empty({N}, opt);
  auto *Tp = Tf.data_ptr<double>(), *cp = cf.data_ptr<double>();
  int k = 0;
  for (double T : Ts)
    for (double c : cs) {
      Tp[k] = T;
      cp[k] = c;
      ++k;
    }

  h2diss::Result R = h2diss::eval(Tf, cf, kNH, kNHe, ab_t);
  auto cz = R.cz.contiguous(), czddc = R.cz_ddC.contiguous(),
       cpR = R.cp_R.contiguous(), cvR = R.cv_R.contiguous(),
       eR = R.e_R.contiguous();
  auto *czp = cz.data_ptr<double>(), *czddcp = czddc.data_ptr<double>(),
       *cpRp = cpR.data_ptr<double>(), *cvRp = cvR.data_ptr<double>(),
       *eRp = eR.data_ptr<double>();

  const double rtol = 1e-12, atol = 1e-9;
  Acc aCz{"cz"}, aCzc{"cz_ddC"}, aCp{"cp_R"}, aCv{"cv_R"}, aE{"e_R"};
  for (int i = 0; i < N; ++i) {
    double T = Tp[i], c = cp[i];
    h2diss_scalar::Result s = h2diss_scalar::eval(T, c, kNH, kNHe, ab);
    aCz.add(s.cz, czp[i], rtol, atol, T, c);
    aCzc.add(s.cz_ddC, czddcp[i], rtol, atol, T, c);
    aCp.add(s.cp_R, cpRp[i], rtol, atol, T, c);
    aCv.add(s.cv_R, cvRp[i], rtol, atol, T, c);
    aE.add(s.e_R, eRp[i], rtol, atol, T, c);
  }

  std::printf(
      "grid: %d cells, T[200,4500]K x c[1e-2,1e4]mol/m3, rtol=%g atol=%g\n", N,
      rtol, atol);
  for (Acc* a : {&aCz, &aCzc, &aCp, &aCv, &aE}) {
    std::printf(
        "  %-7s max|abs|=%.3e  max(residual/tol)=%.3e  (worst @ T=%.0f "
        "c=%.3g)\n",
        a->name, a->maxabs, a->maxrel, a->Tworst, a->cworst);
    EXPECT_LE(a->maxrel, 1.0)
        << a->name << " worst at T=" << a->Tworst << " c=" << a->cworst;
  }
}

TEST(H2DissScalar, T0ReferenceIsCIndependent) {
  auto ab_t = coeffs();
  const double* ab = ab_t.data_ptr<double>();
  double e0 = h2diss_scalar::e0_ref(kNH, kNHe, ab);
  double maxdev = 0;
  for (double c : logspace(1e-4, 1e5, 61)) {
    double u =
        h2diss_scalar::speciate(h2diss_scalar::kTref, c, kNH, kNHe, ab).U;
    maxdev = std::fmax(maxdev, std::fabs(u - e0));
  }
  std::printf("  e0_ref  |s0.U(c)-e0|max=%.3e (e0=%.6f)\n", maxdev, e0);
  EXPECT_LE(maxdev, 1e-9 + 1e-12 * std::fabs(e0));
}
