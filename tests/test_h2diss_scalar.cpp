// Standalone transcription guard for h2_dissociation_scalar.hpp (Design C / S1).
// Compares the scalar per-cell eval against the torch h2diss::eval on a (T,c)
// grid to ~1e-14 rel. SAME closed-form math; only scalar-loop vs tensor-op
// execution differs. Single source of truth for coefficients: both consume the
// (2,3,9) block from nasa9_coeffs_by_name({"H2","H","He"}). Exit 0 = PASS.
//
// Self-contained (no gtest): the kintera C++ test suite (BUILD_TESTS) is gated
// off and drags in cantera/athena; this links only libkintera + torch. Build &
// run on a COMPUTE node (login gcc is broken) from a scratch dir:
//
//   src=/data/users/xiz/apps/canoe/src/kintera
//   cat > CMakeLists.txt <<'EOF'
//   cmake_minimum_required(VERSION 3.18)
//   project(h2s LANGUAGES CXX)
//   set(CMAKE_CXX_STANDARD 17)
//   find_package(Torch REQUIRED)
//   add_executable(h2s ${SRC}/tests/test_h2diss_scalar.cpp)
//   target_include_directories(h2s PRIVATE ${SRC} ${SRC}/build)
//   target_link_libraries(h2s PRIVATE ${SRC}/build/lib/libkintera_release.so ${TORCH_LIBRARIES})
//   EOF
//   source /data/users/xiz/apps/canoe/buildenv.sh
//   cmake -DSRC=$src -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') .
//   cmake --build . -j8 && ./h2s

#include <cmath>
#include <cstdio>
#include <vector>

#include <kintera/species.hpp>  // nasa9_coeffs_by_name

#include "../src/thermo/h2_dissociation.hpp"         // torch reference
#include "../src/thermo/h2_dissociation_scalar.hpp"  // under test

using namespace kintera;

namespace {
constexpr double kNH = 1.6667, kNHe = 0.16667;  // examples/30 "dry" species

std::vector<double> logspace(double lo, double hi, int n) {
  std::vector<double> v(n);
  double a = std::log(lo), b = std::log(hi);
  for (int i = 0; i < n; ++i) v[i] = std::exp(a + (b - a) * i / (n - 1));
  return v;
}
// running max of |a-b| and of |a-b|/(atol+|b|) (a "tolerance-units" residual)
struct Acc {
  const char* name;
  double maxabs = 0, maxrel = 0;
  double Tworst = 0, cworst = 0;
  void add(double a, double b, double rtol, double atol, double T, double c) {
    double e = std::fabs(a - b);
    double r = e / (std::fabs(b) * rtol + atol);  // <=1 means within tol
    if (e > maxabs) maxabs = e;
    if (r > maxrel) { maxrel = r; Tworst = T; cworst = c; }
  }
};
}  // namespace

int main() {
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);  // all tensors fp64
  auto ab_t = nasa9_coeffs_by_name({"H2", "H", "He"}, opt).contiguous();  // (2,3,9)
  if (ab_t.numel() != 2 * 3 * 9) {
    std::printf("FAIL: ab numel %ld != 54\n", (long)ab_t.numel());
    return 2;
  }
  const double* ab = ab_t.data_ptr<double>();

  std::vector<double> Ts = logspace(200., 4500., 71);
  std::vector<double> cs = logspace(1e-2, 1e4, 41);
  int N = (int)(Ts.size() * cs.size());

  auto Tf = torch::empty({N}, opt), cf = torch::empty({N}, opt);
  auto *Tp = Tf.data_ptr<double>(), *cp = cf.data_ptr<double>();
  int k = 0;
  for (double T : Ts)
    for (double c : cs) { Tp[k] = T; cp[k] = c; ++k; }

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

  std::printf("grid: %d cells, T[200,4500]K x c[1e-2,1e4]mol/m3, rtol=%g atol=%g\n",
              N, rtol, atol);
  int bad = 0;
  for (Acc* a : {&aCz, &aCzc, &aCp, &aCv, &aE}) {
    bool ok = a->maxrel <= 1.0;
    std::printf("  %-7s max|abs|=%.3e  max(residual/tol)=%.3e  %s"
                "  (worst @ T=%.0f c=%.3g)\n",
                a->name, a->maxabs, a->maxrel, ok ? "PASS" : "FAIL",
                a->Tworst, a->cworst);
    if (!ok) ++bad;
  }

  // T0 reference is a model constant (c-independent): the fused loop hoists it.
  double e0 = h2diss_scalar::e0_ref(kNH, kNHe, ab);
  double maxdev = 0;
  for (double c : logspace(1e-4, 1e5, 61)) {
    double u = h2diss_scalar::speciate(h2diss_scalar::kTref, c, kNH, kNHe, ab).U;
    maxdev = std::fmax(maxdev, std::fabs(u - e0));
  }
  bool e0ok = maxdev <= 1e-9 + 1e-12 * std::fabs(e0);
  std::printf("  e0_ref  |s0.U(c)-e0|max=%.3e (e0=%.6f)  %s\n", maxdev, e0,
              e0ok ? "PASS" : "FAIL");
  if (!e0ok) ++bad;

  std::printf(bad ? "\nRESULT: FAIL (%d fields)\n" : "\nRESULT: PASS\n", bad);
  return bad ? 1 : 0;
}
