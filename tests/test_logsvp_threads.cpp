// external
#include <gtest/gtest.h>

// C/C++
#include <atomic>
#include <string>
#include <thread>
#include <vector>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/vapors/vapor_functions.h>

#include <kintera/thermo/log_svp.hpp>

using namespace kintera;

// Two threads evaluate LogSVPFunc with different svp columns at the same
// time, each the way every caller does: init(its options), then call() and
// grad(). Each thread must get its own curve. A shared state rewritten by
// init() lets one thread read the other's columns between its init() and its
// call(), and the values come back silently wrong (same shape).

namespace {

constexpr int kIters = 2000;

struct Curve {
  std::string name;
  double (*value)(double);
  double (*derivative)(double);
};

int run_thread(Curve const& curve, std::atomic<bool> const& go) {
  auto nucleation = NucleationOptionsImpl::create();
  nucleation->logsvp({curve.name});
  auto temp = torch::tensor({250.0, 300.0}, torch::kFloat64);
  double t0 = 250.0, t1 = 300.0;

  while (!go.load()) std::this_thread::yield();
  int wrong = 0;
  for (int i = 0; i < kIters; ++i) {
    LogSVPFunc::init(nucleation);
    auto v = LogSVPFunc::call(temp);
    auto d = LogSVPFunc::grad(temp);
    bool ok = v.size(-1) == 1 && d.size(-1) == 1 &&
              std::abs(v[0][0].item<double>() - curve.value(t0)) < 1.e-10 &&
              std::abs(v[1][0].item<double>() - curve.value(t1)) < 1.e-10 &&
              std::abs(d[0][0].item<double>() - curve.derivative(t0)) <
                  1.e-10 &&
              std::abs(d[1][0].item<double>() - curve.derivative(t1)) < 1.e-10;
    wrong += !ok;
  }
  return wrong;
}

}  // namespace

TEST(LogSVPThreads, TwoConfigsConcurrently) {
  Curve a{"h2o_ideal", h2o_ideal, h2o_ideal_ddT};
  Curve b{"nh3_ideal", nh3_ideal, nh3_ideal_ddT};
  ASSERT_GT(std::abs(a.value(300.0) - b.value(300.0)), 1.e-3);

  // build both tables first, so the threads race on init() and not the build
  for (auto const& c : {a, b}) {
    auto n = NucleationOptionsImpl::create();
    n->logsvp({c.name});
    LogSVPFunc::init(n);
    LogSVPFunc::call(torch::tensor({300.0}, torch::kFloat64));
  }

  std::atomic<bool> go{false};
  int wrong_a = 0, wrong_b = 0;
  std::thread ta([&] { wrong_a = run_thread(a, go); });
  std::thread tb([&] { wrong_b = run_thread(b, go); });
  go = true;
  ta.join();
  tb.join();

  EXPECT_EQ(wrong_a, 0) << a.name << ": " << wrong_a << " of " << kIters
                        << " evaluations returned another thread's curve";
  EXPECT_EQ(wrong_b, 0) << b.name << ": " << wrong_b << " of " << kIters
                        << " evaluations returned another thread's curve";
}
