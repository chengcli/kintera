// V3 K5 (UCzhangxi/kintera v3-design/K 8d962405, K.rst:790, :636-639): the
// saturation-failure count lives in failures() only. ThermoYImpl has no
// nfail_ member any more, and take_saturation_adjustment_failures() reads and
// zeroes row 0 of failures().

// C/C++
#include <type_traits>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/thermo/thermo.hpp>

using namespace kintera;

namespace {

template <typename T, typename = void>
struct has_nfail : std::false_type {};

template <typename T>
struct has_nfail<T, std::void_t<decltype(std::declval<T&>().nfail_)>>
    : std::true_type {};

}  // namespace

TEST(FailureCounter, thermo_y_has_no_nfail_member) {
  EXPECT_FALSE(has_nfail<ThermoYImpl>::value);
}

TEST(FailureCounter, take_reads_and_zeroes_row_0) {
  auto op = ThermoOptionsImpl::create();
  ThermoY thermo(op);
  EXPECT_EQ(thermo->take_saturation_adjustment_failures(), 0);  // no counter
  thermo->reset_failures();  // makes the counter on the module's device
  auto f = thermo->failures();
  ASSERT_TRUE(f.defined());
  f.copy_(torch::tensor({3, 4, 5}, torch::kLong));
  EXPECT_EQ(thermo->take_saturation_adjustment_failures(), 3);
  EXPECT_TRUE(
      torch::equal(thermo->failures(), torch::tensor({0, 4, 5}, torch::kLong)));
  EXPECT_EQ(thermo->take_saturation_adjustment_failures(), 0);
}
