// external
#include <gtest/gtest.h>

// kintera
#include <kintera/kintera_formatter.hpp>
#include <kintera/reaction.hpp>

TEST_P(DeviceTest, reaction) {
  auto reaction = Reaction("H2O => H2O(l)");
  std::cout << fmt::format("{}", reaction) << std::endl;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
