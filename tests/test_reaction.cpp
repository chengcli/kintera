// external
#include <gtest/gtest.h>

// kintera
#include <kintera/kintera_formatter.hpp>
#include <kintera/reaction.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;

TEST_P(DeviceTest, reaction) {
  auto reaction = Reaction("H2O => H2O(l)");
  std::cout << fmt::format("{}", reaction) << std::endl;
}

TEST_P(DeviceTest, reaction_with_efficiencies) {
  Reaction r1;
  EXPECT_TRUE(r1.efficiencies().empty());
  
  std::map<std::string, double> eff = {{"N2", 1.0}, {"O2", 0.4}, {"AR", 0.83}};
  r1.efficiencies(eff);
  EXPECT_EQ(r1.efficiencies().size(), 3);
  EXPECT_DOUBLE_EQ(r1.efficiencies().at("N2"), 1.0);
  EXPECT_DOUBLE_EQ(r1.efficiencies().at("O2"), 0.4);
  EXPECT_DOUBLE_EQ(r1.efficiencies().at("AR"), 0.83);
  
  // Test Reaction created from equation has empty efficiencies by default
  auto r2 = Reaction("H2O => H2O(l)");
  EXPECT_TRUE(r2.efficiencies().empty());
  
  // Test that efficiencies can be set and retrieved
  r2.efficiencies({{"H2O", 15.4}});
  EXPECT_EQ(r2.efficiencies().size(), 1);
  EXPECT_DOUBLE_EQ(r2.efficiencies().at("H2O"), 15.4);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
