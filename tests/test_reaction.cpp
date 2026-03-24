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

TEST_P(DeviceTest, reaction_falloff_type) {
  // Default falloff type is "none" (Lindemann or simple three-body)
  Reaction r1;
  EXPECT_EQ(r1.falloff_type(), "none");
  
  // Test setting Troe falloff type
  r1.falloff_type("Troe");
  EXPECT_EQ(r1.falloff_type(), "Troe");
  
  // Test setting SRI falloff type
  Reaction r2;
  r2.falloff_type("SRI");
  EXPECT_EQ(r2.falloff_type(), "SRI");
  
  // Test Reaction created from equation has default falloff type
  auto r3 = Reaction("2 OH (+ M) <=> H2O2 (+ M)");
  EXPECT_EQ(r3.falloff_type(), "none");
  
  // Test chaining: set both efficiencies and falloff type
  r3.efficiencies({{"AR", 0.7}, {"H2", 2.0}, {"H2O", 6.0}});
  r3.falloff_type("Troe");
  EXPECT_EQ(r3.efficiencies().size(), 3);
  EXPECT_EQ(r3.falloff_type(), "Troe");
}
