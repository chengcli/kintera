#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <kintera/species.hpp>

namespace kintera {
extern std::vector<std::array<double, 9>> species_nasa9_low;
extern std::vector<std::array<double, 9>> species_nasa9_high;
}  // namespace kintera

using namespace kintera;

namespace {

class ScopedCurrentPath {
 public:
  explicit ScopedCurrentPath(std::filesystem::path path)
      : original_(std::filesystem::current_path()) {
    std::filesystem::current_path(std::move(path));
  }

  ~ScopedCurrentPath() { std::filesystem::current_path(original_); }

 private:
  std::filesystem::path original_;
};

}  // namespace

TEST(SpeciesInit, prefers_current_directory_nasa9_file) {
  auto temp_dir =
      std::filesystem::temp_directory_path() / "kintera_test_species_nasa9";
  std::filesystem::create_directories(temp_dir);

  {
    std::ofstream yaml(temp_dir / "species.yaml");
    yaml << "species:\n";
    yaml << "  - name: dry\n";
    yaml << "    composition: {H: 2}\n";
    yaml << "    cv_R: 2.5\n";
    yaml << "  - name: H2O\n";
    yaml << "    composition: {H: 2, O: 1}\n";
    yaml << "    cv_R: 2.5\n";
  }

  {
    std::ofstream data(temp_dir / "nasa9.dat");
    data << "H2O\n";
    data << "1 2 3 4 5\n";
    data << "6 7 8 9 10\n";
    data << "11 12 13 14 15\n";
    data << "16 17 18 19 20\n";
  }

  ScopedCurrentPath cwd(temp_dir);
  init_species_from_yaml((temp_dir / "species.yaml").string());

  ASSERT_GE(species_nasa9_low.size(), 2u);
  ASSERT_GE(species_nasa9_high.size(), 2u);

  EXPECT_DOUBLE_EQ(species_nasa9_low[1][0], 1.0);
  EXPECT_DOUBLE_EQ(species_nasa9_low[1][6], 7.0);
  EXPECT_DOUBLE_EQ(species_nasa9_low[1][7], 9.0);
  EXPECT_DOUBLE_EQ(species_nasa9_low[1][8], 10.0);
  EXPECT_DOUBLE_EQ(species_nasa9_high[1][0], 11.0);
  EXPECT_DOUBLE_EQ(species_nasa9_high[1][6], 17.0);
  EXPECT_DOUBLE_EQ(species_nasa9_high[1][7], 19.0);
  EXPECT_DOUBLE_EQ(species_nasa9_high[1][8], 20.0);
}
