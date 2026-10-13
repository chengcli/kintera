#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <kintera/species.hpp>

namespace kintera {
extern std::vector<std::string> species_phases;
extern std::vector<std::array<double, 9>> species_nasa9_low;
extern std::vector<std::array<double, 9>> species_nasa9_high;
extern std::vector<double> species_nasa9_Tmid;
}  // namespace kintera

using namespace kintera;

namespace {

class ScopedCurrentPath {
 public:
  explicit ScopedCurrentPath(std::filesystem::path path)
      : original_(std::filesystem::current_path()) {
    std::filesystem::current_path(std::move(path));
  }

  ~ScopedCurrentPath() noexcept {
    std::error_code ec;
    std::filesystem::current_path(original_, ec);
  }

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

TEST(SpeciesInit, clear_invalidates_success_and_same_card_reloads) {
  auto card = YAML::Load(R"(
species:
  - {name: N2, phase: gas, composition: {N: 2}, cv_R: 2.5}
)");
  init_species_from_yaml(card);
  ASSERT_TRUE(species_initialized);
  clear_species_registry();
  EXPECT_FALSE(species_initialized);
  EXPECT_TRUE(species_names.empty());
  ensure_species_initialized(card);
  EXPECT_TRUE(species_initialized);
  EXPECT_EQ(species_names, std::vector<std::string>{"N2"});
}

TEST(SpeciesInit, failed_parse_clears_partial_state_and_never_caches_success) {
  auto valid = YAML::Load(R"(
species:
  - {name: N2, phase: gas, composition: {N: 2}, cv_R: 2.5}
)");
  for (auto const& bad_record :
       {"{name: H2O, phase: plasma, composition: {H: 2, O: 1}}",
        "{name: H2O, phase: gas, composition: {H: 2, O: 1}, cv_R: bad}"}) {
    init_species_from_yaml(valid);
    auto invalid = YAML::Load(
        std::string("species:\n"
                    "  - {name: H2, phase: gas, composition: {H: 2}}\n"
                    "  - ") +
        bad_record);
    for (int attempt = 0; attempt < 3; ++attempt) {
      EXPECT_ANY_THROW(ensure_species_initialized(invalid));
      EXPECT_FALSE(species_initialized);
      EXPECT_TRUE(species_names.empty());
      EXPECT_TRUE(species_phases.empty());
      EXPECT_TRUE(species_weights.empty());
      EXPECT_TRUE(species_cref_R.empty());
      EXPECT_TRUE(species_uref_R.empty());
      EXPECT_TRUE(species_sref_R.empty());
      EXPECT_TRUE(species_nasa9_low.empty());
      EXPECT_TRUE(species_nasa9_high.empty());
      EXPECT_TRUE(species_nasa9_Tmid.empty());
    }
    ensure_species_initialized(valid);
    EXPECT_TRUE(species_initialized);
    EXPECT_EQ(species_names, std::vector<std::string>{"N2"});
  }
}
