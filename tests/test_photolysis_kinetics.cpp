// C/C++
#include <algorithm>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/kinetics/kinetics_formatter.hpp>
#include <kintera/photolysis/photochem.hpp>
#include <kintera/photolysis/photolysis.hpp>
#include <kintera/utils/parse_comp_string.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;

class PhotolysisKineticsTest : public DeviceTest {
 protected:
  void SetUp() override {
    DeviceTest::SetUp();
    // Initialize species names for testing
    kintera::species_names = {"CH4", "CH3", "(1)CH2", "(3)CH2",
                              "CH",  "H2",  "H",      "N2"};
    kintera::species_weights = {16.0, 15.0, 14.0, 14.0, 13.0, 2.0, 1.0, 28.0};
    kintera::species_cref_R = {2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 1.5, 2.5};
    kintera::species_uref_R = {0., 0., 0., 0., 0., 0., 0., 0.};
    kintera::species_sref_R = {0., 0., 0., 0., 0., 0., 0., 0.};
    kintera::species_initialized = true;
  }
};

TEST_P(PhotolysisKineticsTest, PhotoChemOptionsWithPhotolysis) {
  auto photo_opts = PhotoChemOptionsImpl::create();
  EXPECT_NE(photo_opts->photolysis(), nullptr);
  EXPECT_EQ(photo_opts->photolysis()->reactions().size(), 0);
}

TEST_P(PhotolysisKineticsTest, PhotolysisReactionsInPhotoChem) {
  auto photo_opts = PhotoChemOptionsImpl::create();

  photo_opts->photolysis()->wavelength() = {100., 150., 200.};
  photo_opts->photolysis()->temperature() = {200., 300.};
  photo_opts->photolysis()->reactions().push_back(Reaction("N2 => N2"));
  photo_opts->photolysis()->cross_section() = {1.e-18, 2.e-18, 1.e-18};
  photo_opts->photolysis()->branches().push_back({parse_comp_string("N2:1")});

  auto all_reactions = photo_opts->reactions();
  EXPECT_EQ(all_reactions.size(), 1);
}

TEST_P(PhotolysisKineticsTest, PhotoChemModuleWithPhotolysis) {
  auto photo_opts = PhotoChemOptionsImpl::create();

  photo_opts->vapor_ids() = {7};
  photo_opts->cref_R() = {2.5};
  photo_opts->uref_R() = {0.};
  photo_opts->sref_R() = {0.};

  photo_opts->photolysis()->wavelength() = {100., 150., 200.};
  photo_opts->photolysis()->temperature() = {200., 300.};
  photo_opts->photolysis()->reactions().push_back(Reaction("N2 => N2"));
  photo_opts->photolysis()->cross_section() = {1.e-18, 2.e-18, 1.e-18};
  photo_opts->photolysis()->branches().push_back({parse_comp_string("N2:1")});

  PhotoChem photo(photo_opts);
  photo->to(device, dtype);

  EXPECT_EQ(photo->stoich.size(1), 1);
}

TEST_P(PhotolysisKineticsTest, StoichiometryMatrixIncludesPhotolysis) {
  auto photo_opts = PhotoChemOptionsImpl::create();

  photo_opts->vapor_ids() = {0, 7};
  photo_opts->cref_R() = {2.5, 2.5};
  photo_opts->uref_R() = {0., 0.};
  photo_opts->sref_R() = {0., 0.};

  photo_opts->photolysis()->wavelength() = {100., 200.};
  photo_opts->photolysis()->temperature() = {200., 300.};
  photo_opts->photolysis()->reactions().push_back(Reaction("N2 => N2"));
  photo_opts->photolysis()->cross_section() = {1.e-18, 1.e-18};
  photo_opts->photolysis()->branches().push_back({parse_comp_string("N2:1")});

  PhotoChem photo(photo_opts);
  photo->to(device, dtype);

  EXPECT_EQ(photo->stoich.size(1), 1);
}

TEST_P(PhotolysisKineticsTest, PhotoChemForward) {
  auto photo_opts = PhotoChemOptionsImpl::create();
  photo_opts->vapor_ids() = {7};
  photo_opts->cref_R() = {2.5};
  photo_opts->uref_R() = {0.};
  photo_opts->sref_R() = {0.};
  photo_opts->photolysis()->wavelength() = {100., 150., 200.};
  photo_opts->photolysis()->temperature() = {200., 300.};
  photo_opts->photolysis()->reactions().push_back(Reaction("N2 => N2"));
  photo_opts->photolysis()->cross_section() = {1.e-18, 2.e-18, 1.e-18};
  photo_opts->photolysis()->branches().push_back({parse_comp_string("N2:1")});

  PhotoChem photo(photo_opts);
  photo->to(device, dtype);

  auto temp = torch::tensor({250.0}, torch::device(device).dtype(dtype));
  auto conc = torch::tensor({{1.0e18}}, torch::device(device).dtype(dtype));
  auto wave = photo->photolysis_evaluator->wavelength.to(device, dtype);
  auto rate = photo->forward(temp, conc, torch::ones_like(wave) * 1.0e14);
  EXPECT_EQ(rate.size(-1), 1);
  EXPECT_TRUE(rate.isfinite().all().item<bool>());
}

INSTANTIATE_TEST_SUITE_P(
    DeviceTests, PhotolysisKineticsTest,
    testing::Values(Parameters{torch::kCPU, torch::kFloat64},
                    Parameters{torch::kCUDA, torch::kFloat64}),
    [](const testing::TestParamInfo<PhotolysisKineticsTest::ParamType>& info) {
      std::string name = torch::Device(info.param.device_type).str();
      name += "_";
      name += torch::toString(info.param.dtype);
      std::replace(name.begin(), name.end(), '.', '_');
      return name;
    });

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
