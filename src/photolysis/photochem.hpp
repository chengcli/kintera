#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>

// kintera
#include <kintera/photolysis/photolysis.hpp>
#include <kintera/species.hpp>

// arg
#include <kintera/add_arg.h>

namespace YAML {
class Node;
}

namespace kintera {

struct PhotoChemOptionsImpl final : public SpeciesThermoImpl {
  static std::shared_ptr<PhotoChemOptionsImpl> create() {
    auto op = std::make_shared<PhotoChemOptionsImpl>();
    op->photolysis() = PhotolysisOptionsImpl::create();
    return op;
  }

  static std::shared_ptr<PhotoChemOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);
  static std::shared_ptr<PhotoChemOptionsImpl> from_yaml(
      YAML::Node const& config, bool verbose = false);
  static std::shared_ptr<PhotoChemOptionsImpl> from_kinetics_base(
      std::string const& master_input_path,
      std::string const& photo_catalog_path = "",
      std::string const& cross_dir = "", bool verbose = false);

  void report(std::ostream& os) const {
    os << "-- photochemistry options --\n";
    os << "* evolve_temperature = " << (evolve_temperature() ? "true" : "false")
       << "\n";
    if (photolysis()) photolysis()->report(os);
  }

  std::vector<Reaction> reactions() const;

  ADD_ARG(PhotolysisOptions, photolysis) = PhotolysisOptionsImpl::create();
  ADD_ARG(bool, evolve_temperature) = false;
  ADD_ARG(bool, verbose) = false;
};
using PhotoChemOptions = std::shared_ptr<PhotoChemOptionsImpl>;

class PhotoChemImpl : public torch::nn::Cloneable<PhotoChemImpl> {
 public:
  static std::shared_ptr<PhotoChemImpl> create(
      PhotoChemOptions const& opts, torch::nn::Module* p,
      std::string const& name = "photochem");

  torch::Tensor stoich;
  PhotoChemOptions options;
  Photolysis photolysis_evaluator;

  PhotoChemImpl() : options(PhotoChemOptionsImpl::create()) {}
  explicit PhotoChemImpl(const PhotoChemOptions& options_);
  void reset() override;

  torch::Tensor forward(torch::Tensor temp, torch::Tensor conc,
                        torch::Tensor actinic_flux);

  torch::Tensor jacobian(torch::Tensor conc, torch::Tensor rate) const;

 private:
  torch::Tensor react_stoich_;
};

TORCH_MODULE(PhotoChem);

}  // namespace kintera

#undef ADD_ARG
