#pragma once

// C/C++
#include <map>
#include <memory>
#include <set>
#include <vector>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include <kintera/kintera_formatter.hpp>
#include <kintera/reaction.hpp>

// arg
#include <kintera/add_arg.h>

namespace YAML {
class Node;
}

namespace kintera {

//! Options to initialize all photolysis reaction rate constants
struct PhotolysisOptionsImpl {
  static std::shared_ptr<PhotolysisOptionsImpl> create() {
    return std::make_shared<PhotolysisOptionsImpl>();
  }

  //! Create options from YAML node
  static std::shared_ptr<PhotolysisOptionsImpl> from_yaml(
      const YAML::Node& node,
      std::shared_ptr<PhotolysisOptionsImpl> derived_type_ptr = nullptr);

  virtual std::string name() const { return "photolysis"; }
  virtual ~PhotolysisOptionsImpl() = default;

  void report(std::ostream& os) const {
    os << "* reactions = " << fmt::format("{}", reactions()) << "\n"
       << "* nwave = " << wavelength().size() << "\n"
       << "* ntemp = " << temperature().size() << "\n"
       << "* nbranch = " << branches().size() << "\n";
  }

  //! List of photolysis reactions
  ADD_ARG(std::vector<Reaction>, reactions) = {};

  //! Wavelength grid [nm]
  ADD_ARG(std::vector<double>, wavelength) = {};

  //! Temperature grid [K] for temperature-dependent cross-sections
  ADD_ARG(std::vector<double>, temperature) = {};

  //! Cross-section data [cm^2 molecule^-1]
  //! Shape: (ntemp, nwave, nbranch) flattened
  ADD_ARG(std::vector<double>, cross_section) = {};

  //! Branch compositions for each reaction
  //! Each branch is a Composition (map<string, double>)
  ADD_ARG(std::vector<std::vector<Composition>>, branches) = {};

  //! Branch names for each reaction
  ADD_ARG(std::vector<std::vector<std::string>>, branch_names) = {};
};
using PhotolysisOptions = std::shared_ptr<PhotolysisOptionsImpl>;

//! Add species to vapor/cloud sets from photolysis reactions
void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set, PhotolysisOptions op);

//! Photolysis rate evaluator module
/*!
 * Computes photolysis rates by integrating cross-sections weighted by
 * actinic flux over wavelength:
 *
 * \f[
 *    k = \int_{\lambda_1}^{\lambda_2} \sigma(\lambda, T) F(\lambda) d\lambda
 * \f]
 *
 * where \f$ \sigma \f$ is the cross-section, \f$ F \f$ is the actinic flux,
 * and \f$ \lambda \f$ is the wavelength.
 */
class PhotolysisImpl : public torch::nn::Cloneable<PhotolysisImpl> {
 public:
  //! Wavelength grid [nm], shape (nwave,)
  torch::Tensor wavelength;

  //! Temperature grid [K], shape (ntemp,)
  torch::Tensor temp_grid;

  //! Cross-section data [cm^2 molecule^-1], shape (nreaction, ntemp, nwave,
  //! nbranch)
  std::vector<torch::Tensor> cross_section;

  //! Stoichiometry for each branch of each reaction
  //! Shape: (nreaction, nbranch, nspecies) - stored as vector of tensors
  std::vector<torch::Tensor> branch_stoich;

  //! options with which this `PhotolysisImpl` was constructed
  PhotolysisOptions options;

  //! Constructor to initialize the layer
  PhotolysisImpl() : options(PhotolysisOptionsImpl::create()) {}
  explicit PhotolysisImpl(PhotolysisOptions const& options_);
  void reset() override;
  void pretty_print(std::ostream& os) const override;

  //! Compute photolysis rate constants
  /*!
   * \param T temperature [K], shape (...)
   * \param P pressure [Pa], shape (...)
   * \param C concentration [mol/m^3], shape (..., nspecies) - unused but
   *          kept for interface compatibility
   * \param other map containing:
   *        - "wavelength": wavelength grid [nm], shape (nwave,)
   *        - "actinic_flux": actinic flux [photons cm^-2 s^-1 nm^-1],
   *                          shape (nwave, ...) or (..., nwave)
   *        - "stoich": stoichiometry matrix for updating branch weights
   * \return photolysis rate constants [s^-1], shape (..., nreaction)
   */
  torch::Tensor forward(torch::Tensor T, torch::Tensor P, torch::Tensor C,
                        std::map<std::string, torch::Tensor> const& other);

  //! Get effective stoichiometry coefficients for a reaction
  /*!
   * Returns the weighted stoichiometry based on branch ratios
   *
   * \param rxn_idx reaction index
   * \param wave wavelength grid [nm]
   * \param aflux actinic flux
   * \param temp temperature for cross-section interpolation
   * \return effective stoichiometry coefficients
   */
  torch::Tensor get_effective_stoich(int rxn_idx, torch::Tensor wave,
                                     torch::Tensor aflux, torch::Tensor temp);

  //! Interpolate cross-section to given wavelength and temperature
  /*!
   * \param rxn_idx reaction index
   * \param wave wavelength [nm]
   * \param temp temperature [K]
   * \return interpolated cross-section [cm^2], shape (..., nbranch)
   */
  torch::Tensor interp_cross_section(int rxn_idx, torch::Tensor wave,
                                     torch::Tensor temp);

 private:
  //! Number of reactions
  int _nreaction;

  //! Number of branches per reaction
  std::vector<int> _nbranches;
};

TORCH_MODULE(Photolysis);

}  // namespace kintera

#undef ADD_ARG

