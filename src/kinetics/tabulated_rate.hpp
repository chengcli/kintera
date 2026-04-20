#pragma once

// C/C++
#include <set>
#include <string>
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

//! Data structure holding a loaded rate table
struct RateTable {
  //! Coordinate arrays, one per dimension
  std::vector<torch::Tensor> coords;
  //! Values tensor, shape (n0, n1, ..., nd)
  torch::Tensor values;
};

//! Read a rate table from a text file
/*!
 * File format:
 * \code
 * ndim
 * size_0
 * val_0_0 val_0_1 ... val_0_{size_0-1}
 * size_1
 * val_1_0 val_1_1 ...
 * ...
 * nvalues
 * v_0 v_1 v_2 ...
 * \endcode
 *
 * Lines starting with '#' are skipped.
 * Values and coordinates can be on multiple lines.
 */
RateTable read_rate_table(std::string const& filepath);

//! Options for tabulated reaction rate constants
struct TabulatedRateOptionsImpl {
  static std::shared_ptr<TabulatedRateOptionsImpl> create() {
    return std::make_shared<TabulatedRateOptionsImpl>();
  }
  static std::shared_ptr<TabulatedRateOptionsImpl> from_yaml(
      const YAML::Node& node);

  std::string name() const { return "tabulated"; }

  std::shared_ptr<TabulatedRateOptionsImpl> clone() const {
    return std::make_shared<TabulatedRateOptionsImpl>(*this);
  }

  void report(std::ostream& os) const {
    os << "* reactions = " << fmt::format("{}", reactions()) << "\n"
       << "* log_interpolation = " << (log_interpolation() ? "true" : "false")
       << "\n"
       << "* files = " << fmt::format("{}", files()) << "\n";
  }

  //! Reactions handled by this evaluator
  ADD_ARG(std::vector<Reaction>, reactions) = {};

  //! Table data files, one per reaction
  ADD_ARG(std::vector<std::string>, files) = {};

  //! Whether to interpolate in log10 space
  ADD_ARG(bool, log_interpolation) = true;

  //! Loaded table data (populated during from_yaml or manually)
  ADD_ARG(std::vector<RateTable>, tables) = {};
};
using TabulatedRateOptions = std::shared_ptr<TabulatedRateOptionsImpl>;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set,
                        TabulatedRateOptions op);

class TabulatedRateImpl : public torch::nn::Cloneable<TabulatedRateImpl> {
 public:
  //! options with which this module was constructed
  TabulatedRateOptions options;

  //! Constructor
  TabulatedRateImpl() : options(TabulatedRateOptionsImpl::create()) {}
  explicit TabulatedRateImpl(TabulatedRateOptions const& options_);
  void reset() override;
  void pretty_print(std::ostream& os) const override;

  //! Compute the rate constant by interpolation
  /*!
   * \param T temperature [K], shape (...)
   * \param P pressure [Pa], shape (...)
   * \param C concentration [mol/m^3], shape (..., nspecies)
   * \param other additional parameters
   * \return reaction rate constant in (mol, m, s), shape (..., nreaction)
   */
  torch::Tensor forward(torch::Tensor T, torch::Tensor P, torch::Tensor C,
                        std::map<std::string, torch::Tensor> const& other);

 private:
  //! Coordinate buffers for each reaction and dimension
  //! Stored as registered buffers named "coords_{rxn}_{dim}"
  //! Values buffer named "values_{rxn}"
  int nreaction_ = 0;
  int ndim_ = 0;

  //! Cached buffer references (populated in reset)
  std::vector<std::vector<torch::Tensor>> coords_cache_;
  std::vector<torch::Tensor> values_cache_;
};
TORCH_MODULE(TabulatedRate);

}  // namespace kintera
