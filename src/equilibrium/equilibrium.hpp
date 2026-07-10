#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/torch.h>

#include <kintera/add_arg.h>

namespace kintera {

using Matrix = std::vector<std::vector<double>>;

struct EquilibriumOptionsImpl final {
  static std::shared_ptr<EquilibriumOptionsImpl> create() {
    return std::make_shared<EquilibriumOptionsImpl>();
  }

  static std::shared_ptr<EquilibriumOptionsImpl>
  from_yaml(std::string const &filename, bool verbose = false);

  void report(std::ostream &os) const;
  void validate() const;

  ADD_ARG(std::vector<std::string>, components);
  ADD_ARG(std::vector<std::string>, elements);
  ADD_ARG(std::vector<std::string>, phases);
  ADD_ARG(std::vector<std::string>, reactions);
  ADD_ARG(std::vector<int>, phase_ids);
  ADD_ARG(Matrix, stoich);
  ADD_ARG(Matrix, element_matrix);
  ADD_ARG(int, gas_phase) = 0;
  ADD_ARG(double, standard_pressure) = 1.e5;
  ADD_ARG(int, max_iter) = 50;
  ADD_ARG(double, ftol) = 1.e-8;
  ADD_ARG(double, mole_floor) = 1.e-30;
};
using EquilibriumOptions = std::shared_ptr<EquilibriumOptionsImpl>;

class EquilibriumImpl : public torch::nn::Cloneable<EquilibriumImpl> {
public:
  EquilibriumOptions options;

  torch::Tensor stoich;
  torch::Tensor phase_ids;
  torch::Tensor element_matrix;

  EquilibriumImpl() : options(EquilibriumOptionsImpl::create()) {}
  explicit EquilibriumImpl(EquilibriumOptions const &options_);

  void reset() override;
  void pretty_print(std::ostream &os) const override;

  //! Solve fixed-temperature, fixed-pressure multiphase equilibrium.
  /*!
   * Shapes use arbitrary broadcast batch dimensions:
   *   temp, pres: (...)
   *   moles:      (..., ncomponent)
   *   log_k:      (..., nreaction)
   *
   * Returns equilibrium moles, the final reaction Jacobian/gain matrix, and
   * diagnostics (..., 4): status, iterations, equilibrium error, and element
   * conservation error.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  forward(torch::Tensor temp, torch::Tensor pres, torch::Tensor moles,
          torch::Tensor log_k, bool warm_start = false);
};

TORCH_MODULE(Equilibrium);

} // namespace kintera
