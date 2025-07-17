#pragma once

// C/C++
#include <set>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include <kintera/kintera_formatter.hpp>
#include <kintera/reaction.hpp>
#include <kintera/thermo/nucleation.hpp>

// arg
#include <kintera/add_arg.h>

namespace YAML {
class Node;
}

namespace kintera {

//! Options to initialize all reaction rate constants
struct EvaporationOptions : public NucleationOptions {
  static EvaporationOptions from_yaml(const YAML::Node& node);
  void report(std::ostream& os) const {
    NucleationOptions::report(os);
    os << "* Tref = " << Tref() << " K\n"
       << "* Pref = " << Pref() << " Pa\n"
       << "* diff_c = " << fmt::format("{}", diff_c()) << "\n"
       << "* diff_T = " << fmt::format("{}", diff_T()) << "\n"
       << "* diff_P = " << fmt::format("{}", diff_P()) << "\n"
       << "* vm = " << fmt::format("{}", vm()) << "\n"
       << "* diameter = " << fmt::format("{}", diameter()) << "\n";
  }

  // reference temperature
  ADD_ARG(double, Tref) = 300.0;

  // reference pressure
  ADD_ARG(double, Pref) = 1.e5;

  //! Diffusivity [m^2/s] at reference temperature and pressure
  ADD_ARG(std::vector<double>, diff_c) = {};

  //! Diffusivity temperature exponent
  ADD_ARG(std::vector<double>, diff_T) = {};

  //! Diffusivity pressure exponent
  ADD_ARG(std::vector<double>, diff_P) = {};

  //! Molar volume [m^3/mol]
  ADD_ARG(std::vector<double>, vm) = {};

  //! Particle diameter [m]
  ADD_ARG(std::vector<double>, diameter) = {};
};

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set,
                        EvaporationOptions op);

class EvaporationImpl : public torch::nn::Cloneable<EvaporationImpl> {
 public:
  //! diffusivity m^2/s, shape (nreaction,)
  torch::Tensor diff_c,
      diff_T,  // temperature exponent
      diff_P;  // pressure exponent

  //! molar volume m^3/mol, shape (nreaction,)
  torch::Tensor vm;

  //! log particle diameter m, shape (nreaction,)
  torch::Tensor diameter;

  //! options with which this `EvaporationImpl` was constructed
  EvaporationOptions options;

  //! Constructor to initialize the layer
  EvaporationImpl() = default;
  explicit EvaporationImpl(EvaporationOptions const& options_);
  void reset() override;
  void pretty_print(std::ostream& os) const override;

  //! Compute the rate constant
  /*!
   * \param T temperature [K], shape (...)
   * \param P pressure [pa], shape (...)
   * \param C concentration [mol/m^3], shape (..., nspecies)
   * \param other additional parameters, e.g., concentration
   * \return rate constant in (mol, m, s), shape (..., nreaction)
   */
  torch::Tensor forward(torch::Tensor T, torch::Tensor P, torch::Tensor C,
                        std::map<std::string, torch::Tensor> const& other);
};
TORCH_MODULE(Evaporation);

}  // namespace kintera

#undef ADD_ARG
