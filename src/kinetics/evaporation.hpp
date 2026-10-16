#pragma once

// C/C++
#include <algorithm>
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
struct EvaporationOptionsImpl final : public NucleationOptionsImpl {
  static std::shared_ptr<EvaporationOptionsImpl> create() {
    return std::make_shared<EvaporationOptionsImpl>();
  }
  static std::shared_ptr<EvaporationOptionsImpl> from_yaml(
      const YAML::Node& node);

  std::string name() const override { return "evaporation"; }
  EvaporationOptionsImpl() = default;
  EvaporationOptionsImpl(const NucleationOptionsImpl& nucleation)
      : NucleationOptionsImpl(nucleation) {}

  std::shared_ptr<EvaporationOptionsImpl> clone() const {
    return std::make_shared<EvaporationOptionsImpl>(*this);
  }
  void report(std::ostream& os) const {
    NucleationOptionsImpl::report(os);
    os << "* Tref = " << Tref() << " K\n"
       << "* Pref = " << Pref() << " Pa\n"
       << "* diff_c = " << fmt::format("{}", diff_c()) << "\n"
       << "* diff_T = " << fmt::format("{}", diff_T()) << "\n"
       << "* diff_P = " << fmt::format("{}", diff_P()) << "\n"
       << "* vm = " << fmt::format("{}", vm()) << "\n"
       << "* diameter = " << fmt::format("{}", diameter()) << "\n";
    if (std::count(form().begin(), form().end(), "kessler") > 0)
      os << "* form = " << fmt::format("{}", form()) << "\n"
         << "* k3 = " << fmt::format("{}", k3()) << "\n";
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

  //! Rate form per reaction: "diffusion" (default) or "kessler".
  //! An empty vector means "diffusion" for every reaction.
  //!
  //! diffusion: kappa = 12 D vm / d^2 [m^3/(mol s)]
  //! kessler:   kappa = k3 / n, n = P/(R T) + sum of the condensed
  //!            concentrations [mol/m^3]. The rate kappa c_p (c_sat - c_v)+
  //!            is then dx_v/dt = k3 x_p (x_sat - x_v)+ in mole fractions
  //!            x = c / n of all species, gas and condensed (Kessler 1969).
  ADD_ARG(std::vector<std::string>, form) = {};

  //! Kessler evaporation rate coefficient k3 [1/s] (form "kessler" only)
  ADD_ARG(std::vector<double>, k3) = {};
};
using EvaporationOptions = std::shared_ptr<EvaporationOptionsImpl>;

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

  //! reactions with two gaseous products; not a buffer, so dtype casts skip it
  torch::Tensor two_product_rxns_;

  //! Extended stoichiometries evaluated by the surface-equilibrium solver.
  torch::Tensor general_rxns_;

  //! reactions with the Kessler rate form and their k3 [1/s]; not buffers
  torch::Tensor kessler_rxns_, kessler_k3_;

  //! Constructor to initialize the layer
  EvaporationImpl() : options(EvaporationOptionsImpl::create()) {}
  explicit EvaporationImpl(EvaporationOptions const& options_);
  void reset() override;
  void pretty_print(std::ostream& os) const override;

  //! Compute the rate constant
  /*!
   * \param T temperature [K], shape (...)
   * \param P pressure [pa], shape (...)
   * \param C concentration [mol/m^3], shape (..., nspecies)
   * \param other stoich and optional gas_mask (bool, nspecies). Mixed
   *              gas/condensed reactants require gas_mask.
   * \return rate constant in (mol, m, s), shape (..., nreaction)
   */
  torch::Tensor forward(torch::Tensor T, torch::Tensor P, torch::Tensor C,
                        std::map<std::string, torch::Tensor> const& other);
};
TORCH_MODULE(Evaporation);

}  // namespace kintera

#undef ADD_ARG
