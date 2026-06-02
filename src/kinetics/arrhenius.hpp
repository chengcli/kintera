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

// arg
#include <kintera/add_arg.h>

namespace YAML {
class Node;
}

namespace kintera {

//! Options to initialize all reaction rate constants
struct ArrheniusOptionsImpl {
  static std::shared_ptr<ArrheniusOptionsImpl> create() {
    return std::make_shared<ArrheniusOptionsImpl>();
  }
  static std::shared_ptr<ArrheniusOptionsImpl> from_yaml(
      const YAML::Node& node,
      std::shared_ptr<ArrheniusOptionsImpl> derived_type_ptr = nullptr);

  virtual std::string name() const { return "arrhenius"; }
  virtual ~ArrheniusOptionsImpl() = default;

  std::shared_ptr<ArrheniusOptionsImpl> clone() const {
    return std::make_shared<ArrheniusOptionsImpl>(*this);
  }
  void report(std::ostream& os) const {
    os << "* reactions = " << fmt::format("{}", reactions()) << "\n"
       << "* Tref = " << Tref() << " K\n"
       << "* units = " << units() << "\n"
       << "* A = " << fmt::format("{}", A()) << "\n"
       << "* b = " << fmt::format("{}", b()) << "\n"
       << "* Ea_R = " << fmt::format("{}", Ea_R()) << " K\n"
       << "* E4_R = " << fmt::format("{}", E4_R()) << "\n";
    if (!A_ranges().empty()) {
      os << "* A_ranges = " << fmt::format("{}", A_ranges()) << "\n"
         << "* b_ranges = " << fmt::format("{}", b_ranges()) << "\n"
         << "* Ea_R_ranges = " << fmt::format("{}", Ea_R_ranges()) << "\n"
         << "* T_ranges = " << fmt::format("{}", T_ranges()) << "\n";
    }
  }

  // reference temperature
  ADD_ARG(double, Tref) = 300.0;

  // units
  ADD_ARG(std::string, units) = "molecule,cm,s";

  //! reactions
  ADD_ARG(std::vector<Reaction>, reactions) = {};

  //! Pre-exponential factor.
  //! actual units depend on the reaction order
  ADD_ARG(std::vector<double>, A) = {};

  //! Dimensionless temperature exponent
  ADD_ARG(std::vector<double>, b) = {};

  //! Activation energy in K
  ADD_ARG(std::vector<double>, Ea_R) = {};

  //! Additional 4th parameter in the rate expression
  ADD_ARG(std::vector<double>, E4_R) = {};

  //! Multi-range parameters, indexed [reaction][range].
  /*!
   * When `A_ranges` is non-empty it takes precedence over the single-range
   * (`A`, `b`, `Ea_R`) lists and defines the reaction count. Each reaction
   * carries one or more temperature ranges, each with its own (A, b, Ea_R)
   * triple. This represents KINETICS-base (KB) AK/AK2/AK3 constants. The rate
   * within a range is the usual `A * (T / Tref)^b * exp(-Ea_R / T)`; KB's
   * ZK1 (B>0) and ZK2 (B<0) forms are the same expression with the sign of
   * `b` carried through, so both map onto this option directly.
   */
  ADD_ARG(std::vector<std::vector<double>>, A_ranges) = {};

  //! Per-range temperature exponent, indexed [reaction][range].
  ADD_ARG(std::vector<std::vector<double>>, b_ranges) = {};

  //! Per-range activation energy [K], indexed [reaction][range].
  ADD_ARG(std::vector<std::vector<double>>, Ea_R_ranges) = {};

  //! Per-range upper temperature bound [K], indexed [reaction][range].
  /*!
   * One entry per range, ascending. Range `r` is active for
   * `[T_ranges[r-1], T_ranges[r])`; the first range extends down to 0 K and
   * the last range extends up to +inf (its stated bound is ignored), so the
   * ranges always cover all temperatures with no gaps.
   */
  ADD_ARG(std::vector<std::vector<double>>, T_ranges) = {};
};
using ArrheniusOptions = std::shared_ptr<ArrheniusOptionsImpl>;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set, ArrheniusOptions op);

class ArrheniusImpl : public torch::nn::Cloneable<ArrheniusImpl> {
 public:
  //! pre-exponential factor [molecule, cm, s], shape (nreaction,)
  torch::Tensor A;

  //! temperature exponent, shape (nreaction,)
  torch::Tensor b;

  //! activation energy [K], shape (nreaction,)
  torch::Tensor Ea_R;

  //! additional 4th parameter in the rate expression, shape (nreaction,)
  torch::Tensor E4_R;

  //! per-range pre-exponential factor, shape (nreaction, nrange)
  torch::Tensor Amr;

  //! per-range temperature exponent, shape (nreaction, nrange)
  torch::Tensor bmr;

  //! per-range activation energy [K], shape (nreaction, nrange)
  torch::Tensor Ea_Rmr;

  //! per-range lower temperature bound [K], shape (nreaction, nrange)
  torch::Tensor Tlo;

  //! per-range upper temperature bound [K], shape (nreaction, nrange)
  torch::Tensor Thi;

  //! number of temperature ranges (>= 1)
  int nrange = 1;

  //! options with which this `ArrheniusImpl` was constructed
  ArrheniusOptions options;

  //! Constructor to initialize the layer
  ArrheniusImpl() : options(ArrheniusOptionsImpl::create()) {}
  explicit ArrheniusImpl(ArrheniusOptions const& options_);
  void reset() override;
  void pretty_print(std::ostream& os) const override;

  //! Compute the rate constant
  /*!
   * \param T temperature [K], shape (...)
   * \param P pressure [pa], shape (...)
   * \param C concentration [mol/m^3], shape (..., nspecies)
   * \param other additional parameters
   * \return reaction rate constant in (mol, m, s), (..., nreaction)
   */
  torch::Tensor forward(torch::Tensor T, torch::Tensor P, torch::Tensor C,
                        std::map<std::string, torch::Tensor> const& other);
};
TORCH_MODULE(Arrhenius);

}  // namespace kintera

#undef ADD_ARG
