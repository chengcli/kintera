#pragma once

// C/C++
#include <map>
#include <memory>
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

//! Options for KINETICS-base (KB) falloff reactions.
/*!
 * Reproduces the validated Titan falloff form
 * (`kinetics_base/titan/physics.py::_pun_rate_constant`):
 *
 *   k_low  = A * (T/Tref)^b * exp(-Ea_R/T)        (low-pressure / 2-body limit)
 *   k_high = A'* (T/Tref)^b'* exp(-Ea_R'/T)       (high-pressure limit)
 *   ratio  = k_low * n / k_high                   (n = total number density)
 *   k      = (k_low / (1 + ratio)) * fc^(1/(1 + log10(ratio)^2))
 *
 * with `fc = 0.6` and `n` the total number density. This is an effective
 * *bimolecular* rate constant (the third body is folded into `ratio`), so it
 * differs from the standard Lindemann form `k0*[M]/(1+Pr)`. Built CGS-native:
 * the pre-exponential factors are stored raw in molecule,cm,s.
 */
struct KBFalloffOptionsImpl {
  static std::shared_ptr<KBFalloffOptionsImpl> create() {
    return std::make_shared<KBFalloffOptionsImpl>();
  }

  virtual std::string name() const { return "kb-falloff"; }
  virtual ~KBFalloffOptionsImpl() = default;

  std::shared_ptr<KBFalloffOptionsImpl> clone() const {
    return std::make_shared<KBFalloffOptionsImpl>(*this);
  }

  void report(std::ostream& os) const {
    os << "* reactions = " << fmt::format("{}", reactions()) << "\n"
       << "* Tref = " << Tref() << " K\n"
       << "* fc = " << fc() << "\n"
       << "* nreactions = " << reactions().size() << "\n";
  }

  ADD_ARG(double, Tref) = 1.0;
  ADD_ARG(std::string, units) = "molecule,cm,s";
  //! Troe center broadening factor (KB hardcodes 0.6)
  ADD_ARG(double, fc) = 0.6;
  ADD_ARG(std::vector<Reaction>, reactions) = {};

  //! Low-pressure (2-body) Arrhenius: k_low = A*(T/Tref)^b*exp(-Ea_R/T)
  ADD_ARG(std::vector<double>, k0_A) = {};
  ADD_ARG(std::vector<double>, k0_b) = {};
  ADD_ARG(std::vector<double>, k0_Ea_R) = {};

  //! High-pressure Arrhenius: k_high = A*(T/Tref)^b*exp(-Ea_R/T)
  ADD_ARG(std::vector<double>, kinf_A) = {};
  ADD_ARG(std::vector<double>, kinf_b) = {};
  ADD_ARG(std::vector<double>, kinf_Ea_R) = {};
};
using KBFalloffOptions = std::shared_ptr<KBFalloffOptionsImpl>;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set, KBFalloffOptions op);

class KBFalloffImpl : public torch::nn::Cloneable<KBFalloffImpl> {
 public:
  //! Low-pressure Arrhenius parameters, shape (nreaction,)
  torch::Tensor k0_A;
  torch::Tensor k0_b;
  torch::Tensor k0_Ea_R;

  //! High-pressure Arrhenius parameters, shape (nreaction,)
  torch::Tensor kinf_A;
  torch::Tensor kinf_b;
  torch::Tensor kinf_Ea_R;

  //! options with which this `KBFalloffImpl` was constructed
  KBFalloffOptions options;

  KBFalloffImpl() : options(KBFalloffOptionsImpl::create()) {}
  explicit KBFalloffImpl(KBFalloffOptions const& options_);
  void reset() override;
  void pretty_print(std::ostream& os) const override;

  //! Compute the effective (bimolecular) rate constant.
  /*!
   * \param T temperature [K], shape (...)
   * \param P pressure, shape (...)
   * \param C concentration, shape (..., nspecies) or (..., nspecies, nreaction)
   * \param other optional inputs; if `other["number_density"]` is present it is
   *        used as the total number density `n` (shape broadcastable to (...)),
   *        otherwise `n` is the sum over species of `C`.
   * \return rate constant, shape (..., nreaction)
   */
  torch::Tensor forward(torch::Tensor T, torch::Tensor P, torch::Tensor C,
                        std::map<std::string, torch::Tensor> const& other);

 private:
  torch::Tensor compute_arrhenius(torch::Tensor T, torch::Tensor A,
                                  torch::Tensor b, torch::Tensor Ea_R) const;
};
TORCH_MODULE(KBFalloff);

}  // namespace kintera

#undef ADD_ARG
