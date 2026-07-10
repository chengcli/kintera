#include "equilibrium.hpp"

#include <ATen/TensorIterator.h>

#include <algorithm>
#include <cmath>
#include <kintera/reaction.hpp>
#include <kintera/utils/stoichiometry.hpp>

#include "equilibrium_dispatch.hpp"

namespace kintera {

void EquilibriumOptionsImpl::report(std::ostream &os) const {
  os << "-- equilibrium options --\n"
     << "* components = " << components().size() << "\n"
     << "* phases = " << phases().size() << "\n"
     << "* reactions = " << reactions().size() << "\n"
     << "* gas_phase = " << gas_phase() << "\n"
     << "* standard_pressure = " << standard_pressure() << " Pa\n"
     << "* max_iter = " << max_iter() << "\n"
     << "* ftol = " << ftol() << "\n";
}

void EquilibriumOptionsImpl::validate() const {
  TORCH_CHECK(!components().empty(), "Equilibrium requires components");
  TORCH_CHECK(!phases().empty(), "Equilibrium requires phases");
  TORCH_CHECK(phase_ids().size() == components().size(),
              "phase_ids must contain one entry per component");
  TORCH_CHECK(!reactions().empty(), "Equilibrium requires reactions");
  TORCH_CHECK(components().size() <= 64,
              "the active-set solver currently supports at most 64 components");
  for (int phase : phase_ids()) {
    TORCH_CHECK(phase >= 0 && phase < static_cast<int>(phases().size()),
                "phase id out of range: ", phase);
  }
  TORCH_CHECK(
      gas_phase() >= 0 && gas_phase() < static_cast<int>(phases().size()),
      "gas_phase is out of range");
  TORCH_CHECK(standard_pressure() > 0., "standard_pressure must be positive");
  TORCH_CHECK(max_iter() > 0, "max_iter must be positive");
  TORCH_CHECK(ftol() > 0., "ftol must be positive");
  TORCH_CHECK(mole_floor() >= 0., "mole_floor must be nonnegative");

  for (auto const &equation : reactions()) {
    Reaction reaction(equation);
    for (auto const &[name, coefficient] : reaction.reactants()) {
      (void)coefficient;
      TORCH_CHECK(std::find(components().begin(), components().end(), name) !=
                      components().end(),
                  "reaction references unknown component: ", name);
    }
    for (auto const &[name, coefficient] : reaction.products()) {
      (void)coefficient;
      TORCH_CHECK(std::find(components().begin(), components().end(), name) !=
                      components().end(),
                  "reaction references unknown component: ", name);
    }
  }
}

EquilibriumTPImpl::EquilibriumTPImpl(EquilibriumOptions const &options_)
    : options(options_) {
  reset();
}

void EquilibriumTPImpl::reset() {
  options->validate();
  std::vector<Reaction> reactions;
  reactions.reserve(options->reactions().size());
  for (auto const &equation : options->reactions())
    reactions.emplace_back(equation);
  stoich = register_buffer(
      "stoich", generate_stoichiometry_matrix(reactions, options->components())
                    .transpose(0, 1)
                    .contiguous()
                    .to(torch::kFloat64));
  auto rank = torch::linalg_matrix_rank(stoich).item<int64_t>();
  TORCH_CHECK(rank == stoich.size(1),
              "stoich reaction columns must be independent");
  phase_ids = register_buffer("phase_ids",
                              torch::tensor(options->phase_ids(), torch::kInt));
}

void EquilibriumTPImpl::pretty_print(std::ostream &os) const {
  os << "EquilibriumTP(components=" << options->components().size()
     << ", phases=" << options->phases().size() << ")";
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
EquilibriumTPImpl::forward(torch::Tensor temp, torch::Tensor pres,
                           torch::Tensor moles, torch::Tensor log_k,
                           bool warm_start) {
  (void)warm_start;
  TORCH_CHECK(moles.dim() >= 1, "moles must have a component dimension");
  TORCH_CHECK(log_k.dim() >= 1, "log_k must have a reaction dimension");
  int nspecies = stoich.size(0);
  int nreaction = stoich.size(1);
  TORCH_CHECK(moles.size(-1) == nspecies, "moles last dimension must be ",
              nspecies);
  TORCH_CHECK(log_k.size(-1) == nreaction, "log_k last dimension must be ",
              nreaction);
  TORCH_CHECK(moles.is_floating_point() && log_k.is_floating_point(),
              "moles and log_k must be floating-point tensors");
  TORCH_CHECK(moles.device() == log_k.device() &&
                  moles.device() == temp.device() &&
                  moles.device() == pres.device(),
              "all equilibrium inputs must be on the same device");
  TORCH_CHECK(moles.scalar_type() == log_k.scalar_type() &&
                  moles.scalar_type() == temp.scalar_type() &&
                  moles.scalar_type() == pres.scalar_type(),
              "all equilibrium inputs must have the same dtype");

  auto moles_scalar = moles.select(-1, 0);
  auto logk_scalar = log_k.select(-1, 0);
  auto broadcast =
      torch::broadcast_tensors({temp, pres, moles_scalar, logk_scalar});
  auto batch = broadcast[0].sizes().vec();
  auto component_shape = batch;
  component_shape.push_back(nspecies);
  auto reaction_shape = batch;
  reaction_shape.push_back(nreaction);

  auto out_moles = moles.expand(component_shape).contiguous().clone();
  auto logk_expanded = log_k.expand(reaction_shape).contiguous();
  auto gain_shape = batch;
  gain_shape.push_back(nreaction * nreaction);
  auto gain = torch::zeros(gain_shape, moles.options());
  auto diag_shape = batch;
  diag_shape.push_back(3);
  auto diag = torch::zeros(diag_shape, moles.options());

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(false)
                  .declare_static_shape(component_shape,
                                        {static_cast<int64_t>(batch.size())})
                  .add_output(gain)
                  .add_output(diag)
                  .add_output(out_moles)
                  .add_owned_input(broadcast[0].unsqueeze(-1))
                  .add_owned_input(broadcast[1].unsqueeze(-1))
                  .add_owned_input(moles.expand(component_shape))
                  .add_input(logk_expanded)
                  .build();

  auto s = stoich.to(moles.options()).contiguous();
  auto p = phase_ids.to(moles.device(), torch::kInt).contiguous();
  at::native::call_equilibrium(moles.device().type(), iter, s, p,
                               options->phases().size(), options->gas_phase(),
                               options->standard_pressure(), options->ftol(),
                               options->mole_floor(), options->max_iter());

  gain_shape.pop_back();
  gain_shape.push_back(nreaction);
  gain_shape.push_back(nreaction);
  return {out_moles, gain.view(gain_shape), diag};
}

}  // namespace kintera
