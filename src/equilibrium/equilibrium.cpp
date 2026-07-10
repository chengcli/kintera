#include <ATen/TensorIterator.h>
#include <algorithm>
#include <cmath>

#include "equilibrium.hpp"
#include "equilibrium_dispatch.hpp"

namespace kintera {

void EquilibriumOptionsImpl::report(std::ostream &os) const {
  os << "-- equilibrium options --\n"
     << "* components = " << components().size() << "\n"
     << "* elements = " << elements().size() << "\n"
     << "* phases = " << phases().size() << "\n"
     << "* reactions = " << (stoich().empty() ? 0 : stoich()[0].size()) << "\n"
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
  TORCH_CHECK(stoich().size() == components().size(),
              "stoich must have one row per component");
  TORCH_CHECK(!stoich()[0].empty(), "Equilibrium requires reactions");
  if (!reactions().empty()) {
    TORCH_CHECK(reactions().size() == stoich()[0].size(),
                "reactions and stoich column counts must match");
  }
  if (!elements().empty()) {
    TORCH_CHECK(elements().size() == element_matrix().size(),
                "elements and element_matrix row counts must match");
  }
  TORCH_CHECK(components().size() <= 64,
              "the active-set solver currently supports at most 64 components");
  auto nreaction = stoich()[0].size();
  for (auto const &row : stoich()) {
    TORCH_CHECK(row.size() == nreaction, "stoich rows must have equal length");
  }

  // Independent reaction columns are required by the square Newton system.
  auto work = stoich();
  size_t rank = 0;
  for (size_t column = 0; column < nreaction && rank < work.size(); ++column) {
    size_t pivot = rank;
    for (size_t row = rank + 1; row < work.size(); ++row) {
      if (std::abs(work[row][column]) > std::abs(work[pivot][column]))
        pivot = row;
    }
    if (std::abs(work[pivot][column]) <= 1.e-12)
      continue;
    std::swap(work[rank], work[pivot]);
    for (size_t row = rank + 1; row < work.size(); ++row) {
      double factor = work[row][column] / work[rank][column];
      for (size_t j = column; j < nreaction; ++j) {
        work[row][j] -= factor * work[rank][j];
      }
    }
    ++rank;
  }
  TORCH_CHECK(rank == nreaction, "stoich reaction columns must be independent");
  for (int phase : phase_ids()) {
    TORCH_CHECK(phase >= 0 && phase < static_cast<int>(phases().size()),
                "phase id out of range: ", phase);
  }
  TORCH_CHECK(gas_phase() >= 0 &&
                  gas_phase() < static_cast<int>(phases().size()),
              "gas_phase is out of range");
  TORCH_CHECK(standard_pressure() > 0., "standard_pressure must be positive");
  TORCH_CHECK(max_iter() > 0, "max_iter must be positive");
  TORCH_CHECK(ftol() > 0., "ftol must be positive");
  TORCH_CHECK(mole_floor() >= 0., "mole_floor must be nonnegative");

  if (!element_matrix().empty()) {
    for (auto const &row : element_matrix()) {
      TORCH_CHECK(row.size() == components().size(),
                  "element_matrix rows must match component count");
      for (size_t j = 0; j < nreaction; ++j) {
        double balance = 0.;
        for (size_t i = 0; i < components().size(); ++i) {
          balance += row[i] * stoich()[i][j];
        }
        TORCH_CHECK(std::abs(balance) <= 1.e-10, "reaction ", j,
                    " does not conserve an element");
      }
    }
  }
}

EquilibriumImpl::EquilibriumImpl(EquilibriumOptions const &options_)
    : options(options_) {
  reset();
}

void EquilibriumImpl::reset() {
  options->validate();
  int nspecies = static_cast<int>(options->components().size());
  int nreaction = static_cast<int>(options->stoich()[0].size());

  std::vector<double> stoich_flat;
  stoich_flat.reserve(nspecies * nreaction);
  for (auto const &row : options->stoich()) {
    stoich_flat.insert(stoich_flat.end(), row.begin(), row.end());
  }
  stoich = register_buffer(
      "stoich",
      torch::tensor(stoich_flat, torch::kFloat64).view({nspecies, nreaction}));
  phase_ids = register_buffer("phase_ids",
                              torch::tensor(options->phase_ids(), torch::kInt));

  int nelement = static_cast<int>(options->element_matrix().size());
  std::vector<double> element_flat;
  if (nelement == 0) {
    // A zero row keeps the device kernel and diagnostic layout uniform.
    nelement = 1;
    element_flat.resize(nspecies, 0.);
  } else {
    element_flat.reserve(nelement * nspecies);
    for (auto const &row : options->element_matrix()) {
      element_flat.insert(element_flat.end(), row.begin(), row.end());
    }
  }
  element_matrix = register_buffer(
      "element_matrix",
      torch::tensor(element_flat, torch::kFloat64).view({nelement, nspecies}));
}

void EquilibriumImpl::pretty_print(std::ostream &os) const {
  os << "Equilibrium(components=" << options->components().size()
     << ", phases=" << options->phases().size() << ")";
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
EquilibriumImpl::forward(torch::Tensor temp, torch::Tensor pres,
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
  diag_shape.push_back(4);
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
  auto e = element_matrix.to(moles.options()).contiguous();
  at::native::call_equilibrium(moles.device().type(), iter, s, p, e,
                               options->phases().size(), options->gas_phase(),
                               options->standard_pressure(), options->ftol(),
                               options->mole_floor(), options->max_iter());

  gain_shape.pop_back();
  gain_shape.push_back(nreaction);
  gain_shape.push_back(nreaction);
  return {out_moles, gain.view(gain_shape), diag};
}

} // namespace kintera
