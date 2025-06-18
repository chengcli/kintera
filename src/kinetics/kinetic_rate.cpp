// kintera
#include "kinetic_rate.hpp"

namespace kintera {

KineticRateImpl::KineticRateImpl(const KineticRateOptions& options_)
    : options(options_) {
  reset();
}

void KineticRateImpl::reset() {
  auto reactions = options.reactions();
  auto species = options.species();

  // order = register_buffer("order",
  //     torch::zeros({nspecies, nreaction}), torch::kFloat64);
  stoich = register_buffer("stoich", torch::zeros({nspecies, nreaction}),
                           torch::kFloat64);

  for (int j = 0; j < reactions.size(); ++j) {
    for (int i = 0; i < species.size(); ++i) {
      auto it = reactions[j].reactants().find(species[i]);
      if (it != reactions[j].reactants().end()) {
        stoich[i][j] = -it->second;
      }
      it = reactions[j].products().find(species[i]);
      if (it != reactions[j].products().end()) {
        stoich[i][j] = it->second;
      }
    }
  }
}

torch::Tensor KineticRateImpl::forward(torch::Tensor conc,
                                       torch::Tensor log_rate_constant) {
  int nreaction = order.size(0);
  int nspecies = order.size(1);
  return (order.view({1, 1, nreaction, -1})
              .matmul(conc.unsqueeze(-1).log())
              .squeeze(-1) +
          log_rate_constant)
      .exp();
}

}  // namespace kintera
