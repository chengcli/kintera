// torch
#include <torch/torch.h>

// kintera
#include <kintera/constants.h>

#include "condenser.hpp"

namespace kintera {

// x -> y at constant pressure (mole fraction)
inline void satfunc1p(torch::Tensor& b, torch::Tensor& jac, torch::Tensor xfrac,
                      torch::Tensor s, torch::Tensor xg, int j, int ix,
                      int iy) {
  auto const& x = xfrac.select(-1, ix);
  auto const& y = xfrac.select(-1, iy);

  b.select(-1, j) = torch::where(s > 1., -y, (x - s * xg) / (1. - s));
  b.select(-1, j).clamp_(-y);

  jac.select(-2, j).select(-1, ix) = where(b.select(-1, j) > 0., 1., 0.);
  jac.select(-2, j).select(-1, iy) = where(b.select(-1, j) < 0., -1., 0.);
}

CondenserXImpl::CondenserXImpl(const CondenserOptions& options_)
    : options(options_) {
  reset();
}

void CondenserXImpl::reset() {
  int nspecies = options.species().size();
  int nreact = options.react().size();

  stoich = register_buffer("stoich",
                           torch::zeros({nspecies, nreact}, torch::kFloat64));

  // populate stoichiometry matrix
  for (int j = 0; j < options.react().size(); ++j) {
    auto const& r = options.react()[j];
    for (int i = 0; i < options.species().size(); ++i) {
      auto it = r.reaction().reactants().find(options.species()[i]);
      if (it != r.reaction().reactants().end()) {
        stoich[i][j] = -it->second;
      }
      it = r.reaction().products().find(options.species()[i]);
      if (it != r.reaction().products().end()) {
        stoich[i][j] = it->second;
      }
    }
  }
}

torch::Tensor CondenserXImpl::forward(torch::Tensor temp, torch::Tensor pres,
                                      torch::Tensor xfrac) {
  int nspecies = xfrac.size(-1);
  int nreact = options.react().size();
  auto vec = xfrac.sizes().vec();

  if (nreact == 0) {
    return torch::zeros_like(xfrac);
  }

  // (..., nreact)
  vec[temp.dim()] = nreact;
  auto b = torch::zeros(vec, temp.options());

  // (..., nreact, nspecies)
  vec.push_back(nspecies);
  auto jac = torch::zeros(vec, temp.options());

  // (..., nspecies, nreact)
  vec[temp.dim()] = nspecies;
  vec[temp.dim() + 1] = nreact;
  auto stoich_local = torch::zeros(vec, temp.options());

  // (1..., nspecies, nreact)
  for (int i = 0; i < vec.size() - 2; ++i) {
    vec[i] = 1;
  }
  stoich_local.copy_(stoich.view(vec));

  // (..., nspecies)
  vec = xfrac.sizes().vec();

  auto dens = pres / (constants::Rgas * temp);
  auto xgas = xfrac.narrow(-1, 0, options.ngas()).sum(-1);

  for (int j = 0; j < options.react().size(); ++j) {
    auto const& r = options.react()[j];

    auto svp_RT = r.eval_func(temp) / (constants::Rgas * temp);

    int ix = species_index(r.reaction().reactants().begin()->first);
    int iy = species_index(r.reaction().products().begin()->first);

    satfunc1p(b, jac, xfrac, svp_RT / dens, xgas, j, ix, iy);
    auto mask = (svp_RT < 0.).unsqueeze(-1).expand(vec);
    stoich_local.select(-1, j).masked_fill_(mask, 0.);
  }

  auto A = jac.matmul(stoich_local);
  auto rates = -torch::linalg_solve(A, b, true);

  return stoich_local.matmul(rates.unsqueeze(-1)).squeeze(-1);
}

}  // namespace kintera
