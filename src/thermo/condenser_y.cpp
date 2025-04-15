// torch
#include <torch/torch.h>

// kintera
#include <kintera/constants.h>

#include "condenser.hpp"

namespace kintera {

// x -> y at constant volume (mole concentration)
inline torch::Tensor satfunc1v(torch::Tensor& b, torch::Tensor& jac,
                               torch::Tensor conc, torch::Tensor s,
                               torch::Tensor logs_ddT, int j, int ix, int iy) {
  auto const& x = conc.select(-1, ix);
  auto const& y = conc.select(-1, iy);

  b.select(-1, j) = x - s;
  b.select(-1, j).clamp_(-y);

  jac.select(-2, j).select(-1, ix) = where(b.select(-1, j) > 0., 1., 0.);
  jac.select(-2, j).select(-1, iy) = where(b.select(-1, j) < 0., -1., 0.);

  return -s * logs_ddT;
}

CondenserYImpl::CondenserYImpl(const CondenserOptions& options_)
    : options(options_) {
  reset();
}

void CondenserYImpl::reset() {
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

torch::Tensor CondenserYImpl::forward(torch::Tensor temp, torch::Tensor pres,
                                      torch::Tensor conc,
                                      torch::Tensor intEng_RT,
                                      torch::Tensor cv_R,
                                      torch::optional<torch::Tensor> krate) {
  int nspecies = conc.size(-1);
  int nreact = options.react().size();
  auto vec = conc.sizes().vec();

  if (nreact == 0) {
    return torch::zeros_like(conc);
  }

  // (..., nreact)
  vec[temp.dim()] = nreact;
  auto b = torch::zeros(vec, temp.options());

  // (..., nreact, nspecies)
  vec.push_back(nspecies);
  auto jac = torch::zeros(vec, temp.options());
  auto rate_ddT = torch::zeros(vec, temp.options());

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
  vec = conc.sizes().vec();

  for (int j = 0; j < options.react().size(); ++j) {
    auto const& r = options.react()[j];

    auto svp_RT = r.eval_func(temp) / (constants::Rgas * temp);
    auto logsvp_ddT = r.eval_logf_ddT(temp);

    int ix = species_index(r.reaction().reactants().begin()->first);
    int iy = species_index(r.reaction().products().begin()->first);

    auto b_ddt = satfunc1v(b, jac, conc, svp_RT, logsvp_ddT, j, ix, iy);
    rate_ddT.select(-1, j) =
        intEng_RT * b_ddt.unsqueeze(-1) * temp.unsqueeze(-1);

    auto mask = (svp_RT < 0.).unsqueeze(-1).expand(vec);
    stoich_local.select(-1, j).masked_fill_(mask, 0.);
  }

  auto conc_dot_cv = conc.matmul(cv_R).unsqueeze(-1).unsqueeze(-1);
  jac -= (rate_ddT + cv_R * b.unsqueeze(-1)) / conc_dot_cv;
  // jac -= rate_ddT / (conc.matmul(cv_R)).unsqueeze(-1).unsqueeze(-1);
  // std::cout << "srv = " << srv << std::endl;

  auto A = jac.matmul(stoich_local);
  auto rates = -torch::linalg_solve(A, b, true);

  if (!krate.has_value()) {
    return stoich_local.matmul(rates.unsqueeze(-1)).squeeze(-1);
  }

  // accelerate convergence
  torch::Tensor result = stoich_local.matmul(rates.unsqueeze(-1)).squeeze(-1);
  // same sign
  auto mask = torch::sign((result * intEng_RT).sum(-1)) * krate.value() < 0;

  int iter = 0;
  for (; iter < options.max_iter(); ++iter) {
    auto krate_ = where(mask, krate.value().abs(), 1.).unsqueeze(-1);
    result = stoich_local.matmul((krate_ * rates).unsqueeze(-1)).squeeze(-1);

    // check any negative value
    auto neg = ((conc + result) < 0.).any(/*dim=*/-1);

    // same sign and has negative concentration
    krate.value() *= where(mask.logical_and(neg), 0.5, 1.);

    if (!neg.any().item<bool>()) break;
  }

  TORCH_CHECK(iter < options.max_iter(), "negative concentration");

  return result;
}

}  // namespace kintera
