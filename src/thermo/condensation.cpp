// yaml
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/constants.h>

#include "condensation.hpp"
#include "thermo_formatter.hpp"

namespace kintera {

CondensationOptions CondensationOptions::from_yaml(std::string const& filename) {
  CondensationOptions options;
  YAML::Node config = YAML::LoadFile(filename);

  return options;
}

// x -> y at constant volume (mole concentration)
inline torch::Tensor satfunc1v(torch::Tensor& b, torch::Tensor& jac,
                               torch::Tensor conc, torch::Tensor s,
                               torch::Tensor logs_ddT, int j, int ix, int iy) {
  auto const& x = conc.select(3, ix);
  auto const& y = conc.select(3, iy);

  b.select(3, j) = x - s;
  b.select(3, j).clamp_(-y);

  jac.select(3, j).select(3, ix) = where(b.select(3, j) > 0., 1., 0.);
  jac.select(3, j).select(3, iy) = where(b.select(3, j) < 0., -1., 0.);

  return -s * logs_ddT;
}

// x -> y at constant pressure (mole fraction)
inline void satfunc1p(torch::Tensor& b, torch::Tensor& jac, torch::Tensor xfrac,
                      torch::Tensor s, torch::Tensor xg, int j, int ix,
                      int iy) {
  auto const& x = xfrac.select(3, ix);
  auto const& y = xfrac.select(3, iy);

  b.select(3, j) = torch::where(s > 1., -y, (x - s * xg) / (1. - s));
  b.select(3, j).clamp_(-y);

  jac.select(3, j).select(3, ix) = where(b.select(3, j) > 0., 1., 0.);
  jac.select(3, j).select(3, iy) = where(b.select(3, j) < 0., -1., 0.);
}

CondensationImpl::CondensationImpl(const CondensationOptions& options_)
    : options(options_) {
  reset();
}

void CondensationImpl::reset() {
  int nspecies = options.species().size();
  int nreact = options.react().size();

  stoich = torch::zeros({nspecies, nreact}, torch::kFloat64);

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

torch::Tensor CondensationImpl::forward(torch::Tensor temp, torch::Tensor pres,
                                        torch::Tensor conc,
                                        torch::Tensor intEng_RT,
                                        torch::Tensor cv_R,
                                        torch::optional<torch::Tensor> krate) {
  int nc3 = conc.size(0);
  int nc2 = conc.size(1);
  int nc1 = conc.size(2);
  int nspecies = conc.size(3);
  int nreact = options.react().size();

  if (nreact == 0) {
    return torch::zeros({nc3, nc2, nc1, nspecies}, temp.options());
  }

  torch::Tensor b = torch::zeros({nc3, nc2, nc1, nreact}, temp.options());
  torch::Tensor jac =
      torch::zeros({nc3, nc2, nc1, nreact, nspecies}, temp.options());
  torch::Tensor rate_ddT =
      torch::zeros({nc3, nc2, nc1, nreact, nspecies}, temp.options());
  torch::Tensor stoich_local =
      torch::zeros({nc3, nc2, nc1, nspecies, nreact}, temp.options());

  stoich_local.copy_(stoich.view({1, 1, 1, nspecies, nreact}));

  for (int j = 0; j < options.react().size(); ++j) {
    auto const& r = options.react()[j];

    auto svp_RT = r.eval_func(temp) / (constants::Rgas * temp);
    auto logsvp_ddT = r.eval_logf_ddT(temp);

    int ix = species_index(r.reaction().reactants().begin()->first);
    int iy = species_index(r.reaction().products().begin()->first);

    auto b_ddt = satfunc1v(b, jac, conc, svp_RT, logsvp_ddT, j, ix, iy);
    rate_ddT.select(3, j) =
        intEng_RT * b_ddt.unsqueeze(-1) * temp.unsqueeze(-1);

    auto mask = (svp_RT < 0.).unsqueeze(-1).expand({nc3, nc2, nc1, nspecies});
    stoich_local.select(4, j).masked_fill_(mask, 0.);
  }

  auto conc_dot_cv = conc.matmul(cv_R).unsqueeze(-1).unsqueeze(-1);
  jac -= (rate_ddT + cv_R * b.unsqueeze(3)) / conc_dot_cv;
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

torch::Tensor CondensationImpl::equilibrate_tp(torch::Tensor temp,
                                               torch::Tensor pres,
                                               torch::Tensor xfrac,
                                               int ngas) const {
  int nc3 = xfrac.size(0);
  int nc2 = xfrac.size(1);
  int nc1 = xfrac.size(2);
  int nspecies = xfrac.size(3);
  int nreact = options.react().size();

  if (nreact == 0) {
    return torch::zeros({nc3, nc2, nc1, nspecies}, temp.options());
  }

  torch::Tensor b = torch::zeros({nc3, nc2, nc1, nreact}, temp.options());
  torch::Tensor jac =
      torch::zeros({nc3, nc2, nc1, nreact, nspecies}, temp.options());
  torch::Tensor stoich_local =
      torch::zeros({nc3, nc2, nc1, nspecies, nreact}, temp.options());

  stoich_local.copy_(stoich.view({1, 1, 1, nspecies, nreact}));

  auto dens = pres / (constants::Rgas * temp);
  auto xgas = xfrac.narrow(-1, 0, ngas).sum(-1);

  for (int j = 0; j < options.react().size(); ++j) {
    auto const& r = options.react()[j];

    auto svp_RT = r.eval_func(temp) / (constants::Rgas * temp);

    int ix = species_index(r.reaction().reactants().begin()->first);
    int iy = species_index(r.reaction().products().begin()->first);

    satfunc1p(b, jac, xfrac, svp_RT / dens, xgas, j, ix, iy);
    auto mask = (svp_RT < 0.).unsqueeze(-1).expand({nc3, nc2, nc1, nspecies});
    stoich_local.select(4, j).masked_fill_(mask, 0.);
  }

  auto A = jac.matmul(stoich_local);
  auto rates = -torch::linalg_solve(A, b, true);

  return stoich_local.matmul(rates.unsqueeze(-1)).squeeze(-1);
}

}  // namespace kintera
