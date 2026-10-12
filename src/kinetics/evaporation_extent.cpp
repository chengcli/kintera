#include "evaporation_extent.hpp"

#include <cmath>
#include <limits>

namespace kintera {

torch::Tensor EvaporationExtent::forward(torch::autograd::AutogradContext* ctx,
                                         torch::Tensor conc, torch::Tensor nu,
                                         torch::Tensor logkc) {
  const bool single = conc.scalar_type() == torch::kFloat32;
  const double tiny = single ? std::numeric_limits<float>::denorm_min()
                             : std::numeric_limits<double>::denorm_min();
  const double eps = single ? std::numeric_limits<float>::epsilon()
                            : std::numeric_limits<double>::epsilon();
  const double maxval = single ? std::numeric_limits<float>::max()
                               : std::numeric_limits<double>::max();
  TORCH_CHECK(torch::isfinite(conc).all().item<bool>() &&
                  conc.ge(0).all().item<bool>() &&
                  torch::isfinite(logkc).all().item<bool>(),
              "General evaporation requires finite nonnegative concentrations "
              "and a finite log saturation quotient");

  auto product = nu.gt(0);
  auto reactant = nu.lt(0);
  auto participating = nu.ne(0);
  auto safe_c =
      torch::where(participating, conc, torch::ones_like(conc)).clamp_min(tiny);
  auto f0 = (nu * safe_c.log()).sum(-2) - logkc;
  auto missing_product = (product & conc.eq(0)).any(-2);
  auto missing_reactant = (reactant & conc.eq(0)).any(-2);
  auto active = (~missing_reactant) & (missing_product | f0.lt(0));

  // Product-only bound: c_i + nu_i*x >= nu_i*x, while consumed gas
  // concentrations cannot exceed their ambient values. This also scales very
  // small roots without spending hundreds of iterations bisecting the H2 bound.
  auto pos = nu.clamp_min(0);
  auto log_upper =
      (logkc - (nu.clamp_max(0) * safe_c.log()).sum(-2) -
       (pos * torch::where(product, pos, torch::ones_like(pos)).log()).sum(0)) /
      pos.sum(0);
  auto upper = log_upper.clamp_max(std::log(maxval) - 1).exp();
  // An abundant product can put the root many orders below the empty-air
  // bound. Bound each product in turn, holding every other product at its
  // ambient value. Ignore a candidate if any of those other products is zero.
  auto product_logs = pos * safe_c.log();
  auto log_surface_bound =
      (logkc.unsqueeze(-2) - (nu.clamp_max(0) * safe_c.log()).sum(-2, true) -
       product_logs.sum(-2, true) + product_logs) /
      torch::where(product, pos, torch::ones_like(pos));
  // Pad outward for cancellation in sums of logs, particularly when an
  // ambient product is extremely large and another is zero.
  log_surface_bound += 16 * eps * (1 + log_surface_bound.abs());
  auto candidate =
      (log_surface_bound.clamp_max(std::log(maxval) - 1).exp() - conc)
          .clamp_min(0) /
      torch::where(product, pos, torch::ones_like(pos));
  auto zero_products = (product & conc.eq(0));
  auto other_zeros =
      zero_products.sum(-2, true) - zero_products.to(torch::kLong);
  candidate = torch::where(product & other_zeros.eq(0) & candidate.gt(0),
                           candidate, torch::full_like(candidate, maxval));
  upper = torch::minimum(upper, std::get<0>(candidate.min(-2)));
  auto gas_bound =
      std::get<0>(torch::where(reactant, conc / (-nu).clamp_min(tiny),
                               torch::full_like(conc, maxval))
                      .min(-2));
  upper = torch::minimum(upper, gas_bound);
  active = active & upper.gt(0);
  // Give inactive columns a harmless, finite dummy problem.
  auto c = torch::where(active.unsqueeze(-2), conc, torch::ones_like(conc));
  auto target = torch::where(active, logkc, torch::zeros_like(logkc));
  upper = torch::where(active, upper, torch::ones_like(upper));
  auto residual = [&](torch::Tensor x) {
    auto surface = torch::where(participating, c + nu * x.unsqueeze(-2),
                                torch::ones_like(c))
                       .clamp_min(tiny);
    return (nu * surface.log()).sum(-2) - target;
  };
  TORCH_CHECK((~active | residual(upper).ge(-32 * eps) | upper.eq(gas_bound))
                  .all()
                  .item<bool>(),
              "General evaporation surface extent exceeds the dtype range");
  auto lo = torch::zeros_like(upper);
  auto hi = upper.clone();
  auto x = 0.5 * upper;
  auto done = ~active;
  for (int iteration = 0; iteration < 128; ++iteration) {
    auto f = residual(x);
    hi = torch::where((~done) & f.ge(0), x, hi);
    lo = torch::where((~done) & f.lt(0), x, lo);
    // A narrow extent bracket is also convergence when equilibrium lies less
    // than an ulp from complete depletion of a gaseous reactant.
    done = done | f.abs().le(32 * eps) |
           (hi - lo).le(8 * eps * x.abs().clamp_min(tiny));
    if (done.all().item<bool>()) break;
    auto surface = torch::where(participating, c + nu * x.unsqueeze(-2),
                                torch::ones_like(c))
                       .clamp_min(tiny);
    auto slope = (nu.square() / surface).sum(-2);
    auto next = x - f / slope;
    auto bisect = (lo + (hi - lo) * 0.5);
    // Safeguarding away from bracket endpoints ensures continued contraction.
    auto width = hi - lo;
    next = torch::where(torch::isfinite(next) & next.gt(lo + .05 * width) &
                            next.lt(hi - .05 * width),
                        next, bisect);
    x = torch::where(done, x, next);
  }
  TORCH_CHECK(done.all().item<bool>(),
              "General evaporation surface solve did not converge");
  x = torch::where(active, x, torch::zeros_like(x));

  // Scale reciprocal concentrations before summing F_x to avoid overflow for
  // trace gases. dx/dc_i = -(nu_i/c_i,s)/F_x; dx/dlogK = 1/F_x.
  auto surface = torch::where(participating, c + nu * x.unsqueeze(-2),
                              torch::full_like(c, maxval))
                     .clamp_min(tiny);
  auto smallest = std::get<0>(surface.min(-2));
  auto ratio = smallest.unsqueeze(-2) / surface;
  auto denom = (nu.square() * ratio).sum(-2);
  auto dc = -nu * ratio / denom.unsqueeze(-2);
  auto dk = smallest / denom;
  dc = torch::where(active.unsqueeze(-2), dc, torch::zeros_like(dc));
  dk = torch::where(active, dk, torch::zeros_like(dk));
  ctx->save_for_backward({dc, dk});
  return x;
}

std::vector<torch::Tensor> EvaporationExtent::backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<torch::Tensor> grad_outputs) {
  auto saved = ctx->get_saved_variables();
  return {saved[0] * grad_outputs[0].unsqueeze(-2), torch::Tensor(),
          saved[1] * grad_outputs[0]};
}

}  // namespace kintera
