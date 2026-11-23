// kintera
#include <kintera/constants.h>

#include <kintera/utils/check_resize.hpp>

#include "eval_uhs.hpp"
#include "h2_rotation.h"
#include "log_svp.hpp"
#include "thermo.hpp"
#include "thermo_dispatch.hpp"
#include "thermo_formatter.hpp"

namespace kintera {

std::shared_ptr<ThermoYImpl> ThermoYImpl::create(ThermoOptions const& opts,
                                                 torch::nn::Module* p,
                                                 std::string const& name) {
  TORCH_CHECK(p, "[ThermoY] Parent module is null");
  TORCH_CHECK(opts, "[ThermoY] Options pointer is null");
  return p->register_module(name, ThermoY(opts));
}

ThermoYImpl::ThermoYImpl(const ThermoOptions& options_) : options(options_) {
  populate_thermo(options);
  reset();
}

ThermoYImpl::ThermoYImpl(const ThermoOptions& options1,
                         const SpeciesThermo& options2)
    : options(options1) {
  populate_thermo(options1);
  populate_thermo(options2);
  auto merged = merge_thermo(options1, options2);
  static_cast<SpeciesThermoImpl&>(*options) = (*merged);
  reset();
}

void ThermoYImpl::reset() {
  thermo_check_options(options, "ThermoY");

  auto const& mu_vec = options->mu();
  inv_mu =
      register_buffer("inv_mu", 1. / torch::tensor(mu_vec, torch::kFloat64));

  if (options->verbose()) {
    std::cout << "[ThermoY] species molecular weights (kg/mol): "
              << fmt::format("{}", mu_vec) << std::endl;
  }

  thermo_zero_offsets(options, "ThermoY");

  auto cv_R = torch::tensor(options->cref_R(), torch::kFloat64);
  auto uref_R = torch::tensor(options->uref_R(), torch::kFloat64);

  // J/kg/K
  cv0 = register_buffer("cv0", cv_R * constants::Rgas * inv_mu);

  // J/kg
  u0 = register_buffer("u0", uref_R * constants::Rgas * inv_mu);

  auto nspecies = options->species().size();
  auto reactions = options->reactions();
  stoich = register_buffer("stoich", thermo_stoich(options));

  uv_partitionable = !reactions.empty();
  std::vector<bool> used_species(nspecies, false);
  auto stoich_values = stoich.accessor<double, 2>();
  for (int reaction_index = 0; reaction_index < reactions.size();
       ++reaction_index) {
    int vapor_count = 0;
    int cloud_count = 0;
    for (int species_index = 0; species_index < nspecies; ++species_index) {
      double coefficient = stoich_values[species_index][reaction_index];
      if (coefficient == 0.) continue;
      if (!std::isfinite(coefficient) || used_species[species_index]) {
        uv_partitionable = false;
      }
      used_species[species_index] = true;
      if (species_index < options->vapor_ids().size() && coefficient < 0.) {
        ++vapor_count;
      } else if (species_index >= options->vapor_ids().size() &&
                 coefficient > 0.) {
        ++cloud_count;
      } else {
        uv_partitionable = false;
      }
    }
    if (vapor_count != 1 || cloud_count != 1) uv_partitionable = false;
  }

  if (options->verbose()) {
    std::cout << "[ThermoY] stoichiometry matrix: " << std::endl;
    std::cout << stoich << std::endl;
  }

  auto none = [](std::vector<std::string> const& v) {
    return std::all_of(v.begin(), v.end(),
                       [](auto const& s) { return s.empty() || s == "null"; });
  };
  bool cv_const = !options->use_nasa9_cp() && !options->use_h2_cp();
  temp_plan = cv_const && none(options->intEng_R_extra()) ? SolvePlan::exact
                                                          : SolvePlan::newton;
  pres_plan = none(options->czh()) ? SolvePlan::exact : SolvePlan::newton;

  // clone() calls reset(); drop any count copied from the original.
  failures_ = torch::Tensor();
  tables_ = nullptr;
}

torch::Tensor const& ThermoYImpl::_counter(torch::Device device) {
  if (failures_.defined() && failures_.device() == device) return failures_;
  TORCH_CHECK(!capture_active(device), "[ThermoY] prepare() before capture");
  c10::InferenceMode off(false);  // a normal tensor: add_ legal in both modes
  return failures_ = torch::zeros({3}, at::device(device).dtype(at::kLong));
}

void ThermoYImpl::reset_failures() {
  _counter(failures_.defined() ? failures_.device() : inv_mu.device()).zero_();
}

void ThermoYImpl::consume_failures(torch::Tensor out) {
  auto f = _counter(failures_.defined() ? failures_.device() : out.device());
  out.copy_(f[0] > 0), f[0].zero_();
}

ThermoTablesPtr ThermoYImpl::prepare(torch::Device device, torch::Dtype dtype,
                                     torch::IntArrayRef shape) {
  _counter(device), _active_set(device, shape, /*cold=*/false);
  if (tables_ && tables_->cref_R.device() == device &&
      tables_->cref_R.scalar_type() == dtype)
    return NoTableBuild::record(tables_), tables_;  // used in the scope
  TORCH_CHECK(!capture_active(device), "[ThermoY] prepare() before capture");
  auto like = torch::empty({0}, torch::dtype(dtype).device(device));
  return tables_ = tables_for(options, like);
}

void ThermoYImpl::_active_set(torch::Device device, torch::IntArrayRef shape,
                              bool cold) {
  auto vec = shape.vec();
  vec.push_back(options->reactions().size());
  bool fits = reaction_set.defined() && reaction_set.device() == device &&
              reaction_set.sizes().equals(vec) && nactive.sizes().equals(shape);
  if (fits && !cold) return;
  TORCH_CHECK(fits || !capture_active(device), "[ThermoY] prepare() first");
  c10::InferenceMode off(false);  // normal tensors: written in place later
  auto opts = torch::dtype(torch::kInt).device(device);
  auto cold_set = torch::arange(0, (int)vec.back(), opts).expand(vec);
  reaction_set = fits ? reaction_set.copy_(cold_set) : cold_set.contiguous();
  nactive = fits ? nactive.zero_() : torch::zeros(shape, opts);
}

ActiveSet ThermoYImpl::snapshot_active_set(ActiveSet out) {
  _active_set(out.second.device(), out.second.sizes(), /*cold=*/false);
  return out.first.copy_(reaction_set), out.second.copy_(nactive), out;
}

void ThermoYImpl::restore_active_set(ActiveSet const& in) {
  reaction_set.copy_(in.first), nactive.copy_(in.second);
}

ActiveSet ThermoYImpl::make_active_set_storage() const {
  TORCH_CHECK(nactive.defined(), "[ThermoY] prepare() first");
  return {torch::empty_like(reaction_set), torch::empty_like(nactive)};
}

void ThermoYImpl::pretty_print(std::ostream& os) const {
  os << fmt::format("ThermoY({})", options) << std::endl;
}

torch::Tensor ThermoYImpl::compute(std::string ab,
                                   std::vector<torch::Tensor> const& args) {
  if (ab == "TUS->F") return args[1] - args[0] * args[2];  // T, U, S
  auto const& a = args.back();
  auto vec = a.sizes().vec();
  if (ab == "V->Y") {  // (..., 1 + ny) -> (ny, ...)
    vec.pop_back(), vec.insert(vec.begin(), a.size(-1) - 1);
  } else if (ab == "Y->X" || ab == "DY->V") {  // (ny, ...) -> (..., 1 + ny)
    vec.erase(vec.begin()), vec.push_back(a.size(0) + 1);
  } else {  // main's set_: the broadcast cell shape (V's last dim: species)
    vec.clear();
    for (size_t i = 0; i < args.size(); ++i)
      vec = at::infer_size(
          vec, args[i].sizes().slice(0, args[i].dim() - (ab[i] == 'V')));
  }
  auto out = torch::empty(vec, a.options());
  return compute_into(ab, args, out), out;
}

void ThermoYImpl::compute_into(std::string ab,
                               std::vector<torch::Tensor> const& args,
                               torch::Tensor out) {
  if (ab == "V->Y" || ab == "Y->X" || ab == "DY->V") {
    auto ny = ab == "V->Y" ? args[0].size(-1) - 1 : args.back().size(0);
    TORCH_CHECK(
        ny + 1 == options->vapor_ids().size() + options->cloud_ids().size(),
        "mass fraction size mismatch");
  }
  if (ab == "VU->T" || ab == "PV->T") _counter(args[0].device());

  if (ab == "V->Y") {
    _ivol_to_yfrac(args[0], out);
  } else if (ab == "Y->X") {
    _yfrac_to_xfrac(args[0], out);
  } else if (ab == "DY->V") {
    _yfrac_to_ivol(args[0], args[1], out);
  } else if (ab == "PV->T") {
    _pres_to_temp(args[0], args[1], out);
  } else if (ab == "VT->cv") {
    _cv_vol(args[0], args[1], out);
  } else if (ab == "VT->U") {
    _intEng_vol(args[0], args[1], out);
  } else if (ab == "VU->T") {
    _intEng_to_temp(args[0], args[1], out);
  } else if (ab == "VT->P") {
    _temp_to_pres(args[0], args[1], out);
  } else if (ab == "PVT->S") {
    _entropy_vol(args[0], args[1], args[2], out);
  } else if (ab == "TUS->F") {
    out.copy_(args[1] - args[0] * args[2]);
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

torch::Tensor ThermoYImpl::forward(torch::Tensor rho, torch::Tensor intEng,
                                   torch::Tensor const& yfrac, bool warm_start,
                                   torch::optional<torch::Tensor> diag) {
  if (options->reactions().size() == 0) {  // no-op
    return torch::Tensor();
  }

  // the module's table (eager code prepares it); a stale one throws (debug)
  auto t = prepare(rho.device(), rho.scalar_type(), rho.sizes());
#ifndef NDEBUG
  TORCH_CHECK(
      capture_active(rho.device()) || t->digest == options_digest(options),
      "[ThermoY] options changed after prepare(): call reset() first");
#endif

  auto yfrac0 = yfrac.clone();
  auto ivol = compute("DY->V", {rho, yfrac});
  auto vec = ivol.sizes().vec();
  auto reactions = options->reactions();

  // |reactions| x |reactions| weight matrix
  vec[ivol.dim() - 1] = reactions.size() * reactions.size();
  auto gain = torch::empty(vec, ivol.options());

  // A warm start is the active set of the last call or of prepare(). It is
  // only valid for that shape and device. Another shape (one cell, a column)
  // must get a cold start, not those buffers.
  _active_set(rho.device(), rho.sizes(), /*cold=*/!warm_start);

  // diagnostic array
  vec[ivol.dim() - 1] = 1;
  if (!diag.has_value()) {
    diag = torch::zeros(vec, ivol.options());
  }

  // initial guess
  auto temp = compute("VU->T", {ivol, intEng});
  auto pres = compute("VT->P", {ivol, temp});
  auto conc = ivol * inv_mu;

  // prepare data
  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .check_all_same_dtype(false)
          .declare_static_shape(conc.sizes(), /*squash_dims=*/{conc.dim() - 1})
          .add_output(gain)
          .add_output(diag.value())
          .add_output(conc)
          .add_owned_output(temp.unsqueeze(-1))
          .add_owned_input(intEng.unsqueeze(-1))
          .add_input(reaction_set)
          .add_owned_input(nactive.unsqueeze(-1))
          .build();

  int uv_solver = -1;
  if (options->uv_solver() == "kkt") uv_solver = 0;
  if (options->uv_solver() == "auto") uv_solver = 1;
  if (options->uv_solver() == "partition") uv_solver = 2;
  TORCH_CHECK(uv_solver >= 0, "Invalid UV solver: ", options->uv_solver());
  TORCH_CHECK(uv_solver != 2 || uv_partitionable,
              "UV partition solver requires disjoint vapor-cloud reactions");
  if (!uv_partitionable) uv_solver = 0;

  auto intEng_offset = u0 / inv_mu;  // J/kg -> J/mol
  auto cv_const = cv0 / inv_mu;      // J/(kg K) -> J/(mol K)

  // use_h2_cp: hand the solver the H2 energy of eval_intEng_R (the table's
  // intEng_extra holds it in the H2 column)
  auto names = options->species();
  int h2 = std::find(names.begin(), names.end(), "H2") - names.begin();
  if (options->use_h2_cp() && h2 < (int)options->intEng_R_extra().size()) {
    intEng_offset[h2] =
        (options->uref_R()[h2] + kNasa9Tref * options->cref_R()[h2]) *
        constants::Rgas;
    cv_const[h2] = 0.;
  }

  // call the equilibrium solver
  at::native::call_equilibrate_uv(conc.device().type(), iter,
                                  options->vapor_ids().size(), stoich,
                                  intEng_offset, cv_const, *t, options->ftol(),
                                  options->max_iter(), uv_solver);

  // Counts stay on the cells' device, no sync (on CPU, warn here). Row 0 of
  // failures() counts in place (made outside inference mode).
  auto unadjusted = (diag.value() < 0).sum().to(torch::kLong);
  failures_[0].add_(unadjusted);
  if (unadjusted.is_cpu()) {
    auto n = unadjusted.item<int64_t>();
    if (n > 0) {
      TORCH_WARN("ThermoYImpl::forward: saturation adjustment failed in ", n,
                 " cell(s); diag = -(100 * status + iterations)");
    }
  }

  ivol = conc / inv_mu;
  yfrac.copy_(compute("V->Y", {ivol}));

  vec[ivol.dim() - 1] = reactions.size();
  vec.push_back(reactions.size());
  return gain.view(vec);
}

int64_t ThermoYImpl::take_saturation_adjustment_failures() {
  if (!failures_.defined()) return 0;
  auto n = failures_[0].item<int64_t>();
  failures_[0].zero_();
  return n;
}

void ThermoYImpl::_ivol_to_yfrac(torch::Tensor ivol, torch::Tensor& out) const {
  int ny = ivol.size(-1) - 1;
  auto vec = ivol.sizes().vec();
  // (..., ny + 1) -> (ny, ...)
  int ndim = ivol.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  out.permute(vec) = ivol.narrow(-1, 1, ny) / ivol.sum(-1, /*keepdim=*/true);
}

void ThermoYImpl::_yfrac_to_xfrac(torch::Tensor yfrac,
                                  torch::Tensor& out) const {
  int ny = yfrac.size(0);
  auto vec = yfrac.sizes().vec();
  // (ny, ...) -> (..., ny + 1)
  int ndim = yfrac.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  auto mud = options->mu()[0];
  out.narrow(-1, 1, ny) = yfrac.permute(vec) * inv_mu.narrow(0, 1, ny) * mud;

  auto sum = 1. + yfrac.permute(vec).matmul(mud * inv_mu.narrow(0, 1, ny) - 1.);
  out.narrow(-1, 1, ny) /= sum.unsqueeze(-1);
  out.select(-1, 0) = 1. - out.narrow(-1, 1, ny).sum(-1);
}

void ThermoYImpl::_yfrac_to_ivol(torch::Tensor rho, torch::Tensor yfrac,
                                 torch::Tensor& out) const {
  int ny = yfrac.size(0);
  auto vec = yfrac.sizes().vec();
  // (ny, ...) -> (..., ny + 1)
  int ndim = yfrac.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  out.select(-1, 0) = rho * (1. - yfrac.sum(0));
  out.narrow(-1, 1, ny) = (rho.unsqueeze(-1) * yfrac.permute(vec));
}

// T +/- step(T) into `out` as m's plan and mode say: PV->T (row 2) subtracts
// its step, VU->T (row 1) adds it; a failed solve adds one to that row
static void solve(ThermoYImpl const& m, torch::Tensor T, torch::Tensor& out,
                  int row, std::function<at::Tensor(at::Tensor const&)> step) {
  auto next = [&] { return row == 2 ? T - step(T) : T + step(T); };
  auto run = torch::ones({}, T.options().dtype(torch::kBool));
  bool exact = (row == 2 ? m.pres_plan : m.temp_plan) == SolvePlan::exact;
  bool cap = !exact && capture_active(T.device());
  bool fixed = cap || m.solve_mode == SolveMode::fixed;
  TORCH_CHECK(!cap || m.solve_mode != SolveMode::eager_exit,
              "[ThermoY] solve mode eager_exit is not capturable");
  int iter = 0, max_iter = m.options->max_iter();
  for (; iter < (exact ? 1 : max_iter); ++iter) {  // exact: no read
    auto T_prev = T;
    T = fixed ? torch::where(run, next(), T) : next();
    auto err = (1. - T_prev / T).abs().max();
    if (exact) {
      run = T.isfinite().all().logical_not();
    } else if (fixed) {  // the negated test: a NaN keeps running, as on main
      run.logical_and_(
          (err.to(torch::kFloat64) < m.options->ftol()).logical_not());
    } else if (err.item<double>() < m.options->ftol()) {
      run.fill_(false);
      break;
    }
  }
  if (!exact && !fixed && iter == max_iter)
    TORCH_WARN("ThermoYImpl::", row == 2 ? "_pres_to_temp" : "_intEng_to_temp",
               ": max iterations reached");
  m.failures()[row].add_(run);
  out.copy_(T);
}

void ThermoYImpl::_pres_to_temp(torch::Tensor pres, torch::Tensor ivol,
                                torch::Tensor& out) const {
  int ngas = options->vapor_ids().size();

  // kg/m^3 -> mol/m^3, in the compute dtype
  auto conc_gas = (ivol * tables_for(options, ivol)->inv_mu)
                      .narrow(-1, 0, ngas)
                      .clamp_min(options->gas_floor());
  auto T0 = pres / (conc_gas.sum(-1) * constants::Rgas);
  solve(*this, T0, out, /*row=*/2, [&](torch::Tensor const& T) {
    auto cz = eval_czh(T, conc_gas, options);
    auto func = T * (cz * conc_gas).sum(-1) - pres / constants::Rgas;
    auto cv_R = eval_cv_R(T, conc_gas, options);
    auto cp_R = eval_cp_R(T, conc_gas, options);
    // Newton: T <- T - f/f'.  f = T*sum(cz*c) - P/R increases with T, so the
    // step must be SUBTRACTED. (cp_R-cv_R)*c >= f' = sum(cz + T dcz/dT)*c for a
    // dissociating gas (Mayer: cp-cv = (z+T z_T)^2/(z+c z_c) with z_T>0,
    // z_c<0), so this is a safely damped Newton. The old '+=' doubled the error
    // each iteration; it was invisible because for cz == 1 the initial guess T0
    // = P/(R*sum c) is already exact (func == 0) and the loop exits at once.
    // use_h2_dissociation is the first cz != 1 consumer and made PV->T diverge
    // (3900 -> 17000 K).
    return func / ((cp_R - cv_R) * conc_gas).sum(-1);
  });
}

void ThermoYImpl::_cv_vol(torch::Tensor ivol, torch::Tensor temp,
                          torch::Tensor& out) const {
  // kg/m^3 -> mol/m^3, in the compute dtype
  auto conc = ivol * tables_for(options, ivol)->inv_mu;
  auto cv = eval_cv_R(temp, conc, options) * constants::Rgas;
  out.copy_((cv * conc).sum(-1));
}

void ThermoYImpl::_intEng_to_temp(torch::Tensor ivol, torch::Tensor intEng,
                                  torch::Tensor& out) const {
  // kg/m^3 -> mol/m^3, in the compute dtype
  auto t = tables_for(options, ivol);
  auto u0_sum = (ivol * t->u0).sum(-1);
  auto cv0_sum = (ivol * t->cv0).sum(-1);
  auto conc = ivol * t->inv_mu;
  solve(*this, (intEng - u0_sum) / cv0_sum, out, /*row=*/1,
        [&](torch::Tensor const& T) {
          auto u = eval_intEng_R(T, conc, options) * constants::Rgas;
          auto cv = eval_cv_R(T, conc, options) * constants::Rgas;
          return (intEng - (u * conc).sum(-1)) / (cv * conc).sum(-1);
        });
}

void ThermoYImpl::_temp_to_pres(torch::Tensor ivol, torch::Tensor temp,
                                torch::Tensor& out) const {
  int ngas = options->vapor_ids().size();

  // kg/m^3 -> mol/m^3, in the compute dtype
  auto conc_gas =
      (ivol * tables_for(options, ivol)->inv_mu).narrow(-1, 0, ngas);
  auto cz = eval_czh(temp, conc_gas, options);
  out.copy_(constants::Rgas * temp * (cz * conc_gas).sum(-1));
}

}  // namespace kintera
