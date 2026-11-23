#pragma once

// C/C++
#include <initializer_list>
#include <iomanip>

// torch
#include <ATen/native/DispatchStub.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include <kintera/constants.h>

#include <kintera/species.hpp>

#include "eval_uhs.hpp"
#include "nucleation.hpp"

// arg
#include <kintera/add_arg.h>

namespace YAML {
class Node;
}  // namespace YAML

namespace at::native {  // kernels: utils_dispatch.cpp (CPU), .cu (CUDA)
using fill_fn = at::Tensor (*)(std::vector<std::string> const& names, int arity,
                               at::Device device);
using capture_fn = bool (*)(at::Device device);
DECLARE_DISPATCH(fill_fn, fill_func_table);
DECLARE_DISPATCH(capture_fn, capture_active);
}  // namespace at::native

namespace kintera {

//! true while the current stream of `device` captures a graph
inline bool capture_active(torch::Device device) {
  return device.is_cuda() && at::native::capture_active(device.type(), device);
}

//! X opens one around a segment's warm-up and capture: tables_for builds no
//! table inside it; used() lists the tables returned or used in the scope
class NoTableBuild {
 public:
  NoTableBuild() : prev_(top()) { top() = this; }
  ~NoTableBuild() { top() = prev_; }
  NoTableBuild(NoTableBuild const&) = delete;
  std::vector<ThermoTablesPtr> const& used() const { return used_; }
  static NoTableBuild*& top() {  // the innermost open scope of this thread
    static thread_local NoTableBuild* scope = nullptr;
    return scope;
  }
  static void record(ThermoTablesPtr const& t) {
    if (top()) top()->used_.push_back(t);
  }

 private:
  NoTableBuild* prev_;
  std::vector<ThermoTablesPtr> used_;
};

// MPS can prepare a column's tensor state. Its unused callback pointers and
// FP64 saturation parameters must stay on the host.
inline at::Device thermo_callback_device(at::Device device) {
  return device.is_mps() ? at::Device(at::kCPU) : device;
}

template <typename T>  // FNV-1a over values and sizes, for options_digest
void fnv1a(uint64_t& h, T const& v) {
  if constexpr (std::is_arithmetic_v<T>) {
    auto c = reinterpret_cast<unsigned char const*>(&v);
    for (size_t i = 0; i < sizeof(T); ++i) h = (h ^ c[i]) * 1099511628211ull;
  } else {
    for (auto const& x : v) fnv1a(h, x);
    fnv1a(h, v.size());
  }
}

//! the species columns of op's table for like's device and dtype (tables_for;
//! with use_h2_cp, the H2 column of intEng_extra is the H2 energy)
inline std::shared_ptr<ThermoTables> species_tables(SpeciesThermo const& op,
                                                    at::Tensor const& like) {
  auto t = std::make_shared<ThermoTables>();
  auto o = like.options();
  auto dev = thermo_callback_device(like.device());
  auto ft = [&](std::vector<std::string> n, int arity, std::string sfx = "") {
    for (auto& s : n) s += s.empty() ? "" : sfx;
    return at::native::fill_func_table(dev.type(), n, arity, dev);
  };
  t->cref_R = torch::tensor(op->cref_R(), o);
  t->uref_R = torch::tensor(op->uref_R(), o);
  t->sref_R = torch::tensor(op->sref_R(), o);
  if (op->mu().size() == op->cref_R().size()) {  // as ThermoY's buffers
    t->inv_mu = (1. / torch::tensor(op->mu(), torch::kFloat64)).to(o);
    t->cv0 = t->cref_R * constants::Rgas * t->inv_mu;
    t->u0 = t->uref_R * constants::Rgas * t->inv_mu;
  }
  if (op->has_nasa9()) {
    t->nasa9_low = op->nasa9_coeffs_low_tensor(o);
    t->nasa9_high = op->nasa9_coeffs_high_tensor(o);
    t->nasa9_Tmid = op->nasa9_Tmid_tensor(o);
  }
  int nsp = op->cref_R().size(), h2 = nsp;
  auto u = op->intEng_R_extra();
  std::vector<std::string> s(nsp);
  if (op->use_h2_cp()) {
    auto names = op->species();
    h2 = std::find(names.begin(), names.end(), "H2") - names.begin();
  }
  t->h2_mask = torch::zeros({nsp}, o.dtype(torch::kBool));
  if (h2 < nsp) {
    t->h2_mask[h2] = true;
    s[h2] = "h2_rot_" + op->h2_cp_mode() + "_s";
    if (h2 < (int)u.size()) u[h2] = "h2_rot_" + op->h2_cp_mode();
  }
  t->intEng_extra = ft(u, 2), t->intEng_extra_ddT = ft(u, 2, "_ddT");
  t->h2_entropy = ft(s, 2), t->cp_R_extra = ft(op->cp_R_extra(), 2);
  t->czh = ft(op->czh(), 2), t->czh_ddC = ft(op->czh_ddC(), 2);
  t->entropy_extra = ft(op->entropy_R_extra(), 3);
  return t;
}

template <typename T>
inline std::vector<T> insert_first(T value, std::vector<T> const& input) {
  std::vector<T> result;
  result.reserve(input.size() + 1);
  result.push_back(value);
  result.insert(result.end(), input.begin(), input.end());
  return result;
}

struct ThermoOptionsImpl final : public SpeciesThermoImpl {
  static std::shared_ptr<ThermoOptionsImpl> create() {
    return std::make_shared<ThermoOptionsImpl>();
  }

  //! from a YAML file or node: a `ThermoOptions` if it contains
  //! "reference-state", else nullptr
  static std::shared_ptr<ThermoOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);
  static std::shared_ptr<ThermoOptionsImpl> from_yaml(YAML::Node const& config,
                                                      bool verbose = false);

  std::shared_ptr<ThermoOptionsImpl> clone() const {
    auto op = std::make_shared<ThermoOptionsImpl>(*this);
    if (nucleation() != nullptr) {
      op->nucleation() = nucleation()->clone();
    }
    return op;
  }
  void report(std::ostream& os) const {
    os << "-- thermodynamics options --\n";
    os << "* Tref = " << Tref() << " K\n"
       << "* Pref = " << Pref() << " Pa\n"
       << "* max_iter = " << max_iter() << "\n"
       << "* ftol = " << ftol() << "\n"
       << "* uv_solver = " << uv_solver() << "\n"
       << "* gas_floor = " << gas_floor() << "\n"
       << "* offset_zero = " << (offset_zero() ? "true" : "false") << "\n"
       << "* verbose = " << (verbose() ? "true" : "false") << "\n";
  }

  std::vector<Reaction> reactions() const;

  ADD_ARG(double, Tref) = 300.0;
  ADD_ARG(double, Pref) = 1.e5;

  ADD_ARG(int, max_iter) = 100;
  ADD_ARG(double, ftol) = 1e-6;
  ADD_ARG(double, gas_floor) = 1.e-20;
  ADD_ARG(bool, verbose) = false;
  ADD_ARG(bool, offset_zero) = false;

  ADD_ARG(NucleationOptions, nucleation) = NucleationOptionsImpl::create();
  ADD_ARG(std::string, uv_solver) = "auto";

  //! columns of coagulation copies X(p) and of their sources X
  ADD_ARG(std::vector<int>, copy_cols);
  ADD_ARG(std::vector<int>, copy_src_cols);
};
using ThermoOptions = std::shared_ptr<ThermoOptionsImpl>;

//! The parts of ThermoX::reset and ThermoY::reset that are the same for both
//! (thermo_options.cpp); `tag` names the caller in the verbose prints.
//! Default and check the nucleation options, then check_dimensions.
void thermo_check_options(ThermoOptions const& options, char const* tag);
//! Move uref_R to T = 0 and sref_R to T = 1, P = 1 once per options object
//! (offset_zero), then print the reference tables when verbose.
void thermo_zero_offsets(ThermoOptions const& options, char const* tag);
//! The stoichiometry matrix (float64, nspecies x nreaction) and the
//! nucleation product check.
torch::Tensor thermo_stoich(ThermoOptions const& options);

struct ExtrapOptions {
  // Logarithmic change in pressure (dlnp = ln(p_new / p_old))
  ADD_ARG(double, dlnp) = 0.;
  ADD_ARG(double, dz) = 0.;

  // Gravitational acceleration (m/s^2), positive
  ADD_ARG(double, grav) = 0;
  ADD_ARG(double, ds_dlnp) = 0;

  // Height change (m)
  ADD_ARG(double, ds_dz) = 0;

  // If true, print debug information
  ADD_ARG(bool, verbose) = false;

  // Drop condensates
  ADD_ARG(bool, rainout) = false;
};

//! VU->T and PV->T: exact (constant heat capacity: the first guess and one
//! correction, no read) or newton (iterate as the SolveMode says)
enum class SolvePlan { exact, newton };
//! newton: eager_exit reads the host each pass; fixed runs max_iter passes and
//! freezes the converged state on the device; automatic: fixed in capture
enum class SolveMode { automatic, eager_exit, fixed };

//! (reaction_set, nactive): the KKT warm start, int32 (..., nreaction), (...)
using ActiveSet = std::pair<torch::Tensor, torch::Tensor>;

//! Mass Thermodynamics
class ThermoYImpl : public torch::nn::Cloneable<ThermoYImpl> {
 public:
  //! Create a ThermoY module from `opts`, registered as `name` under `p`
  static std::shared_ptr<ThermoYImpl> create(
      ThermoOptions const& opts, torch::nn::Module* p,
      std::string const& name = "thermo");

  //! 1/mu [mol/kg], cv0 [J/(kg K)], u0 at T = 0 [J/kg] (float64; the tables
  //! hold them per dtype), stoich (nspecies, nreaction)
  torch::Tensor inv_mu, cv0, u0, stoich;

  //! kkt warm start active set
  torch::Tensor reaction_set, nactive;

  //! options with which this `ThermoY` was constructed
  ThermoOptions options;
  bool uv_partitionable = false;
  //! chosen by reset() from the options only; solve_mode is read per call
  SolvePlan temp_plan = SolvePlan::newton, pres_plan = SolvePlan::newton;
  SolveMode solve_mode = SolveMode::automatic;

  ThermoYImpl() : options(ThermoOptionsImpl::create()) {}
  explicit ThermoYImpl(const ThermoOptions& options_);
  //! options1 owns heat-capacity policy; options2 contributes species records.
  //! Merged species records update the shared options1 object in place.
  ThermoYImpl(const ThermoOptions& options1, const SpeciesThermo& options2);
  void reset() override;
  void pretty_print(std::ostream& os) const override;

  //! \brief perform conversions; compute() allocates `out` for compute_into,
  //! which writes it with copy_ (any dtype; tables in the input dtype)
  torch::Tensor compute(std::string ab, std::vector<torch::Tensor> const& args);
  void compute_into(std::string ab, std::vector<torch::Tensor> const& args,
                    torch::Tensor out);

  //! the tables, counter and active set for cells of `shape` (K.rst 3.3),
  //! outside capture; a repeat with the same arguments reallocates nothing
  ThermoTablesPtr prepare(torch::Device device, torch::Dtype dtype,
                          torch::IntArrayRef shape);
  ThermoTablesPtr tables() const { return tables_; }  // null before prepare

  //! copy_ between fixed buffers; a snapshot prepares a missing active set
  ActiveSet snapshot_active_set(ActiveSet out);
  void restore_active_set(ActiveSet const& in);
  ActiveSet make_active_set_storage() const;  // buffers of the module's shape

  //! (3,) int64 on the cells' device, rows saturation, temp_solve,
  //! pres_solve; never read here. reset_ zeroes all rows; consume_ copies
  //! row 0 > 0 into `out` and zeroes row 0. Both capturable.
  torch::Tensor const& failures() const { return failures_; }
  void reset_failures();
  void consume_failures(torch::Tensor out);
  //! host read of failures() row 0, then row 0 zeroed; not capturable
  int64_t take_saturation_adjustment_failures();

  //! \brief saturation adjustment of yfrac (ny, ...) in place, from density
  //! rho and total internal energy intEng [J/m^3]; caches T and P (buffer("T"),
  //! buffer("P")); warm_start: from the previous active set; diag: optional
  //! (..., ndiag); returns the gain matrix (..., nreaction, nreaction)
  torch::Tensor forward(torch::Tensor rho, torch::Tensor intEng,
                        torch::Tensor const& yfrac, bool warm_start = false,
                        torch::optional<torch::Tensor> diag = torch::nullopt);

 private:
  ThermoTablesPtr tables_;  // of the last prepare(); reset() drops it
  torch::Tensor failures_;
  torch::Tensor const& _counter(torch::Device device);  // made if missing
  //! the last warm start, or a cold one (in place when the buffers fit)
  void _active_set(torch::Device device, torch::IntArrayRef shape, bool cold);

  //! inverse specific volume ivol [kg/m^3] (..., 1 + ny) -> yfrac (ny, ...)
  void _ivol_to_yfrac(torch::Tensor ivol, torch::Tensor& out) const;

  //! mass fraction (ny, ...) -> mole fraction (..., 1 + ny)
  void _yfrac_to_xfrac(torch::Tensor yfrac, torch::Tensor& out) const;

  //! total density rho [kg/m^3], yfrac (ny, ...) -> ivol [kg/m^3] (..., 1 + ny)
  void _yfrac_to_ivol(torch::Tensor rho, torch::Tensor yfrac,
                      torch::Tensor& out) const;

  //! pres [Pa], ivol [kg/m^3] (..., 1 + ny) -> temperature [K] (...)
  void _pres_to_temp(torch::Tensor pres, torch::Tensor ivol,
                     torch::Tensor& out) const;

  //! ivol [kg/m^3] (..., 1 + ny), temp [K] -> volumetric cv [J/(m^3 K)] (...)
  void _cv_vol(torch::Tensor ivol, torch::Tensor temp,
               torch::Tensor& out) const;

  //! ivol [kg/m^3] (..., 1 + ny), temp [K] -> volumetric intEng [J/m^3] (...)
  void _intEng_vol(torch::Tensor ivol, torch::Tensor temp,
                   torch::Tensor& out) const {
    // kg/m^3 -> mol/m^3, in the compute dtype
    auto conc = ivol * tables_for(options, ivol)->inv_mu;
    auto ui = eval_intEng_R(temp, conc, options) * constants::Rgas;
    out.copy_((ui * conc).sum(-1));
  }

  //! ivol [kg/m^3] (..., 1 + ny), intEng [J/m^3] (...) -> temperature [K] (...)
  void _intEng_to_temp(torch::Tensor ivol, torch::Tensor intEng,
                       torch::Tensor& out) const;

  //! ivol [kg/m^3] (..., 1 + ny), temp [K] -> pressure [Pa] (...)
  void _temp_to_pres(torch::Tensor ivol, torch::Tensor temp,
                     torch::Tensor& out) const;

  //! pres [Pa], ivol [kg/m^3] (..., 1 + ny), temp [K] -> entropy [J/(m^3 K)]
  void _entropy_vol(torch::Tensor pres, torch::Tensor ivol, torch::Tensor temp,
                    torch::Tensor& out) const {
    // kg/m^3 -> mol/m^3, in the compute dtype
    auto conc = ivol * tables_for(options, ivol)->inv_mu;
    auto si =
        eval_entropy_R(temp, pres, conc, stoich, options) * constants::Rgas;
    out.copy_((si * conc).sum(-1));
  }
};
TORCH_MODULE(ThermoY);

//! host hash of the address, sizes and dtype of m's buffers (a ThermoY adds
//! its counter, active set and table): any reallocation changes it
inline int64_t alloc_epoch(torch::nn::Module const& m) {  // FNV-1a, host only
  auto ts = m.buffers();
  if (auto y = dynamic_cast<ThermoYImpl const*>(&m))
    ts.insert(ts.end(), {y->failures(), y->reaction_set, y->nactive,
                         y->tables() ? y->tables()->cv0 : at::Tensor()});
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
  for (auto const& t : ts) {
    mix(t.defined() ? reinterpret_cast<uintptr_t>(t.data_ptr()) : 0);
    for (auto s : t.defined() ? t.sizes() : at::IntArrayRef()) mix(s);
    mix(t.defined() ? static_cast<uint64_t>(t.scalar_type()) : 99);
  }
  return static_cast<int64_t>(h);
}

//! Molar thermodynamics
class ThermoXImpl : public torch::nn::Cloneable<ThermoXImpl> {
 public:
  //! Create a ThermoX module from `opts`, registered as `name` under `p`
  static std::shared_ptr<ThermoXImpl> create(
      ThermoOptions const& opts, torch::nn::Module* p,
      std::string const& name = "thermo");

  //! mu.
  torch::Tensor mu;

  //! stoichiometry matrix
  torch::Tensor stoich;

  //! kkt warm start active set
  torch::Tensor reaction_set, nactive;

  //! options with which this `ThermoX` was constructed
  ThermoOptions options;

  ThermoXImpl() : options(ThermoOptionsImpl::create()) {}
  explicit ThermoXImpl(const ThermoOptions& options_);
  //! options1 owns heat-capacity policy; options2 contributes species records.
  //! Merged species records update the shared options1 object in place.
  ThermoXImpl(const ThermoOptions& options1, const SpeciesThermo& options2);
  void reset() override;
  void pretty_print(std::ostream& os) const override;

  //! \brief perform the conversion named `ab` on `args`; returns its result
  torch::Tensor compute(std::string ab, std::vector<torch::Tensor> const& args);

  //! equivalent cp [J/(mol K)] from temp [K], pres [Pa], mole fraction xfrac
  //! and the gain; conc is computed if not given
  torch::Tensor effective_cp(
      torch::Tensor temp, torch::Tensor pres, torch::Tensor xfrac,
      torch::Tensor gain, torch::optional<torch::Tensor> conc = torch::nullopt);

  //! extrapolate temp [K], pres [Pa] and mole fraction xfrac in place to a new
  //! pressure along an adiabat
  void extrapolate_dlnp(torch::Tensor temp, torch::Tensor pres,
                        torch::Tensor xfrac, ExtrapOptions const& opts);

  //! the rain-out entropy target of extrapolate_dlnp / extrapolate_dz
  //! (extrapolate_ad.cpp); removes the clouds from xfrac in place
  torch::Tensor rainout_entropy_target(torch::Tensor temp, torch::Tensor pres,
                                       torch::Tensor xfrac,
                                       torch::Tensor const& x_dry0,
                                       torch::Tensor const& entropy_mole0,
                                       double ds_term, int ngas, int ncloud);

  //! the same to a new height
  void extrapolate_dz(torch::Tensor temp, torch::Tensor pres,
                      torch::Tensor xfrac, ExtrapOptions const& opts);

  //! \brief equilibrium state at temp [K] and pres [Pa]: xfrac (..., 1 + ny) in
  //! place; warm_start is accepted for compatibility and has no effect; diag:
  //! optional (..., ndiag); returns the gain matrix (..., nreaction, nreaction)
  torch::Tensor forward(torch::Tensor temp, torch::Tensor pres,
                        torch::Tensor const& xfrac, bool warm_start = false,
                        torch::optional<torch::Tensor> diag = torch::nullopt);

 private:
  //! mole fraction (..., 1 + ny) -> mass fraction (ny, ...)
  void _xfrac_to_yfrac(torch::Tensor xfrac, torch::Tensor& out) const;

  //! mole concentration (..., 1 + ny) -> total density [kg/m^3] (...)
  void _conc_to_dens(torch::Tensor conc, torch::Tensor& out) const {
    out.set_((conc * mu).sum(-1));
  }

  //! temp [K], conc (..., 1 + ny) -> volumetric heat capacity [J/(m^3 K)] (...)
  void _cp_vol(torch::Tensor temp, torch::Tensor conc,
               torch::Tensor& out) const {
    auto cp = eval_cp_R(temp, conc, options) * constants::Rgas;
    out.set_((conc * cp).sum(-1));
  }

  //! temp [K], conc (..., 1 + ny) -> volumetric enthalpy [J/m^3] (...)
  void _enthalpy_vol(torch::Tensor temp, torch::Tensor conc,
                     torch::Tensor& out) const {
    auto hi = eval_enthalpy_R(temp, conc, options) * constants::Rgas;
    out.set_((conc * hi).sum(-1));
  }

  //! temp [K], pres [Pa] (...), conc (..., 1 + ny) -> entropy [J/(m^3 K)] (...)
  void _entropy_vol(torch::Tensor temp, torch::Tensor pres, torch::Tensor conc,
                    torch::Tensor& out) const {
    auto si =
        eval_entropy_R(temp, pres, conc, stoich, options) * constants::Rgas;
    out.set_((conc * si).sum(-1));
  }

  //! pres [Pa], xfrac (..., 1 + ny), entropy [J/(m^3 K)] -> temperature [K]
  void _entropy_to_temp(torch::Tensor pres, torch::Tensor xfrac,
                        torch::Tensor entropy, torch::Tensor& out);

  //! temp [K], pres [Pa], xfrac (..., 1 + ny) -> conc [mol/m^3] (..., 1 + ny)
  void _xfrac_to_conc(torch::Tensor temp, torch::Tensor pres,
                      torch::Tensor xfrac, torch::Tensor& out) const;
};
TORCH_MODULE(ThermoX);

}  // namespace kintera

#undef ADD_ARG
