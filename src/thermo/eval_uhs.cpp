// kintera
#include "eval_uhs.hpp"

#include <kintera/utils/utils_dispatch.hpp>

#include "log_svp.hpp"
#include "thermo.hpp"

namespace kintera {

namespace {
// ---- NASA-9 thermo (opt-in: SpeciesThermo.use_nasa9_cp) ----
// T-dependent cp/cv/internal-energy for species that carry NASA-9 data (e.g. H2 rotational/vibrational),
// leaving species without NASA-9 data (lumped "dry", condensates) exactly on the constant-cref_R baseline.
//   cp/R = a0/T^2 + a1/T + a2 + a3 T + a4 T^2 + a5 T^3 + a6 T^4
//   h/R  = -a0/T + a1 lnT + a2 T + a3 T^2/2 + a4 T^3/3 + a5 T^4/4 + a6 T^5/5 + a7  (a7=b1; H continuous at Tmid)
// Internal energy e/R = h/R - T (ideal gas); the deviation (h(T)-h(T0)) - (T-T0) is referenced to T0 so the
// NASA-9 and constant-cv thermo coincide at T0 (continuity; preserves uref_R/sref_R latent-heat references).
constexpr double kNasa9Tref = 300.0;  // matches ThermoOptions default Tref

inline torch::Tensor nasa9_cp_R(torch::Tensor const& a, torch::Tensor const& T) {
  return a.select(-1, 0) * T.pow(-2) + a.select(-1, 1) / T + a.select(-1, 2) +
         a.select(-1, 3) * T + a.select(-1, 4) * T.pow(2) +
         a.select(-1, 5) * T.pow(3) + a.select(-1, 6) * T.pow(4);
}
inline torch::Tensor nasa9_h_R(torch::Tensor const& a, torch::Tensor const& T) {
  return -a.select(-1, 0) / T + a.select(-1, 1) * T.log() + a.select(-1, 2) * T +
         a.select(-1, 3) * T.pow(2) / 2 + a.select(-1, 4) * T.pow(3) / 3 +
         a.select(-1, 5) * T.pow(4) / 4 + a.select(-1, 6) * T.pow(5) / 5 +
         a.select(-1, 7);
}

struct Nasa9Thermo {
  torch::Tensor cp_R;  // (..., nsp) NASA-9 cp/R
  torch::Tensor e_R;   // (..., nsp) internal-energy/R deviation: (h(T)-h(T0)) - (T-T0)   [0 at T0]
  torch::Tensor mask;  // (nsp,) bool: species carries NASA-9 data
};

inline Nasa9Thermo eval_nasa9(torch::Tensor temp, SpeciesThermo const& op, int nsp) {
  auto o = temp.options();
  auto alow = op->nasa9_coeffs_low_tensor(o).narrow(0, 0, nsp);    // (nsp,9)
  auto ahigh = op->nasa9_coeffs_high_tensor(o).narrow(0, 0, nsp);  // (nsp,9)
  auto Tmid = op->nasa9_Tmid_tensor(o).narrow(0, 0, nsp);          // (nsp,)
  auto mask = (alow.abs().sum(-1) + ahigh.abs().sum(-1)) > 0;      // (nsp,)
  auto Tb = temp.unsqueeze(-1);                                    // (...,1)
  auto a = torch::where((Tb < Tmid).unsqueeze(-1), alow, ahigh);   // (...,nsp,9)
  auto a0 = torch::where((kNasa9Tref < Tmid).unsqueeze(-1), alow, ahigh);  // (nsp,9)
  auto T0 = torch::full({1}, kNasa9Tref, o);
  auto cp = nasa9_cp_R(a, Tb);
  auto e = (nasa9_h_R(a, Tb) - nasa9_h_R(a0, T0)) - (Tb - kNasa9Tref);
  return {cp, e, mask};
}

inline bool nasa9_on(SpeciesThermo const& op) {
  return op->use_nasa9_cp() && op->has_nasa9();
}
}  // namespace

torch::Tensor eval_cv_R(torch::Tensor temp, torch::Tensor conc,
                        SpeciesThermo const& op) {
  auto cv_R_extra = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(cv_R_extra.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(cv_R_extra)
                  .add_owned_input(temp.unsqueeze(-1))
                  .add_input(conc)
                  .build();

  // call the evaluation function
  auto cv_R_extra_func = op->intEng_R_extra();
  for (auto& name : cv_R_extra_func) {
    if (!name.empty()) name += "_ddT";
  }

  at::native::call_func2(cv_R_extra.device().type(), iter, cv_R_extra_func);

  auto cref_R =
      torch::tensor(op->cref_R(), temp.options()).narrow(0, 0, conc.size(-1));
  if (nasa9_on(op)) {
    auto n9 = eval_nasa9(temp, op, conc.size(-1));
    return torch::where(n9.mask, n9.cp_R - 1.0, cv_R_extra + cref_R);  // ideal gas cv = cp - R
  }
  return cv_R_extra + cref_R;
}

torch::Tensor eval_cp_R(torch::Tensor temp, torch::Tensor conc,
                        SpeciesThermo const& op) {
  auto cp_R_extra = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(cp_R_extra.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(cp_R_extra)
                  .add_owned_input(temp.unsqueeze(-1))
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_func2(cp_R_extra.device().type(), iter, op->cp_R_extra());

  auto cref_R =
      torch::tensor(op->cref_R(), temp.options()).narrow(0, 0, conc.size(-1));
  cref_R.narrow(-1, 0, op->vapor_ids().size()) += 1;
  if (nasa9_on(op)) {
    auto n9 = eval_nasa9(temp, op, conc.size(-1));
    return torch::where(n9.mask, n9.cp_R, cp_R_extra + cref_R);
  }
  return cp_R_extra + cref_R;
}

torch::Tensor eval_czh(torch::Tensor temp, torch::Tensor conc,
                       SpeciesThermo const& op) {
  auto cz = torch::zeros_like(conc);
  cz.narrow(-1, 0, op->vapor_ids().size()) = 1.;

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(cz.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(cz)
                  .add_owned_input(temp.unsqueeze(-1))
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_func2(cz.device().type(), iter, op->czh());

  return cz;
}

torch::Tensor eval_czh_ddC(torch::Tensor temp, torch::Tensor conc,
                           SpeciesThermo const& op) {
  auto cz_ddC = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(cz_ddC.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(cz_ddC)
                  .add_owned_input(temp.unsqueeze(-1))
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_func2(cz_ddC.device().type(), iter, op->czh_ddC());

  return cz_ddC;
}

torch::Tensor eval_intEng_R(torch::Tensor temp, torch::Tensor conc,
                            SpeciesThermo const& op) {
  auto intEng_R_extra = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(intEng_R_extra.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(intEng_R_extra)
                  .add_owned_input(temp.unsqueeze(-1))
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_func2(intEng_R_extra.device().type(), iter,
                         op->intEng_R_extra());

  auto cref_R = torch::tensor(op->cref_R(), temp.options());
  auto uref_R = torch::tensor(op->uref_R(), temp.options());

  auto base = uref_R + temp.unsqueeze(-1) * cref_R + intEng_R_extra;
  if (nasa9_on(op)) {
    auto n9 = eval_nasa9(temp, op, conc.size(-1));
    // NASA-9 internal energy referenced to T0: uref + T0*cref + (h(T)-h(T0)) - (T-T0)
    auto intEng_nasa = uref_R + kNasa9Tref * cref_R + n9.e_R;
    return torch::where(n9.mask, intEng_nasa, base);
  }
  return base;
}

torch::Tensor eval_entropy_R(torch::Tensor temp, torch::Tensor pres,
                             torch::Tensor conc, torch::Tensor stoich,
                             SpeciesThermo const& op) {
  int ngas = op->vapor_ids().size();
  int ncloud = op->cloud_ids().size();

  // check dimension consistency
  TORCH_CHECK(conc.size(-1) == ngas + ncloud,
              "The last dimension of `conc` must match the number of species "
              "in the thermodynamic model (ngas + ncloud). "
              "Expected: ",
              ngas + ncloud, ", got: ", conc.size(-1));

  TORCH_CHECK(
      stoich.size(0) == ngas + ncloud,
      "The first dimension of `stoich` must match the number of species "
      "in the thermodynamic model (ngas + ncloud). "
      "Expected: ",
      ngas + ncloud, ", got: ", stoich.size(0));

  //////////// Evaluate gas entropy ////////////
  auto conc_gas = conc.narrow(-1, 0, ngas);

  // only evaluate vapors
  auto entropy_R_extra = torch::zeros_like(conc_gas);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(entropy_R_extra.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(entropy_R_extra)
                  .add_owned_input(temp.unsqueeze(-1))
                  .add_owned_input(pres.unsqueeze(-1))
                  .add_input(conc_gas)
                  .build();

  // call the evaluation function
  at::native::call_func3(entropy_R_extra.device().type(), iter,
                         op->entropy_R_extra());

  // std::cout << "entropy_R_extra = " << entropy_R_extra << std::endl;

  auto sref_R = torch::tensor(op->sref_R(), temp.options());
  auto cp_gas_R =
      torch::tensor(op->cref_R(), temp.options()).narrow(0, 0, ngas);
  cp_gas_R += 1;

  auto entropy_R = torch::zeros_like(conc);

  // gas entropy
  entropy_R.narrow(-1, 0, ngas) =
      sref_R.narrow(0, 0, ngas) + entropy_R_extra +
      temp.log().unsqueeze(-1) * cp_gas_R - pres.log().unsqueeze(-1) -
      (conc_gas / conc_gas.sum(-1, /*keepdim=*/true)).clamp_min(1e-300).log();

  // std::cout << "entropy_R = " << entropy_R << std::endl;

  return entropy_R;
}

torch::Tensor eval_entropy_R(torch::Tensor temp, torch::Tensor pres,
                             torch::Tensor conc, torch::Tensor stoich,
                             ThermoOptions const& op) {
  int ngas = op->vapor_ids().size();
  int ncloud = op->cloud_ids().size();

  // check dimension consistency
  TORCH_CHECK(conc.size(-1) == ngas + ncloud,
              "The last dimension of `conc` must match the number of species "
              "in the thermodynamic model (ngas + ncloud). "
              "Expected: ",
              ngas + ncloud, ", got: ", conc.size(-1));

  TORCH_CHECK(
      stoich.size(0) == ngas + ncloud,
      "The first dimension of `stoich` must match the number of species "
      "in the thermodynamic model (ngas + ncloud). "
      "Expected: ",
      ngas + ncloud, ", got: ", stoich.size(0));

  //////////// Evaluate gas entropy ////////////
  auto conc_gas = conc.narrow(-1, 0, ngas);

  // only evaluate vapors
  auto entropy_R_extra = torch::zeros_like(conc_gas);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(entropy_R_extra.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(entropy_R_extra)
                  .add_owned_input(temp.unsqueeze(-1))
                  .add_owned_input(pres.unsqueeze(-1))
                  .add_input(conc_gas)
                  .build();

  // call the evaluation function
  at::native::call_func3(entropy_R_extra.device().type(), iter,
                         op->entropy_R_extra());

  auto sref_R = torch::tensor(op->sref_R(), temp.options());
  auto cp_gas_R =
      torch::tensor(op->cref_R(), temp.options()).narrow(0, 0, ngas);
  cp_gas_R += 1;

  auto entropy_R = torch::zeros_like(conc);

  // gas entropy
  entropy_R.narrow(-1, 0, ngas) =
      sref_R.narrow(0, 0, ngas) + entropy_R_extra +
      temp.log().unsqueeze(-1) * cp_gas_R - pres.log().unsqueeze(-1) -
      (conc_gas / conc_gas.sum(-1, /*keepdim=*/true)).clamp_min(1e-300).log();

  //////////// Evaluate condensate entropy ////////////

  // (1) Evaluate log-svp
  LogSVPFunc::init(op->nucleation());
  auto logsvp = LogSVPFunc::call(temp);
  // std::cout << "logsvp = " << logsvp << std::endl;

  // (2) Evaluate enthalpies (..., R)
  auto enthalpy_R = eval_enthalpy_R(temp, conc, op).matmul(stoich);
  // std::cout << "enthalpy_R = " << enthalpy_R << std::endl;

  // (3) Evaluate equilibrium vapor entropies (..., V)
  auto entropy_vapor_R = sref_R.narrow(0, 0, ngas) + entropy_R_extra +
                         temp.log().unsqueeze(-1) * cp_gas_R;
  // std::cout << "entropy_vaor_R = " << entropy_vapor_R << std::endl;

  // (4) Assemble b vector (..., R)
  auto b = enthalpy_R / temp.unsqueeze(-1) -
           entropy_vapor_R.matmul(stoich.narrow(0, 0, ngas)) - logsvp;
  // std::cout << "b = " << b << std::endl;

  // (5) Assemble S+ matrix and its inverse
  auto sp = stoich.narrow(0, ngas, ncloud);
  auto sp_inv = torch::linalg_pinv(sp.t());
  // std::cout << "sp_inv = " << sp_inv << std::endl;

  // broadcast the shape of sp.t()
  std::vector<int64_t> vec2(1 + b.dim(), 1);
  vec2[b.dim() - 1] = sp.size(0);
  vec2[b.dim()] = sp.size(1);
  // std::cout << "sp_inv view = " << sp_inv.view(vec2) << std::endl;

  // (6) Solve for condensate entropy
  entropy_R.narrow(-1, ngas, ncloud) =
      sp_inv.view(vec2).matmul(b.unsqueeze(-1)).squeeze(-1);

  // std::cout << "entropy_R = " << entropy_R << std::endl;

  return entropy_R;
}

torch::Tensor eval_enthalpy_R(torch::Tensor temp, torch::Tensor conc,
                              SpeciesThermo const& op) {
  int ngas = op->vapor_ids().size();
  int ncloud = op->cloud_ids().size();

  // check dimension consistency
  TORCH_CHECK(conc.size(-1) == ngas + ncloud,
              "The last dimension of `conc` must match the number of species "
              "in the thermodynamic model (ngas + ncloud). "
              "Expected: ",
              ngas + ncloud, ", got: ", conc.size(-1));

  auto enthalpy_R = torch::zeros_like(conc);
  auto czh = eval_czh(temp, conc, op);

  enthalpy_R.narrow(-1, 0, ngas) =
      eval_intEng_R(temp, conc, op).narrow(-1, 0, ngas) +
      czh.narrow(-1, 0, ngas) * temp.unsqueeze(-1);

  auto cref_R = torch::tensor(op->cref_R(), temp.options());
  auto uref_R = torch::tensor(op->uref_R(), temp.options());
  enthalpy_R.narrow(-1, ngas, ncloud) =
      uref_R.narrow(-1, ngas, ncloud) +
      cref_R.narrow(-1, ngas, ncloud) * temp.unsqueeze(-1) +
      czh.narrow(-1, ngas, ncloud);

  return enthalpy_R;
}

}  // namespace kintera
