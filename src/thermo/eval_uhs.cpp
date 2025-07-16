// kintera
#include "eval_uhs.hpp"

#include <kintera/constants.h>

#include <kintera/utils/utils_dispatch.hpp>

#include "log_svp.hpp"
#include "thermo.hpp"

namespace kintera {

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
  at::native::call_func2(cv_R_extra.device().type(), iter,
                         op.cv_R_extra().data());

  auto cref_R =
      torch::tensor(op.cref_R(), temp.options()).narrow(0, 0, conc.size(-1));
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
  at::native::call_func2(cp_R_extra.device().type(), iter,
                         op.cp_R_extra().data());

  auto cref_R =
      torch::tensor(op.cref_R(), temp.options()).narrow(0, 0, conc.size(-1));
  cref_R.narrow(-1, 0, op.vapor_ids().size()) += 1;
  return cp_R_extra + cref_R;
}

torch::Tensor eval_czh(torch::Tensor temp, torch::Tensor conc,
                       SpeciesThermo const& op) {
  auto cz = torch::zeros_like(conc);
  cz.narrow(-1, 0, op.vapor_ids().size()) = 1.;

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
  at::native::call_func2(cz.device().type(), iter, op.czh().data());

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
  at::native::call_func2(cz_ddC.device().type(), iter, op.czh_ddC().data());

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
                         op.intEng_R_extra().data());

  auto cref_R = torch::tensor(op.cref_R(), temp.options());
  auto uref_R = torch::tensor(op.uref_R(), temp.options());

  return uref_R + temp.unsqueeze(-1) * cref_R + intEng_R_extra;
}

torch::Tensor eval_entropy_R(torch::Tensor temp, torch::Tensor pres,
                             torch::Tensor conc, torch::Tensor stoich,
                             SpeciesThermo const& op) {
  int ngas = op.vapor_ids().size();
  int ncloud = op.cloud_ids().size();

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
                         op.entropy_R_extra().data());

  // std::cout << "entropy_R_extra = " << entropy_R_extra << std::endl;

  auto sref_R = torch::tensor(op.sref_R(), temp.options());
  auto cp_gas_R = torch::tensor(op.cref_R(), temp.options()).narrow(0, 0, ngas);
  cp_gas_R += 1;

  auto entropy_R = torch::zeros_like(conc);

  // gas entropy
  entropy_R.narrow(-1, 0, ngas) =
      sref_R.narrow(0, 0, ngas) + entropy_R_extra +
      temp.log().unsqueeze(-1) * cp_gas_R - pres.log().unsqueeze(-1) -
      (conc_gas / conc_gas.sum(-1, /*keepdim=*/true)).log();

  // std::cout << "entropy_R = " << entropy_R << std::endl;

  return entropy_R;
}

torch::Tensor eval_entropy_R(torch::Tensor temp, torch::Tensor pres,
                             torch::Tensor conc, torch::Tensor stoich,
                             ThermoOptions const& op) {
  int ngas = op.vapor_ids().size();
  int ncloud = op.cloud_ids().size();

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
                         op.entropy_R_extra().data());

  auto sref_R = torch::tensor(op.sref_R(), temp.options());
  auto cp_gas_R = torch::tensor(op.cref_R(), temp.options()).narrow(0, 0, ngas);
  cp_gas_R += 1;

  auto entropy_R = torch::zeros_like(conc);

  // gas entropy
  entropy_R.narrow(-1, 0, ngas) =
      sref_R.narrow(0, 0, ngas) + entropy_R_extra +
      temp.log().unsqueeze(-1) * cp_gas_R - pres.log().unsqueeze(-1) -
      (conc_gas / conc_gas.sum(-1, /*keepdim=*/true)).log();

  //////////// Evaluate condensate entropy ////////////

  // (1) Evaluate log-svp
  LogSVPFunc::init(op.nucleation());
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
  int ngas = op.vapor_ids().size();
  int ncloud = op.cloud_ids().size();

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

  auto cref_R = torch::tensor(op.cref_R(), temp.options());
  auto uref_R = torch::tensor(op.uref_R(), temp.options());
  enthalpy_R.narrow(-1, ngas, ncloud) =
      uref_R.narrow(-1, ngas, ncloud) +
      cref_R.narrow(-1, ngas, ncloud) * temp.unsqueeze(-1) +
      czh.narrow(-1, ngas, ncloud);

  return enthalpy_R;
}

}  // namespace kintera
