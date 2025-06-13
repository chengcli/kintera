// kintera
#include "eval_uhs.hpp"

#include <kintera/constants.h>

#include "thermo.hpp"
#include "thermo_dispatch.hpp"

namespace kintera {

torch::Tensor eval_cv_R(torch::Tensor temp, torch::Tensor conc,
                        ThermoOptions const& op) {
  auto cv_R_extra = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(cv_R_extra.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(cv_R_extra)
                  .add_owned_input(temp.unsqueeze(-1).expand_as(conc))
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_with_TC(cv_R_extra.device().type(), iter,
                           op.cv_R_extra().data());

  auto cref_R =
      torch::tensor(op.cref_R(), temp.options()).narrow(0, 0, conc.size(-1));
  return cv_R_extra + cref_R;
}

torch::Tensor eval_cp_R(torch::Tensor temp, torch::Tensor conc,
                        ThermoOptions const& op) {
  auto cp_R_extra = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(cp_R_extra.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(cp_R_extra)
                  .add_owned_input(temp.unsqueeze(-1).expand_as(conc))
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_with_TC(cp_R_extra.device().type(), iter,
                           op.cp_R_extra().data());

  auto cref_R =
      torch::tensor(op.cref_R(), temp.options()).narrow(0, 0, conc.size(-1));
  cref_R.narrow(-1, 0, 1 + op.vapor_ids().size()) += 1;
  return cp_R_extra + cref_R;
}

torch::Tensor eval_czh(torch::Tensor temp, torch::Tensor conc,
                       ThermoOptions const& op) {
  auto cz = torch::zeros_like(conc);
  cz.narrow(-1, 0, 1 + op.vapor_ids().size()) = 1.;

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(cz.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(cz)
                  .add_owned_input(temp.unsqueeze(-1).expand_as(conc))
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_with_TC(cz.device().type(), iter, op.czh().data());

  return cz;
}

torch::Tensor eval_czh_ddC(torch::Tensor temp, torch::Tensor conc,
                           ThermoOptions const& op) {
  auto cz_ddC = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(cz_ddC.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(cz_ddC)
                  .add_owned_input(temp.unsqueeze(-1).expand_as(conc))
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_with_TC(cz_ddC.device().type(), iter, op.czh_ddC().data());

  return cz_ddC;
}

torch::Tensor eval_intEng_R(torch::Tensor temp, torch::Tensor conc,
                            ThermoOptions const& op) {
  auto intEng_R_extra = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(intEng_R_extra.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(intEng_R_extra)
                  .add_owned_input(temp.unsqueeze(-1).expand_as(conc))
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_with_TC(intEng_R_extra.device().type(), iter,
                           op.intEng_R_extra().data());

  auto cref_R = torch::tensor(op.cref_R(), temp.options());
  auto uref_R = torch::tensor(op.uref_R(), temp.options());

  return uref_R + temp.unsqueeze(-1) * cref_R + intEng_R_extra;
}

torch::Tensor eval_entropy_R(torch::Tensor temp, torch::Tensor pres,
                             torch::Tensor conc, ThermoOptions const& op) {
  int ngas = 1 + op.vapor_ids().size();
  int ncloud = op.cloud_ids().size();

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
                  .add_owned_input(temp.unsqueeze(-1).expand_as(conc_gas))
                  .add_owned_input(pres.unsqueeze(-1).expand_as(conc_gas))
                  .add_input(conc_gas)
                  .build();

  // call the evaluation function
  at::native::call_with_TCP(entropy_R_extra.device().type(), iter,
                            op.entropy_R_extra().data());

  auto sref_R = torch::tensor(op.sref_R(), temp.options());
  auto cp_gas_R = torch::tensor(op.cref_R(), temp.options()).narrow(0, 0, ngas);
  cp_gas_R += 1;

  auto entropy = torch::zeros_like(conc);

  // gas entropy
  entropy.narrow(-1, 0, ngas) =
      sref_R.narrow(0, 0, ngas) + entropy_R_extra +
      temp.log().unsqueeze(-1) * cp_gas_R - pres.log() -
      (conc_gas / conc_gas.sum(-1, /*keepdim=*/true)).log();

  //////////// Evaluate condensate entropy ////////////

  // (1) Evaluate log-svp

  // assemble stoichiometric matrix
  auto stoich = torch::zeros({, op.react().size()}, conc.options());
  for (int j = 0; j < op.react().size(); ++j) {
    auto const& r = options.react()[j];
    for (int i = ngas; i < op.species().size(); ++i) {
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

torch::Tensor eval_enthalpy_R(torch::Tensor temp, torch::Tensor conc,
                              ThermoOptions const& op) {
  int ngas = 1 + op.vapor_ids().size();
  int ncloud = op.cloud_ids().size();

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
