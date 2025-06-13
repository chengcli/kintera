// kintera
#include "integrate_z.hpp"

#include <kintera/constants.h>

#include "thermo_dispatch.hpp"

namespace kintera {

void call_integrate_z_cpu(at::TensorIterator &iter, double dz,
                          char const *method, double grav, double adTdz,
                          user_func1 const *logsvp_func, double logsvp_eps,
                          int max_iter);

void integrate_z_(torch::Tensor temp, torch::Tensor pres, torch::Tensor xfrac,
                  double dz, std::string method, double grav, double adTdz,
                  ThermoX &thermo) {
  // prepare reduced stoichiometry matrix
  int nspecies = thermo->options.species().size();
  int ngas = 1 + thermo->options.vapor_ids().size();

  // prepare mu
  auto mu = torch::zeros({nspecies}, torch::kFloat64);
  mu[0] = constants::Rgas / thermo->options.Rd();
  for (int i = 0; i < nspecies; ++i) {
    mu[i] = thermo->options.mu_ratio()[i] * mu[0];
  }
  mu.to(temp.device());

  // prepare concentration
  auto conc = thermo->compute("TPX->C", {temp, pres, xfrac});

  // prepare enthalpy and cp
  auto enthalpy = thermo->compute("TC->H", {temp, conc});
  auto cp = thermo->compute("TC->cp", {temp, conc});

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .add_output(temp)
                  .add_output(pres)
                  .add_output(xfrac)
                  .add_input(enthalpy)
                  .add_input(cp)
                  .add_input(thermo->stoich)
                  .add_input(mu)
                  .build();

  // prepare svp function
  user_func1 *logsvp_func = new user_func1[thermo->options.react().size()];
  for (int i = 0; i < thermo->options.react().size(); ++i) {
    logsvp_func[i] = thermo->options.react()[i].func();
  }

  // call the integrator
  at::native::call_integrate_z(temp.device().type(), iter, dz, method.c_str(),
                               grav, adTdz, logsvp_func, thermo->options.ftol(),
                               thermo->options.max_iter());

  delete[] logsvp_func;
}

}  // namespace kintera
