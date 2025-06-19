// C/C++
#include <cfloat>

// kintera
#include <kintera/constants.h>

#include "log_svp.hpp"
#include "relative_humidity.hpp"

namespace kintera {

// TODO(cli): correct for non-ideal gas
torch::Tensor relative_humidity(torch::Tensor temp, torch::Tensor conc,
                                torch::Tensor stoich, ThermoOptions const& op) {
  // evaluate svp function
  LogSVPFunc::init(op.nucleation());
  auto logsvp = LogSVPFunc::apply(temp);

  // mark reactants
  auto sm = stoich.narrow(0, 0, op.vapor_ids().size()).clamp_max(0.).abs();

  // broadcast stoich to match temp shape
  std::vector<int64_t> vec(temp.dim(), 1);
  vec.push_back(sm.size(0));
  vec.push_back(sm.size(1));

  auto conc_gas = conc.narrow(-1, 0, op.vapor_ids().size());
  conc_gas.clamp_min_(fmax(op.ftol() / 1.e3, FLT_MIN));  // avoid log(0)

  auto rh = conc_gas.log().unsqueeze(-2).matmul(sm.view(vec)).squeeze(-2);
  rh -= logsvp - sm.sum(0) * (constants::Rgas * temp).log().unsqueeze(-1);
  return rh.exp();
}

}  // namespace kintera
