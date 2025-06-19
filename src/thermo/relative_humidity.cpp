
// kintera
#include "relative_humidity.hpp"

#include "log_svp.hpp"

namespace kintera {

torch::Tensor relative_humidity(torch::Tensor temp, torch::Tensor pres,
                                torch::Tensor conc, torch::Tensor stoich,
                                ThermoOptions const& op) {
  // evaluate svp function
  LogSVPFunc::init(op.nucleation());
  auto logsvp = LogSVPFunc::apply(temp);

  // mark reactants
  auto sm = stoich.clamp_max(0.).abs();

  // broadcast stoich to match temp shape
  std::vector<int64_t> vec(temp.dim(), 1);
  vec.push_back(sm.size(0));
  vec.push_back(sm.size(1));

  auto rh = conc.log().unsqueeze(-2).matmul(sm.view(vec)).squeeze(-2);
  rh -= logsvp - sm.sum(0) * pres.log().unsqueeze(-1);
  return rh.exp();
}

}  // namespace kintera
