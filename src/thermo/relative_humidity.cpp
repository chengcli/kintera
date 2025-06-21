// C/C++
#include <cfloat>

// kintera
#include <kintera/constants.h>

#include "log_svp.hpp"
#include "relative_humidity.hpp"

namespace kintera {

// TODO(cli): correct for non-ideal gas
torch::Tensor relative_humidity(torch::Tensor temp, torch::Tensor conc,
                                torch::Tensor stoich,
                                NucleationOptions const& op) {
  // evaluate svp function
  LogSVPFunc::init(op);
  auto logsvp = LogSVPFunc::apply(temp);

  // mark reactants
  auto sm = stoich.clamp_max(0.).abs();

  // broadcast stoich to match temp shape
  std::vector<int64_t> vec(temp.dim(), 1);
  vec.push_back(sm.size(0));
  vec.push_back(sm.size(1));

  auto rh = conc.unsqueeze(-1).pow(sm.view(vec)).prod(-2);
  rh /= torch::exp(logsvp -
                   sm.sum(0) * (constants::Rgas * temp).log().unsqueeze(-1));
  return rh;
}

}  // namespace kintera
