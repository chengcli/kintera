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
  auto logsvp = LogSVPFunc::call(temp);

  // mark reactants
  auto sm = stoich.clamp_max(0.).abs();

  auto rh = conc.unsqueeze(-1).pow(sm).prod(-2);
  rh /= torch::exp(logsvp -
                   sm.sum(0) * (constants::Rgas * temp).log().unsqueeze(-1));
  return rh;
}

}  // namespace kintera
