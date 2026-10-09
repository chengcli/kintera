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
                                NucleationOptions const &op, int ngas) {
  // evaluate svp function
  LogSVPFunc::init(op);
  auto logsvp = LogSVPFunc::call(temp);

  // mark reactants
  TORCH_CHECK(ngas >= 0 || !op || !op->has_gas_products,
              "relative_humidity requires ngas for gaseous reaction products");
  TORCH_CHECK(ngas >= -1 && ngas <= stoich.size(0),
              "Invalid gas species count");
  auto sm = stoich.clamp_max(0.).abs();
  if (ngas >= 0) {
    sm = torch::zeros_like(stoich);
    sm.narrow(0, 0, ngas).copy_(-stoich.narrow(0, 0, ngas));
  }

  auto rh = conc.unsqueeze(-1).pow(sm).prod(-2);
  rh /= torch::exp(logsvp -
                   sm.sum(0) * (constants::Rgas * temp).log().unsqueeze(-1));
  return rh;
}

}  // namespace kintera
