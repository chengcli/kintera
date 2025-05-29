#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "thermo.h"

double thermo_prop(double temp, double const *conc, int nspecies,
                   double const *offset, double const *first_derivative,
                   user_func1 const *extra) {
  double prop = 0.;
  for (int i = 0; i < nspecies; i++) {
    double propi = offset[i] + first_derivative[i] * temp;
    if (extra[i]) {
      propi += extra[i](temp);
    }
    prop += conc[i] * propi;
  }
  return prop;
}
