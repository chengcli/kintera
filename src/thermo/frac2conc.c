#include <stdio.h>

#include "thermo.h"

void frac2conc(double *conc, double p_ov_RT, double const *xfrac, int nspecies,
               int ngas) {
  // check dimensions
  if (nspecies <= 0 || ngas < 1) {
    fprintf(stderr, "Error: nspecies must be positive and ngas >= 1.\n");
    return;  // error: invalid dimensions
  }

  // gas fractions
  double xg = 0.;
  for (int i = 0; i < ngas; i++) xg += xfrac[i];

  // convert gas concentrations
  for (int i = 0; i < ngas; i++) {
    conc[i] = xfrac[i] / xg * p_ov_RT;
  }

  // convert solid concentrations
  for (int i = ngas; i < nspecies; i++) {
    conc[i] = conc[0] * xfrac[i] / xfrac[0];
  }
}
