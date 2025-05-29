#include <stdio.h>

#include "thermo.h"

void conc2frac(double *xfrac, double const *conc, int nspecies, int ngas) {
  // check dimensions
  if (nspecies <= 0 || ngas < 1) {
    fprintf(stderr, "Error: nspecies must be positive and ngas >= 1.\n");
    return;  // error: invalid dimensions
  }

  // gas concentrations
  double xg = 0.;
  for (int i = 0; i < ngas; i++) xg += conc[i];

  // convert gas fractions
  for (int i = 0; i < ngas; i++) {
    xfrac[i] = conc[i] / xg;
  }

  // convert solid fractions
  for (int i = ngas; i < nspecies; i++) {
    xfrac[i] = xfrac[0] * conc[i] / conc[0];
  }
}
