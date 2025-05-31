// kintera
#include "vapor_functions.hpp"

VAPOR_FUNCTION(h2o_ideal, T) {
  double betal = 24.845, gammal = 4.986009, betas = 22.98, gammas = 0.52,
         tr = 273.16, pr = 611.7;
  return (T > tr ? logsvp_ideal(T / tr, betal, gammal)
                 : logsvp_ideal(T / tr, betas, gammas)) +
         log(pr);
}

VAPOR_FUNCTION(h2o_ideal_ddT, T) {
  double betal = 24.845, gammal = 4.986009, betas = 22.98, gammas = 0.52,
         tr = 273.16;
  return (T > tr ? logsvp_ideal_ddT(T / tr, betal, gammal)
                 : logsvp_ideal_ddT(T / tr, betas, gammas)) /
         tr;
}

VAPOR_FUNCTION(nh3_ideal, T) {
  double betal = 20.08, gammal = 5.62, betas = 20.64, gammas = 1.43, tr = 195.4,
         pr = 6060.;

  return (T > tr ? logsvp_ideal(T / tr, betal, gammal)
                 : logsvp_ideal(T / tr, betas, gammas)) +
         log(pr);
}

VAPOR_FUNCTION(nh3_ideal_ddT, T) {
  double betal = 20.08, gammal = 5.62, betas = 20.64, gammas = 1.43, tr = 195.4,
         pr = 6060.;

  return (T > tr ? logsvp_ideal_ddT(T / tr, betal, gammal)
                 : logsvp_ideal_ddT(T / tr, betas, gammas)) /
         tr;
}

VAPOR_FUNCTION(nh3_h2s_lewis, T) {
  return (14.82 - 4705. / T) * log(10.) + 2. * log(101325.);
}

VAPOR_FUNCTION(nh3_h2s_lewis_ddT, T) { return 4705. * log(10.) / (T * T); }
