
// x -> y at constant pressure (mole fraction)
template <typename T>
inline DISPATCH_MACRO T satfunc1p(T s, T x, T y, T g) {
  // boil all condensates
  if (s > 1.) {
    return -y;
  }

  double rate = (x - s * g) / (1. - s);

  if (rate > 0. || (rate < 0. && y > -rate)) {
    return rate;
  }
  return -y;
}

template <typename T>
inline DISPATCH_MACRO void set_jac1p(T** jac, T s, T const* frac, T g, int j,
                                     int ix, int iy) {
  // boil all condensates
  if (s > 1.) {
    jac(j, iy) = -1.;
    return;
  }
  // s = handle_singularity(s);

  double const& x = frac[ix];
  double const& y = frac[iy];

  double rate = (x - s * g) / (1. - s);

  if (rate > 0. || (rate < 0. && y > -rate)) {
    jac(j, ix) = 1.;
  } else {
    jac(j, iy) = -1.;
  }
}
