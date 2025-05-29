#include <assert.h>

void mvdot(double *r, double const *m, double const *v, int n1, int n2) {
  assert(r != v);  // r and v cannot be the same
  for (int i = 0; i < n1; ++i) {
    r[i] = 0.;
    for (int j = 0; j < n2; ++j) r[i] += m[i * n2 + j] * v[j];
  }
}
