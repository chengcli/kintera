#include <stdlib.h>

#include "linalg.h"

void luminv(double *y, double const *a, int const *indx, int n) {
  double *col = (double *)malloc(n * sizeof(double));
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) col[i] = 0.0;
    col[j] = 1.0;
    lubksb(col, a, indx, n);
    for (int i = 0; i < n; i++) y[i * n + j] = col[i];
  }
  free(col);
}
