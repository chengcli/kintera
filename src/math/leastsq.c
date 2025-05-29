#include <stdlib.h>
#include <string.h>

#include "linalg.h"

void leastsq(double *b, double const *a, int n1, int n2) {
  double *c = (double *)malloc(n1 * sizeof(double));
  memcpy(c, b, n1 * sizeof(double));

  double *y = (double *)malloc(n2 * n2 * sizeof(double));

  for (int i = 0; i < n2; ++i) {
    // calculate A^T.A
    for (int j = 0; j < n2; ++j) {
      y[i * n2 + j] = 0.;
      for (int k = 0; k < n1; ++k)
        y[i * n2 + j] += a[k * n2 + i] * a[k * n2 + j];
    }

    // calculate A^T.b
    b[i] = 0.;
    for (int j = 0; j < n1; ++j) b[i] += a[j * n2 + i] * c[j];
  }

  // calculate (A^T.A)^{-1}.(A^T.b)
  int *indx = (int *)malloc(n2 * sizeof(int));
  ludcmp(y, indx, n2);
  lubksb(b, y, indx, n2);

  free(c);
  free(indx);
  free(y);
}
