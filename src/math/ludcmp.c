#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define A(i, j) a[(i) * n + (j)]

int ludcmp(double *a, int *indx, int n) {
  int i, imax, j, k, d;
  double big, dum, sum, temp;
  double *vv = (double *)malloc(n * sizeof(double));

  d = 1;
  for (i = 0; i < n; i++) {
    big = 0.0;
    for (j = 0; j < n; j++)
      if ((temp = fabs(A(i, j))) > big) big = temp;
    if (big == 0.0) {
      fprintf(stderr, "Singular matrix in routine ludcmp");
      exit(1);
    }
    vv[i] = 1.0 / big;
  }
  for (j = 0; j < n; j++) {
    for (i = 0; i < j; i++) {
      sum = A(i, j);
      for (k = 0; k < i; k++) sum -= A(i, k) * A(k, j);
      A(i, j) = sum;
    }
    big = 0.0;
    for (i = j; i < n; i++) {
      sum = A(i, j);
      for (k = 0; k < j; k++) sum -= A(i, k) * A(k, j);
      A(i, j) = sum;
      if ((dum = vv[i] * fabs(sum)) >= big) {
        big = dum;
        imax = i;
      }
    }
    if (j != imax) {
      for (k = 0; k < n; k++) {
        dum = A(imax, k);
        A(imax, k) = A(j, k);
        A(j, k) = dum;
      }
      d = -d;
      vv[imax] = vv[j];
    }
    indx[j] = imax;
    if (j != n - 1) {
      dum = (1.0 / A(j, j));
      for (i = j + 1; i < n; i++) A(i, j) *= dum;
    }
  }
  free(vv);

  return d;
}

#undef A
