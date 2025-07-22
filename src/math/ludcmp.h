#pragma once

// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>

// base
#include <configure.h>

#define X(i, j) x[(i) * n + (j)]

namespace kintera {

/*!
 * \brief LU decomposition
 *
 * Given a row-major sequential storage of matrix a[0..n*n-1],
 * this routine replaces it by the LU decomposition of a rowwise permutation of
 * itself. This routine is used in combination with lubksb to solve linear
 * equationsor invert a matrix. Adapted from Numerical Recipes in C, 2nd Ed.,
 * p. 46.
 *
 * \param[in,out] a[0..n*n-1] row-major input matrix, output LU decomposition
 * \param[out] indx[0..n-1] vector that records the row permutation effected by
 * the partial pivoting. Outputs as +/- 1 depending on whether the number of row
 * interchanges was even or odd, respectively.
 * \param[in] n size of matrix
 */
template <typename T>
DISPATCH_MACRO int ludcmp(T *x, int *indx, int n) {
  int i, imax, j, k, d;
  T big, dum, sum, temp;
  T *vv = (T *)malloc(n * sizeof(T));

  for (i = 0; i < n; i++) indx[i] = i;

  d = 1;
  for (i = 0; i < n; i++) {
    big = 0.0;
    for (j = 0; j < n; j++)
      if ((temp = fabs(X(i, j))) > big) big = temp;
    if (big == 0.0) {
      // printf("Singular matrix in routine ludcmp\n");
      free(vv);
      return 1;
    }
    vv[i] = 1.0 / big;
  }
  for (j = 0; j < n; j++) {
    for (i = 0; i < j; i++) {
      sum = X(i, j);
      for (k = 0; k < i; k++) sum -= X(i, k) * X(k, j);
      X(i, j) = sum;
    }
    big = 0.0;
    imax = j;
    for (i = j; i < n; i++) {
      sum = X(i, j);
      for (k = 0; k < j; k++) sum -= X(i, k) * X(k, j);
      X(i, j) = sum;
      if ((dum = vv[i] * fabs(sum)) >= big) {
        big = dum;
        imax = i;
      }
    }
    if (j != imax) {
      for (k = 0; k < n; k++) {
        dum = X(imax, k);
        X(imax, k) = X(j, k);
        X(j, k) = dum;
      }
      d = -d;
      vv[imax] = vv[j];
    }
    indx[j] = imax;
    if (j != n - 1) {
      dum = (1.0 / X(j, j));
      for (i = j + 1; i < n; i++) X(i, j) *= dum;
    }
  }
  free(vv);

  return d;
}

}  // namespace kintera

#undef X
