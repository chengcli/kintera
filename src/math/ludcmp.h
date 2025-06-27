#pragma once

// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>

// base
#include <configure.h>

#define A(i, j) a[(i) * n + (j)]

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
DISPATCH_MACRO int ludcmp(T *a, int *indx, int n) {
  int i, imax, j, k, d;
  T big, dum, sum, temp;
  T *vv = (T *)malloc(n * sizeof(T));

  d = 1;
  for (i = 0; i < n; i++) {
    big = 0.0;
    for (j = 0; j < n; j++)
      if ((temp = fabs(A(i, j))) > big) big = temp;
    if (big == 0.0) {
      printf("Singular matrix in routine ludcmp");
      free(vv);
      return 1;
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

}  // namespace kintera

#undef A
