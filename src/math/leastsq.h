#pragma once

// C/C++
#include <cstdlib>
#include <cstring>

// base
#include <configure.h>

// math
#include "lubksb.h"
#include "ludcmp.h"

namespace kintera {

/*!
 * \brief solve least square problem min ||A.x - b||
 *
 * \param[in,out] b[0..n1-1] right-hand-side vector and output. Input dimension
 * is n1, output dimension is n2, requiring n1 >= n2
 * \param[in] a[0..n1*n2-1] row-major input matrix, A
 * \param[in] n1 number of rows in matrix
 * \param[in] n2 number of columns in matrix
 */
template <typename T>
DISPATCH_MACRO void leastsq(T *b, T const *a, int n1, int n2) {
  T *c = (T *)malloc(n1 * sizeof(T));
  memcpy(c, b, n1 * sizeof(T));

  T *y = (T *)malloc(n2 * n2 * sizeof(T));

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

}  // namespace kintera
