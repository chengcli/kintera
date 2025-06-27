#pragma once

// C/C++
#include <cstdio>

// base
#include <configure.h>

namespace kintera {

/*!
 * \brief matrix-matrix dot product: a.b
 * \param[out] r[0..n1*n3-1] output matrix in row-major sequential storage
 * \param[in] a[0..n1*n2-1] row-major sequential storage of n1 x n2 matrix
 * \param[in] b[0..n2*n3-1] row-major sequential storage of n2 x n3 matrix
 * \param[in] n1 number of rows in matrix a
 * \param[in] n2 number of columns in matrix a (and rows in matrix b)
 * \param[in] n3 number of columns in matrix b
 */
template <typename T>
DISPATCH_MACRO void mmdot(T *r, T const *a, T const *b, int n1, int n2,
                          int n3) {
  // Check if r, a, and b are not the same
  if (r == a || r == b || a == b) {
    printf("Error: r, a, and b must be distinct pointers.\n");
    return;
  }

  // Initialize the result matrix to zero
  for (int i = 0; i < n1 * n3; ++i) {
    r[i] = 0.0;
  }

  // Perform matrix multiplication
  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < n3; ++j) {
      for (int k = 0; k < n2; ++k) {
        r[i * n3 + j] += a[i * n2 + k] * b[k * n3 + j];
      }
    }
  }
}

}  // namespace kintera
