#pragma once

// C/C++
#include <cstdio>

// base
#include <configure.h>

namespace kintera {

// matrix utilities (row-major)
template <typename T>
DISPATCH_MACRO void matvec(T *y, const T *A, const T *x, int n, int m) {
  // y = A x, A is n×m
  for (int i = 0; i < n; i++) {
    T sum = 0.0;
    for (int j = 0; j < m; j++) {
      sum += A[i * m + j] * x[j];
    }
    y[i] = sum;
  }
}

template <typename T>
DISPATCH_MACRO void matvec_t(T *y, const T *A, const T *x, int n, int m) {
  // y = A^T x, A is m×n
  for (int i = 0; i < n; i++) {
    T sum = 0.0;
    for (int j = 0; j < m; j++) {
      sum += A[j * n + i] * x[j];
    }
    y[i] = sum;
  }
}

}  // namespace kintera
