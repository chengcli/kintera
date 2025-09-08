#pragma once

// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace kintera {

template <typename T>
DISPATCH_MACRO void swap_cols(T* M, int n, int c1, int c2) {
  if (c1 == c2) return;
  for (int i = 0; i < n; ++i) {
    T tmp = M[i * n + c1];
    M[i * n + c2] = (M[i * n + c1] = M[i * n + c2],
                     tmp);  // no-op trick won't work; do normal swap
  }
}

template <typename T>
DISPATCH_MACRO void swap_cols_safe(T* M, int n, int c1, int c2) {
  if (c1 == c2) return;
  for (int i = 0; i < n; ++i) {
    T tmp = M[i * n + c1];
    M[i * n + c1] = M[i * n + c2];
    M[i * n + c2] = tmp;
  }
}

template <typename T>
DISPATCH_MACRO void swap_vals(T* a, int i, int j) {
  T t = a[i];
  a[i] = a[j];
  a[j] = t;
}

template <typename T>
DISPATCH_MACRO void swap_rows(T* A, int m, int n, int r1, int r2) {
  if (r1 == r2) return;
  for (int j = 0; j < n; ++j) {
    T tmp = A[r1 * n + j];
    A[r1 * n + j] = A[r2 * n + j];
    A[r2 * n + j] = tmp;
  }
}

}  // namespace kintera
