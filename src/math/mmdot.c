#include <stdio.h>

#include "linalg.h"

void mmdot(double *r, double const *a, double const *b, int n1, int n2,
           int n3) {
  // Check if r, a, and b are not the same
  if (r == a || r == b || a == b) {
    fprintf(stderr, "Error: r, a, and b must be distinct pointers.\n");
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
