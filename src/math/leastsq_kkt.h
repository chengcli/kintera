#pragma once

// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

// base
#include <configure.h>

// math
#include "lubksb.h"
#include "ludcmp.h"
#include "psolve.h"

#define A(i, j) a[(i) * n2 + (j)]
#define ATA(i, j) ata[(i) * n2 + (j)]
#define AUG(i, j) aug[(i) * (n2 + nact) + (j)]
#define C(i, j) c[(i) * n2 + (j)]

namespace kintera {

// Compute bitmask hash for a set of integers [0..n-1]
DISPATCH_MACRO inline uint64_t hash_set(const int* arr, int size, int n) {
  uint64_t mask = 0;
  for (int i = 0; i < size; i++) {
    int x = arr[i];
    if (x >= 0 && x < n) {
      mask |= (1ULL << x);
    }
  }
  return mask;
}

template <typename T>
DISPATCH_MACRO T kkt_column_scale(T objective_scale, T column_norm) {
  return column_norm > 0. ? objective_scale / column_norm : 1.;
}

template <typename T>
DISPATCH_MACRO T kkt_row_scale(T const* c, T const* d,
                              T const* column_norm, T objective_scale, int n2,
                              int row) {
  T scale = fabs(d[row]);
  for (int j = 0; j < n2; ++j) {
    T value = fabs(c[row * n2 + j] *
                   kkt_column_scale(objective_scale, column_norm[j]));
    if (value > scale) scale = value;
  }
  return scale > 0. ? scale : 1.;
}

template <typename T>
DISPATCH_MACRO void populate_aug(T* aug, T const* ata, T const* c, int n2,
                                 int nact, int const* ct_indx, float reg = 0.) {
  // populate A^T.A (upper left block)
  for (int i = 0; i < n2; ++i) {
    for (int j = 0; j < n2; ++j) {
      AUG(i, j) = ATA(i, j);
    }
  }

  // populate C (lower left block)
  for (int i = 0; i < nact; ++i) {
    for (int j = 0; j < n2; ++j) {
      AUG(n2 + i, j) = C(ct_indx[i], j);
    }
  }

  // populate C^T (upper right block)
  for (int i = 0; i < n2; ++i) {
    for (int j = 0; j < nact; ++j) {
      AUG(i, n2 + j) = C(ct_indx[j], i);
    }
  }

  // zero (lower right block)
  for (int i = 0; i < nact; ++i) {
    for (int j = 0; j < nact; ++j) {
      AUG(n2 + i, n2 + j) = 0.0;
    }
    // add a small diagonal perturbation to improve numerical stability
    AUG(n2 + i, n2 + i) = reg;
  }
}

template <typename T>
DISPATCH_MACRO void populate_rhs(T* rhs, T const* atb, T const* d, int n2,
                                 int nact, int const* ct_indx) {
  // populate A^T.b (upper part)
  for (int i = 0; i < n2; ++i) {
    rhs[i] = atb[i];
  }

  // populate d (lower part)
  for (int i = 0; i < nact; ++i) {
    rhs[n2 + i] = d[ct_indx[i]];
  }
}

/*!
 * \brief solve constrained least square problem: min ||A.x - b||, s.t. C.x <= d
 *
 * This subroutine solves the constrained least square problem using the active
 * set method based on the KKT conditions. The first `neq` rows of the
 * constraint matrix `C` are treated as equality constraints, while the
 * remaining rows are treated as inequality constraints. Objective columns and
 * constraint rows are scaled internally; the returned solution is unscaled.
 *
 * \param[in,out] b[0..n1-1]    right-hand-side vector and output. Input
 *                              dimension is n1, output dimension is n2,
 * requiring n1 >= n2
 * \param[in] a[0..n1*n2-1]     row-major input matrix, A
 * \param[in] c[0..n3*n2-1]     row-major constraint matrix, C
 * \param[in] d[0..n3-1]        right-hand-side constraint vector, d
 * \param[in] n1                number of rows in matrix A
 * \param[in] n2                number of columns in matrix A
 * \param[in] n3                number of rows in matrix C
 * \param[in] neq               number of equality constraints, 0 <= neq <= n3
 * \param[in,out] max_iter      in: maximum number of iterations to perform,
 *                              out: number of iterations actually performed
 * \param[in] reg               diagonal perturbation in scaled KKT units;
 *                              zero uses regularization only if LU fails
 * \param[in] work              workspace if not null, otherwise allocated
 *                              internally.
 *
 * \return 0 on success, 1 on invalid input (e.g., neq < 0 or neq > n3),
 *         2 on failure (max_iter reached without convergence), or 3 if the
 *         KKT system remains singular or has an inconsistent zero row.
 */
template <typename T>
DISPATCH_MACRO int leastsq_kkt(T* b, T const* a, T const* c, T const* d, int n1,
                               int n2, int n3, int neq, int* max_iter,
                               float reg = 0., char* work = nullptr) {
  // check if n1 > 0, n2 > 0, n3 >= 0
  if (n1 <= 0 || n2 <= 0 || n3 < 0 || n1 < n2) {
    printf(
        "Error: n1 and n2 must be positive integers and n3 >= 0, n1 >= n2.\n");
    return 1;  // invalid input
  }

  // check if 0 <= neq <= n3
  if (neq < 0 || neq > n3) {
    printf("Error: neq must be non-negative.\n");
    return 1;  // invalid input
  }

  // Allocate memory for the augmented matrix and right-hand side vector
  int size = n2 + n3;
  T *aug, *ata, *column_norm, *rhs, *eval;
  int *ct_indx, *lu_indx, *skip_row;

  if (work == nullptr) {
    aug = (T*)malloc(size * size * sizeof(T));
    ata = (T*)malloc(n2 * n2 * sizeof(T));
    column_norm = (T*)malloc(n2 * sizeof(T));
    rhs = (T*)malloc(size * sizeof(T));

    // evaluation of constraints
    eval = (T*)malloc(n3 * sizeof(T));

    // index for the active set
    ct_indx = (int*)malloc(n3 * sizeof(int));

    // index array for the LU decomposition
    lu_indx = (int*)malloc(size * sizeof(int));

    // row indices to skip
    skip_row = (int*)malloc(size * sizeof(int));
  } else {
    aug = alloc_from<T>(work, size * size);
    ata = alloc_from<T>(work, n2 * n2);
    column_norm = alloc_from<T>(work, n2);
    rhs = alloc_from<T>(work, size);
    eval = alloc_from<T>(work, n3);
    ct_indx = alloc_from<int>(work, n3);
    lu_indx = alloc_from<int>(work, size);
    skip_row = alloc_from<int>(work, size);
  }

  T objective_scale = 1.;
  for (int i = 0; i < n1; ++i) {
    T value = fabs(b[i]);
    if (value > objective_scale) objective_scale = value;
  }

  for (int j = 0; j < n2; ++j) {
    column_norm[j] = 0.;
    for (int i = 0; i < n1; ++i) {
      T value = fabs(A(i, j));
      if (value > column_norm[j]) column_norm[j] = value;
    }
  }

  // populate A^T.A
  for (int i = 0; i < n2; ++i) {
    for (int j = 0; j < n2; ++j) {
      ATA(i, j) = 0.0;
      for (int k = 0; k < n1; ++k) {
        T left = column_norm[i] > 0. ? A(k, i) / column_norm[i] : 0.;
        T right = column_norm[j] > 0. ? A(k, j) / column_norm[j] : 0.;
        ATA(i, j) += left * right;
      }
    }
  }

  for (int i = 0; i < n3; ++i) {
    ct_indx[i] = i;
  }

  int nactive = neq;
  int iter = 0;
  int status = 0;
  T fallback_reg = sizeof(T) == sizeof(float) ? 1.e-5 : 1.e-10;
  T pivot_tolerance = 10. * std::numeric_limits<T>::epsilon();

  while (iter++ < *max_iter) {
    /*printf("kkt iter = %d, nactive = %d\n", iter, nactive);
    printf("ct_indx = ");
    for (int i = 0; i < neq; ++i) {
      printf("%d ", ct_indx[i]);
    }
    printf("| ");
    for (int i = neq; i < nactive; ++i) {
      printf("%d ", ct_indx[i]);
    }
    printf("| ");
    for (int i = nactive; i < n3; ++i) {
      printf("%d ", ct_indx[i]);
    }
    printf("\n");*/
    uint64_t hash0 = hash_set(ct_indx, nactive, n3);

    int nact = nactive;
    bool solved = false;
    for (int attempt = 0; attempt < 2; ++attempt) {
      T primal_reg = attempt == 0 ? 0. : fallback_reg;
      T dual_reg = attempt == 0 ? reg : (reg == 0. ? -fallback_reg : reg);
      for (int i = 0; i < n2; ++i) {
        rhs[i] = 0.;
        for (int k = 0; k < n1; ++k) {
          T value = column_norm[i] > 0. ? A(k, i) / column_norm[i] : 0.;
          rhs[i] += value * (b[k] / objective_scale);
        }
        for (int j = 0; j < n2; ++j)
          AUG(i, j) = ATA(i, j) + (i == j ? primal_reg : 0.);
      }
      for (int i = 0; i < nactive; ++i) {
        int row = ct_indx[i];
        T row_scale = kkt_row_scale(c, d, column_norm, objective_scale, n2, row);
        for (int j = 0; j < n2; ++j) {
          T value = C(row, j) *
                        kkt_column_scale(objective_scale, column_norm[j]) /
                    row_scale;
          AUG(n2 + i, j) = value;
          AUG(j, n2 + i) = value;
        }
        rhs[n2 + i] = d[row] / row_scale;
        for (int j = 0; j < nactive; ++j)
          AUG(n2 + i, n2 + j) = i == j ? dual_reg : 0.;
      }

      for (int i = 0; i < n2 + nactive; ++i) {
        bool all_zero = true;
        for (int j = 0; j < n2 + nactive; ++j) {
          if (aug[i * (n2 + nactive) + j] != 0.0) {
            all_zero = false;
            break;
          }
        }
        skip_row[i] = all_zero;
        if (all_zero && rhs[i] != 0.) {
          status = 3;
          break;
        }
      }
      if (status != 0) break;

      if (ludcmp(aug, lu_indx, n2 + nactive, work, skip_row,
                 pivot_tolerance) != 0) {
        lubksb(rhs, aug, lu_indx, n2 + nactive, skip_row);
        solved = true;
        break;
      }
    }
    if (!solved) {
      status = 3;
      break;
    }

    /* print aug
    printf("aug = \n");
    for (int i = 0; i < n2 + nactive; ++i) {
      for (int j = 0; j < n2 + nactive; ++j) {
        printf("%f ", aug[i * (n2 + nactive) + j]);
      }
      printf("| %f", rhs[i]);
      if (skip_row[i]) printf(" *");
      printf("\n");
    }*/

    // evaluate the inactive constraints
    for (int i = nactive; i < n3; ++i) {
      int k = ct_indx[i];
      T row_scale = kkt_row_scale(c, d, column_norm, objective_scale, n2, k);
      eval[k] = -d[k] / row_scale;
      for (int j = 0; j < n2; ++j) {
        eval[k] += C(k, j) *
                   kkt_column_scale(objective_scale, column_norm[j]) /
                   row_scale * rhs[j];
      }
    }

    /* print solution vector (rhs)
    printf("rhs = ");
    for (int i = 0; i < n2; ++i) {
      printf("%f ", rhs[i]);
    }
    printf("| ");
    for (int i = n2; i < n2 + nactive; ++i) {
      printf("%f ", rhs[i]);
    }
    printf("\n");*/

    // remove inactive constraints (three-way swap)
    //           mu < 0
    //           |---------------->|
    //           |<----|<----------|
    //           f     :...m       :...l
    //           |     :   |       :   |
    // | * * * | * * * * | * * * * * | x
    // |-------|---------|-----------|
    // |  EQ   |   INEQ  | INACTIVE  |
    int first = neq;
    int mid = nactive;
    int last = n3;
    while (first < mid) {
      if (rhs[n2 + first] < 0.0) {  // inactive constraint
        // swap with the last active constraint
        int tmp = ct_indx[first];
        ct_indx[first] = ct_indx[mid - 1];
        ct_indx[mid - 1] = ct_indx[last - 1];
        ct_indx[last - 1] = tmp;

        T val = rhs[n2 + first];
        rhs[n2 + first] = rhs[n2 + mid - 1];
        rhs[n2 + mid - 1] = val;
        --last;
        --mid;
      } else {
        ++first;
      }
    }

    /* print ct_indx after removing
    printf("ct_indx after removing = ");
    for (int i = 0; i < neq; ++i) {
      printf("%d ", ct_indx[i]);
    }
    printf("| ");
    for (int i = neq; i < nactive; ++i) {
      printf("%d ", ct_indx[i]);
    }
    printf("| ");
    for (int i = nactive; i < n3; ++i) {
      printf("%d ", ct_indx[i]);
    }
    printf("\n");*/

    // add back inactive constraints (two-way swap)
    //                     C.x <= d
    //                     |<----->|
    //                     f       : l
    //                     |       : |
    // | * * * | * * * * | * * * * * x * |
    // |-------|---------|---------------|
    // |  EQ   |   INEQ  |   INACTIVE    |
    while (first < last) {
      int k = ct_indx[first];
      if (eval[k] > 0.) {
        // add the inactive constraint back to the active set
        ++first;
      } else {
        int tmp = ct_indx[first];
        ct_indx[first] = ct_indx[last - 1];
        ct_indx[last - 1] = tmp;
        --last;
      }
    }

    nactive = first;
    uint64_t hash1 = hash_set(ct_indx, nactive, n3);
    // no change in active set, we are done
    if (hash0 == hash1) break;
  }

  if (status == 0)
    for (int i = 0; i < n2; ++i)
      b[i] = rhs[i] * kkt_column_scale(objective_scale, column_norm[i]);

  if (work == nullptr) {
    free(aug);
    free(ata);
    free(column_norm);
    free(rhs);
    free(eval);
    free(ct_indx);
    free(lu_indx);
    free(skip_row);
  }

  if (status != 0) {
    *max_iter = iter;
    return status;
  }

  if (iter >= *max_iter) {
    *max_iter = iter;
    printf("Warning: leastsq_kkt maximum number of iterations reached (%d).\n",
           *max_iter);
    return 2;  // failure to converge
  }

  *max_iter = iter;
  return 0;  // success
}

}  // namespace kintera

#undef A
#undef ATA
#undef AUG
#undef C
