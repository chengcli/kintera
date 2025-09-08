#pragma once

// C/C++
#include <cstdio>
#include <cstdlib>

// base
#include <configure.h>

// math
#include "luminv.h"
#include "solve_block_system.h"

namespace kintera {

/*!
 * \brief solve constrained least square problem: min ||A.x - b||, s.t. C.x <= d
 *
 * This subroutine solves the constrained least square problem using the active
 * set method based on the KKT conditions. The first `neq` rows of the
 * constraint matrix `C` are treated as equality constraints, while the
 * remaining rows are treated as inequality constraints.
 *
 * \param[in,out] b[0..n1-1]    right-hand-side vector and output. Input
 *                              dimension is n1, output dimension is n2,
 *                              requiring n1 >= n2
 * \param[in] a[0..n1*n2-1]     row-major input matrix, A
 * \param[in] c[0..n3*n2-1]     row-major constraint matrix, C
 * \param[in] d[0..n3-1]        right-hand-side constraint vector, d
 * \param[in] n1                number of rows in matrix A
 * \param[in] n2                number of columns in matrix A
 * \param[in] n3                number of rows in matrix C
 * \param[in] neq               number of equality constraints, 0 <= neq <= n3
 * \param[in,out] max_iter      in: maximum number of iterations to perform,
 *                              out: number of iterations actually performed
 * \param[in] work              workspace if not null, otherwise allocated
 *                              internally.
 *
 * \return 0 on success, 1 on invalid input (e.g., neq < 0 or neq > n3),
 *         2 on failure (max_iter reached without convergence).
 */
template <typename T>
DISPATCH_MACRO int leastsq_kkt(T *b, T const *a, T const *c, T const *d, int n1,
                               int n2, int n3, int neq, int *max_iter,
                               char *work = nullptr) {
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
  T *ata, *atb, *ata_inv, *rhs, *eval;
  int *ct_indx;
  int nmax = n2 > n3 ? n2 : n3;

  if (work == nullptr) {
    ata = (T *)malloc(nmax * n2 * sizeof(T));
    atb = (T *)malloc(n2 * sizeof(T));
    ata_inv = (T *)malloc(n2 * n2 * sizeof(T));
    rhs = (T *)malloc((n2 + n3) * sizeof(T));

    // evaluation of constraints
    eval = (T *)malloc(n3 * sizeof(T));

    // index for the active set
    ct_indx = (int *)malloc(n3 * sizeof(int));
  } else {
    ata = alloc_from<T>(work, nmax * n2);
    atb = alloc_from<T>(work, n2);
    ata_inv = alloc_from<T>(work, n2 * n2);
    rhs = alloc_from<T>(work, n2 + n3);
    eval = alloc_from<T>(work, n3);
    ct_indx = alloc_from<int>(work, n3);
  }

  // populate A^T.A
  for (int i = 0; i < n2; ++i) {
    for (int j = 0; j < n2; ++j) {
      ata[i * n2 + j] = 0.0;
      for (int k = 0; k < n1; ++k) {
        ata[i * n2 + j] += a[k * n2 + i] * a[k * n2 + j];
      }
    }
  }

  // invert A^T.A
  luminv(ata_inv, ata, n2, work);

  /* print ata_inv
  printf("ata_inv = \n");
  for (int i = 0; i < n2; ++i) {
    for (int j = 0; j < n2; ++j) {
      printf("%f ", ata_inv[i * n2 + j]);
    }
    printf("\n");
  }*/

  // populate A^T.b
  for (int i = 0; i < n2; ++i) {
    atb[i] = 0.0;
    for (int j = 0; j < n1; ++j) {
      atb[i] += a[j * n2 + i] * b[j];
    }
  }

  for (int i = 0; i < n3; ++i) {
    ct_indx[i] = i;
  }

  memset(ata, 0, nmax * n2 * sizeof(T));
  T *c_act = ata;  // reuse ata for c_act

  int nactive = neq;
  int iter = 0;

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
    int nactive0 = nactive;

    // populate B
    for (int i = 0; i < nactive; ++i)
      for (int j = 0; j < n2; ++j) {
        c_act[i * n2 + j] = c[ct_indx[i] * n2 + j];
      }

    /* print B
    printf("c_act = \n");
    for (int i = 0; i < nactive; ++i) {
      for (int j = 0; j < n2; ++j) {
        printf("%f ", c_act[i * n2 + j]);
      }
      printf("\n");
    }*/

    // populate c (upper part)
    for (int i = 0; i < n2; ++i) {
      rhs[i] = atb[i];
    }

    // populate d (lower part)
    for (int i = 0; i < nactive; ++i) {
      rhs[n2 + i] = d[ct_indx[i]];
    }

    // solve the KKT system using block elimination
    solve_block_system(ata_inv, c_act, rhs, rhs + n2, n2, nactive, work);

    // evaluate the inactive constraints
    for (int i = nactive; i < n3; ++i) {
      int k = ct_indx[i];
      eval[k] = 0.;
      for (int j = 0; j < n2; ++j) {
        eval[k] += c[k * n2 + j] * rhs[j];
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
      if (eval[k] > d[k]) {
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
    if (nactive == nactive0) {
      // no change in active set, we are done
      break;
    }
  }

  // copy to output vector b
  for (int i = 0; i < n2; ++i) {
    b[i] = rhs[i];
  }

  if (work == nullptr) {
    free(ata);
    free(atb);
    free(ata_inv);
    free(rhs);
    free(eval);
    free(ct_indx);
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
