// kintera
#include <kintera/math/core.h>
#include <kintera/math/leastsq.h>
#include <kintera/math/leastsq_kkt.h>
#include <kintera/math/lubksb.h>
#include <kintera/math/ludcmp.h>
#include <kintera/math/luminv.h>
#include <kintera/math/psolve.h>
#include <kintera/math/rank.h>
#include <kintera/math/solve_block_system.h>

using namespace kintera;

void solve_block_system() {
  printf("Testing solve_block_system...\n");

  int n = 2;  // size of A
  int m = 1;  // rows of B

  double A[4] = {4, 1, 1, 3};  // 2×2 SPD
  double B[2] = {1, 2};        // 1×2
  double C[2] = {1, 2};
  double D[1] = {3};

  double A_inv[4];
  luminv(A_inv, A, n);

  solve_block_system(A_inv, B, C, D, n, m);

  printf("x = [%f, %f]\n", C[0], C[1]);
  printf("y = [%f]\n", D[0]);

  // examine the result
  double AB[9] = {4, 1, 1,  //
                  1, 3, 2,  //
                  1, 2, 0};
  double CD[3] = {C[0], C[1], D[0]};

  double res[3];
  mvdot(res, AB, CD, 3, 3);
  printf("res = [%f, %f, %f]\n", res[0], res[1], res[2]);

  int indx[3];
  ludcmp(AB, indx, 3);
  double res2[3] = {1, 2, 3};
  lubksb(res2, AB, indx, 3);
  printf("check = [%f, %f, %f]\n", res2[0], res2[1], res2[2]);
}

// test luminv
void test_luminv() {
  printf("Testing luminv...\n");
  double a[4] = {1.0, 2.0, 3.0, 4.0};
  int n = 2;
  double y[4];

  printf("matrix A= \n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", a[i * n + j]);
    }
    printf("\n");
  }

  luminv(y, a, n);
  printf("Inverse matrix Y= \n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", y[i * n + j]);
    }
    printf("\n");
  }
  printf("\n");
}

// test leastsq
void test_leastsq() {
  printf("Testing leastsq...\n");
  double a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double b[3] = {7.0, 8.0, 9.0};
  int n1 = 3;
  int n2 = 2;

  printf("Matrix A= \n");
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", a[i * n2 + j]);
    }
    printf("\n");
  }

  printf("Vector B= \n");
  for (int i = 0; i < n1; i++) {
    printf("%f\n", b[i]);
  }

  leastsq(b, a, n1, n2);
  printf("Least squares solution: \n");
  for (int i = 0; i < n2; i++) {
    printf("%f\n", b[i]);
  }
  printf("\n");
}

void test_leastsq_kkt() {
  printf("Testing leastsq_kkt...\n");
  double a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double c[4] = {7.0, 8.0, 9.0, 10.0};
  double d[2] = {11., 15.};
  double b[3] = {7.0, 8.0, 9.0};
  int n1 = 3;
  int n2 = 2;
  int n3 = 2;
  int neq = 1;

  printf("Matrix A= \n");
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", a[i * n2 + j]);
    }
    printf("\n");
  }

  printf("Vector B= \n");
  for (int i = 0; i < n1; i++) {
    printf("%f\n", b[i]);
  }

  printf("Matrix C= \n");
  for (int i = 0; i < n3; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", c[i * n2 + j]);
    }
    printf("\n");
  }
  printf("neq = %d\n", neq);

  printf("Vector D= \n");
  for (int i = 0; i < n3; i++) {
    printf("%f\n", d[i]);
  }

  int max_iter = 20;
  int err = leastsq_kkt(b, a, c, d, n1, n2, n3, neq, &max_iter);
  if (err != 0) {
    fprintf(stderr, "Error in leastsq_kkt: %d\n", err);
  }

  printf("Constrained least squares solution: \n");
  for (int i = 0; i < n2; i++) {
    printf("%f\n", b[i]);
  }
  printf("Number of iterations: %d\n", max_iter);
  printf("\n");
}

void test_leastsq_kkt_large() {
  // read X matrix from file, "X.txt"
  // data size: 184x15
  printf("Testing leastsq_kkt_large...\n");

  int n1 = 184;  // number of rows
  int n2 = 15;   // number of columns

  double *a = (double *)malloc(n1 * n2 * sizeof(double));
  FILE *file_a = fopen("X.txt", "r");
  if (file_a == NULL) {
    fprintf(stderr, "Could not open file X.txt\n");
    return;
  }
  for (int i = 0; i < 2760; i++) {
    if (fscanf(file_a, "%lf", &a[i]) != 1) {
      fprintf(stderr, "Error reading data from file\n");
      fclose(file_a);
      return;
    }
  }
  fclose(file_a);

  double *b = (double *)malloc(n1 * sizeof(double));
  FILE *file_b = fopen("Y.txt", "r");
  if (file_b == NULL) {
    fprintf(stderr, "Could not open file b.txt\n");
    return;
  }
  for (int i = 0; i < 184; i++) {
    if (fscanf(file_b, "%lf", &b[i]) != 1) {
      fprintf(stderr, "Error reading data from file b.txt\n");
      fclose(file_b);
      return;
    }
  }
  fclose(file_b);

  // print the first 5 rows of matrix a
  printf("Matrix A :\n");
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", a[i * n2 + j]);
    }
    printf("\n");
  }
  printf("...\n");
  for (int i = n1 - 5; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", a[i * n2 + j]);
    }
    printf("\n");
  }

  // print the first 5 rows of vector b
  printf("Vector B :\n");
  for (int i = 0; i < 5; i++) {
    printf("%f\n", b[i]);
  }
  printf("...\n");
  for (int i = n1 - 5; i < n1; i++) {
    printf("%f\n", b[i]);
  }

  int neq = 1;        // number of equality constraints
  int n3 = neq + n2;  // number of constraints
  double *c = (double *)malloc(n3 * n2 * sizeof(double));

  // first row: add up to 1.0
  for (int i = 0; i < n2; i++) {
    c[i] = 1.0;  // equal weights
  }

  // negative identity matrix for the rest of the constraints
  for (int i = neq; i < n3; i++) {
    for (int j = 0; j < n2; j++) {
      if (i - neq == j) {
        c[i * n2 + j] = -1.0;  // diagonal elements
      } else {
        c[i * n2 + j] = 0.0;  // off-diagonal elements
      }
    }
  }

  double *d = (double *)malloc(n3 * sizeof(double));
  // first constraint: sum to 1.0
  d[0] = 1.0;
  // other constraints: set to 0.0
  for (int i = neq; i < n3; i++) {
    d[i] = 0.0;
  }

  // print the constraint matrix c
  printf("Constraint Matrix C (first 5 rows):\n");
  for (int i = 0; i < n3; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", c[i * n2 + j]);
    }
    printf("\n");
  }

  // print the constraint vector d
  printf("Constraint Vector D:\n");
  for (int i = 0; i < n3; i++) {
    printf("%f\n", d[i]);
  }

  // call leastsq_kkt
  // copy b
  double *b0 = (double *)malloc(n1 * sizeof(double));
  memcpy(b0, b, n1 * sizeof(double));

  // test solution
  double b1[15] = {0.0784, 0.1049, 0.0383, 0.1059, 0.1002,
                   0.0880, 0.0682, 0.0139, 0.0139, 0.0139,
                   0.0491, 0.0699, 0.0733, 0.0139, 0.1680};

  int max_iter = 20;
  int err = leastsq_kkt(b, a, c, d, n1, n2, n3, neq, &max_iter);
  if (err != 0) {
    fprintf(stderr, "Error in leastsq_kkt: %d\n", err);
  }

  printf("Constrained least squares solution: \n");
  for (int i = 0; i < n2; i++) {
    printf("%f\n", b[i]);
  }
  printf("Number of iterations: %d\n", max_iter);

  double cost = 0.0, cost1 = 0.0;
  for (int i = 0; i < n1; i++) {
    double diff = b0[i];
    double diff1 = b0[i];
    for (int j = 0; j < n2; j++) {
      diff -= a[i * n2 + j] * b[j];
      diff1 -= a[i * n2 + j] * b1[j];
    }
    cost += diff * diff;
    cost1 += diff1 * diff1;
  }
  printf("Cost function =  %f\n", cost);
  printf("Cost function1 =  %f\n", cost1);

  free(a);
  free(b);
  free(c);
  free(d);
  free(b0);
  printf("\n");
}

void test_psolve() {
  printf("Testing psolve...\n");
  int n = 2;

  double A[4] = {3.e-6, 0.0, 0.0, 0.0};
  double b[2] = {0.008312, 0.};

  psolve(b, A, n);

  // Print solution vector x
  for (int i = 0; i < n; ++i) {
    printf("%.12g%c", b[i], (i + 1 < n) ? ' ' : '\n');
  }
}

void test_rank() {
  int m, n;
  double A[4] = {3.e-6, 0.0, 0.0, 0.0};

  int r = matrix_rank(A, 2, 2);
  printf("%d\n", r);
}

int main(int argc, char **argv) {
  test_luminv();
  test_leastsq();
  solve_block_system();
  test_leastsq_kkt();
  test_leastsq_kkt_large();
  test_psolve();
  test_rank();
}
