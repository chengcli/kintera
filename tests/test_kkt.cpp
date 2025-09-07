// kintera
#include <kintera/math/lubksb.h>
#include <kintera/math/ludcmp.h>
#include <kintera/math/mvdot.h>
#include <kintera/math/solve_block_system.h>

using namespace kintera;

void solve_block_system() {
  int n = 2;  // size of A
  int m = 1;  // rows of B

  double A[4] = {4, 1, 1, 3};  // 2×2 SPD
  double B[2] = {1, 2};        // 1×2
  double C[2] = {1, 2};
  double D[1] = {3};

  solve_block_system(A, B, C, D, n, m);

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

int main(int argc, char **argv) { solve_block_system(); }
