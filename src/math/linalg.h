#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief vector-vector dot product: a.b
 * \param[in] a[0..n-1] first vector
 * \param[in] b[0..n-1] second vector
 */
double vvdot(double const *a, double const *b, int n);

/*!
 * \brief matrix-vector dot product: m.v
 * \param[out] r[0..n1-1] output vector
 * \param[in] m[0..n1*n2 -1] row-major sequential storage of n1 x n2 matrix
 * \param[in] v[0..n2-1] input vector
 * \param[in] n1 number of rows in matrix
 * \param[in] n2 number of columns in matrix
 */
void mvdot(double *r, double const *m, double const *v, int n1, int n2);

/*!
 * \brief matrix-matrix dot product: a.b
 * \param[out] r[0..n1*n3-1] output matrix in row-major sequential storage
 * \param[in] a[0..n1*n2-1] row-major sequential storage of n1 x n2 matrix
 * \param[in] b[0..n2*n3-1] row-major sequential storage of n2 x n3 matrix
 * \param[in] n1 number of rows in matrix a
 * \param[in] n2 number of columns in matrix a (and rows in matrix b)
 * \param[in] n3 number of columns in matrix b
 */
void mmdot(double *r, double const *a, double const *b, int n1, int n2, int n3);

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
int ludcmp(double *a, int *indx, int n);

/*!
 * \brief Solves the set of n linear equations A.X = B.
 *
 * This routine takes into account the possibility that b will begin with many
 * zero elements, so it is efficient for use in matrix inversion.
 * Adapted from Numerical Recipes in C, 2nd Ed., p. 47.
 *
 * \param[in,out] b[0..n-1] input as the right-hand side vector B, and returns
 * with the solution vector X.
 * \param[in] a[0..n*n-1] row-major input matrix, not as the matrix A but rather
 * as its LU decomposition, determined by the routine ludcmp.
 * \param[in] indx[0..n-1] input the permutation vector returned by ludcmp.
 * \param[in] n size of matrix
 */
void lubksb(double *b, double const *a, int const *indx, int n);

/*!
 * \brief invert a matrix, Y = A^{-1}
 *
 * Using the backsubstitution routines,
 * it is completely straightforward to find the inverse of a matrix
 * column by column.
 *
 * \param[out] y[0..n*n-1] row-major output matrix, Y = A^{-1}
 * \param[in] a[0..n*n-1] row-major LU decomposition matrix
 * \param[in] indx[0..n-1] the permutation vector returned by ludcmp.
 * \param[in] n size of matrix
 */
void luminv(double *y, double const *a, int const *indx, int n);

/*!
 * \brief solve least square problem min ||A.x - b||
 *
 * \param[in,out] b[0..n1-1] right-hand-side vector and output. Input dimension
 * is n1, output dimension is n2, requiring n1 >= n2
 * \param[in] a[0..n1*n2-1] row-major input matrix, A
 * \param[in] n1 number of rows in matrix
 * \param[in] n2 number of columns in matrix
 */
void leastsq(double *b, double const *a, int n1, int n2);

/*!
 * \brief solve constrained least square problem: min ||A.x - b||, s.t. C.x <= d
 *
 * This subroutine solves the constrained least square problem using the active
 * set method based on the KKT conditions. The first `neq` rows of the
 * constraint matrix `C` are treated as equality constraints, while the
 * remaining rows are treated as inequality constraints.
 *
 * \param[in,out] b[0..n1-1] right-hand-side vector and output. Input dimension
 * is n1, output dimension is n2, requiring n1 >= n2
 * \param[in] a[0..n1*n2-1] row-major input matrix, A
 * \param[in] c[0..n3*n2-1] row-major constraint matrix, C
 * \param[in] d[0..n3-1] right-hand-side constraint vector, d
 * \param[in] n1 number of rows in matrix A
 * \param[in] n2 number of columns in matrix A
 * \param[in] n3 number of rows in matrix C
 * \param[in] neq number of equality constraints, 0 <= neq <= n3
 * \param[in,out] max_iter in: maximum number of iterations to perform, out:
 * number of iterations actually performed
 * \return 0 on success, 1 on invalid input (e.g., neq < 0 or neq > n3),
 *         2 on failure (max_iter reached without convergence).
 */
int leastsq_kkt(double *b, double const *a, double const *c, double const *d,
                int n1, int n2, int n3, int neq, int *max_iter);

#ifdef __cplusplus
} /* extern "C" */
#endif
