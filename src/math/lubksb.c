void lubksb(double *b, double const *a, int const *indx, int n) {
  int i, ii = 0, ip, j;
  double sum;

  for (i = 0; i < n; i++) {
    ip = indx[i];
    sum = b[ip];
    b[ip] = b[i];
    if (ii)
      for (j = ii - 1; j < i; j++) sum -= a[i * n + j] * b[j];
    else if (sum)
      ii = i + 1;
    b[i] = sum;
  }
  for (i = n - 1; i >= 0; i--) {
    sum = b[i];
    for (j = i + 1; j < n; j++) sum -= a[i * n + j] * b[j];
    b[i] = sum / a[i * n + i];
  }
}
