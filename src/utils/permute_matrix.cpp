#include <cstring>  // memcpy
#include <iostream>
#include <vector>

template <typename T>
void permute_matrix_inplace(T* gain, int const* perm, int n, char* work) {
  std::vector<bool> visited(n, false);
  std::vector<T> buffer(n);  // single row/col buffer

  // ---- Reorder rows ----
  for (int i = 0; i < n; i++) {
    if (visited[i]) continue;

    int current = i;
    // save starting row
    std::memcpy(buffer.data(), gain + current * n, n * sizeof(T));

    while (!visited[current]) {
      visited[current] = true;
      int next = perm[current];
      if (visited[next]) break;
      // move row 'next' into current
      std::memcpy(gain + current * n, gain + next * n, n * sizeof(T));
      current = next;
    }
    // place saved buffer into last slot
    std::memcpy(gain + current * n, buffer.data(), n * sizeof(T));
  }

  // ---- Reorder columns ----
  std::fill(visited.begin(), visited.end(), false);
  for (int j = 0; j < n; j++) {
    if (visited[j]) continue;

    int current = j;
    // save starting column
    for (int row = 0; row < n; row++) buffer[row] = gain[row * n + current];

    while (!visited[current]) {
      visited[current] = true;
      int next = perm[current];
      if (visited[next]) break;
      // move column 'next' into current
      for (int row = 0; row < n; row++)
        gain[row * n + current] = gain[row * n + next];
      current = next;
    }
    // place saved buffer into last slot
    for (int row = 0; row < n; row++) gain[row * n + current] = buffer[row];
  }
}
