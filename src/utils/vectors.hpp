#pragma once

// C/C++
#include <algorithm>
#include <vector>

template <typename T>
std::vector<T> merge_vectors(std::vector<T> const& vec1,
                             std::vector<T> const& vec2) {
  std::vector<T> merged = vec1;
  merged.insert(merged.end(), vec2.begin(), vec2.end());
  return merged;
}

template <typename T>
std::vector<T> merge_vectors(std::vector<T> const& vec1,
                             std::vector<T> const& vec2, int n1, int n2) {
  std::vector<T> merged = vec1;
  merged.insert(merged.begin() + n1, vec2.begin(), vec2.begin() + n2);
  merged.insert(merged.end(), vec2.begin() + n2, vec2.end());
  return merged;
}

template <typename T>
std::vector<T> sort_vectors(std::vector<T> const& vec,
                            std::vector<size_t> const& indices) {
  std::vector<T> sorted(vec.size());
  std::transform(indices.begin(), indices.end(), sorted.begin(),
                 [&vec](size_t index) { return vec[index]; });
  return sorted;
}

template <typename T>
std::vector<int> locate_vectors(std::vector<T> const& a,
                                std::vector<T> const& b) {
  std::unordered_set<T> a_set(a.begin(), a.end());
  std::vector<int> indices;

  for (size_t i = 0; i < b.size(); ++i) {
    if (a_set.count(b[i])) {
      indices.push_back(i);
    }
  }
  return indices;
}
