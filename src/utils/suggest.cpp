// C/C++
#include "suggest.hpp"

#include <algorithm>
#include <iostream>
#include <limits>

namespace kintera {

// Levenshtein distance between two strings
int levenshtein(const std::string& s1, const std::string& s2) {
  int n = s1.size();
  int m = s2.size();
  std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1));

  for (int i = 0; i <= n; ++i) dp[i][0] = i;
  for (int j = 0; j <= m; ++j) dp[0][j] = j;

  for (int i = 1; i <= n; ++i) {
    for (int j = 1; j <= m; ++j) {
      int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
      dp[i][j] = std::min({
          dp[i - 1][j] + 1,        // deletion
          dp[i][j - 1] + 1,        // insertion
          dp[i - 1][j - 1] + cost  // substitution
      });
    }
  }
  return dp[n][m];
}

std::string suggest(const std::string& input,
                    const std::vector<std::string>& allowed) {
  int min_dist = std::numeric_limits<int>::max();
  std::string best_match;

  for (const auto& candidate : allowed) {
    int dist = levenshtein(input, candidate);
    if (dist < min_dist) {
      min_dist = dist;
      best_match = candidate;
    }
  }

  return best_match;
}

}  // namespace kintera
