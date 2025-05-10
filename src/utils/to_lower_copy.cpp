// C/C++
#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

namespace kintera {

std::string to_lower_copy(const std::string& str) {
  std::string lower_str = str;
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return lower_str;
}

}  // namespace kintera
