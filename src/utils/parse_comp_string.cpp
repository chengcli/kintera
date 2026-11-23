// torch
#include <torch/torch.h>

// fmt
#include <fmt/format.h>

// kintera
#include "fp_value.hpp"
#include "get_value.hpp"
#include "parse_comp_string.hpp"
#include "trim_copy.hpp"

namespace kintera {

Composition parse_comp_string(const std::string& ss,
                              const std::vector<std::string>& names) {
  Composition x;
  for (size_t k = 0; k < names.size(); k++) {
    x[names[k]] = 0.0;
  }

  size_t start = 0;
  size_t stop = 0;
  size_t left = 0;
  while (stop < ss.size()) {
    size_t colon = ss.find(':', left);
    if (colon == std::string::npos) {
      break;
    }
    size_t valstart = ss.find_first_not_of(" \t\n", colon + 1);
    stop = ss.find_first_of(", ;\n\t", valstart);
    std::string name = trim_copy(ss.substr(start, colon - start));
    if (!names.empty() && x.find(name) == x.end()) {
      TORCH_CHECK(false, fmt::format("{} not found in names", name));
    }

    double value;
    try {
      value = fp_value_check(ss.substr(valstart, stop - valstart));
    } catch (c10::Error&) {
      // If we have a key containing a colon, we expect this to fail. In
      // this case, take the current substring as part of the key and look
      // to the right of the next colon for the corresponding value.
      // Otherwise, this is an invalid composition string.
      std::string testname = ss.substr(start, stop - start);
      if (testname.find_first_of(" \n\t") != std::string::npos) {
        // Space, tab, and newline are never allowed in names
        throw;
      } else if (ss.substr(valstart, stop - valstart).find(':') !=
                 std::string::npos) {
        left = colon + 1;
        stop = 0;  // Force another iteration of this loop
        continue;
      } else {
        throw;
      }
    }
    TORCH_CHECK(get_value(x, name, 0.0) == 0.0,
                fmt::format("duplicate key {}", name));

    x[name] = value;
    start = ss.find_first_not_of(", ;\n\t", stop + 1);
    left = start;
  }
  TORCH_CHECK(left == start, fmt::format("Unable to parse key-value pair '{}'",
                                         ss.substr(start, stop)));
  if (stop != std::string::npos && !trim_copy(ss.substr(stop)).empty()) {
    TORCH_CHECK(
        false,
        fmt::format("Found non-key:value data in composition string '{}'",
                    ss.substr(stop)));
  }
  return x;
}

}  // namespace kintera
