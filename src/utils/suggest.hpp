#pragma once

// C/C++
#include <string>
#include <vector>

namespace kintera {

std::string suggest(const std::string& input,
                    const std::vector<std::string>& allowed);

}  // namespace kintera
