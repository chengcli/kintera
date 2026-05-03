#pragma once

// C/C++
#include <string>
#include <tuple>
#include <vector>

// kintera
#include <kintera/reaction.hpp>

namespace YAML {
class Node;
}

namespace kintera {

std::tuple<std::vector<double>, std::vector<double>, std::vector<Composition>>
load_xsection_kin7(std::string const& filename,
                   std::vector<std::string> const& branch_strs);

std::tuple<std::vector<double>, std::vector<double>, std::vector<Composition>>
load_xsection_yaml(YAML::Node const& data_node,
                   std::vector<std::string> const& branch_strs);

}  // namespace kintera
