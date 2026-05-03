// kintera
#include "load_xsection.hpp"

// torch
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include <kintera/utils/parse_comp_string.hpp>

namespace kintera {

std::tuple<std::vector<double>, std::vector<double>, std::vector<Composition>>
load_xsection_yaml(YAML::Node const& data_node,
                   std::vector<std::string> const& branch_strs) {
  std::vector<double> wavelength;
  std::vector<double> xsection;
  std::vector<Composition> branches;

  for (auto const& s : branch_strs) {
    branches.push_back(parse_comp_string(s));
  }

  int nbranch = std::max((int)branches.size(), 1);

  for (auto const& entry : data_node) {
    auto row = entry.as<std::vector<double>>();
    TORCH_CHECK(row.size() >= 2, "YAML data row must have at least 2 values");

    wavelength.push_back(row[0]);
    for (int b = 0; b < nbranch; b++) {
      xsection.push_back(b + 1 < (int)row.size() ? row[b + 1] : row[1]);
    }
  }

  return {wavelength, xsection, branches};
}

}  // namespace kintera
