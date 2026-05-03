// kintera
#include "load_xsection.hpp"

// torch
#include <torch/torch.h>

#include <kintera/utils/find_resource.hpp>
#include <kintera/utils/parse_comp_string.hpp>

namespace kintera {

std::tuple<std::vector<double>, std::vector<double>, std::vector<Composition>>
load_xsection_kin7(std::string const& filename,
                   std::vector<std::string> const& branch_strs) {
  auto full_path = find_resource(filename);
  FILE* file = fopen(full_path.c_str(), "r");
  TORCH_CHECK(file, "Could not open file: ", full_path);

  std::vector<double> wavelength;
  std::vector<double> xsection;
  std::vector<Composition> branches;

  for (auto const& s : branch_strs) {
    branches.push_back(parse_comp_string(s));
  }

  int nbranch = branches.size();
  int min_is = 9999, max_ie = 0;

  char* line = NULL;
  size_t len = 0;
  ssize_t read;

  while ((read = getline(&line, &len, file)) != -1) {
    if (line[0] == '\n') continue;

    char equation[61];
    int is, ie, nwave;
    float temp;

    int num =
        sscanf(line, "%60c%4d%4d%4d%6f", equation, &is, &ie, &nwave, &temp);
    min_is = std::min(min_is, is);
    max_ie = std::max(max_ie, ie);

    TORCH_CHECK(num == 5, "Header format error in file '", filename, "'");

    if (wavelength.size() == 0) {
      wavelength.resize(nwave);
      xsection.resize(nwave * nbranch, 0.0);
    }

    int ncols = 7;
    int nrows = ceil(1. * nwave / ncols);

    equation[60] = '\0';
    auto product = parse_comp_string(equation);

    auto it = std::find(branches.begin(), branches.end(), product);

    if (it == branches.end()) {
      for (int i = 0; i < nrows; i++) getline(&line, &len, file);
    } else {
      for (int i = 0; i < nrows; i++) {
        getline(&line, &len, file);
        for (int j = 0; j < ncols; j++) {
          float wave, cross;
          int num = sscanf(line + 17 * j, "%7f%10f", &wave, &cross);
          TORCH_CHECK(num == 2, "Cross-section format error in file '",
                      filename, "'");
          int b = it - branches.begin();
          int k = i * ncols + j;

          if (k >= nwave) break;
          wavelength[k] = wave * 0.1;  // Angstrom -> nm
          xsection[k * nbranch + b] = cross;
        }
      }
    }
  }

  // Trim unused wavelength range
  if (min_is < max_ie && min_is > 0) {
    wavelength = std::vector<double>(wavelength.begin() + min_is - 1,
                                     wavelength.begin() + max_ie);
    xsection = std::vector<double>(xsection.begin() + (min_is - 1) * nbranch,
                                   xsection.begin() + max_ie * nbranch);
  }

  // First branch is total absorption; subtract others to get pure absorption
  for (size_t i = 0; i < wavelength.size(); i++) {
    for (int j = 1; j < nbranch; j++) {
      xsection[i * nbranch] -= xsection[i * nbranch + j];
    }
    xsection[i * nbranch] = std::max(xsection[i * nbranch], 0.);
  }

  free(line);
  fclose(file);

  return {wavelength, xsection, branches};
}

}  // namespace kintera
