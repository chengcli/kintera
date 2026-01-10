//! @file photolysis.cpp
//! @brief Photolysis rate module implementation

#include <yaml-cpp/yaml.h>
#include <fmt/format.h>

#include <kintera/math/interpolation.hpp>
#include <kintera/units/units.hpp>
#include <kintera/utils/find_resource.hpp>
#include <kintera/utils/parse_comp_string.hpp>

#include "photolysis.hpp"

namespace kintera {

extern std::vector<std::string> species_names;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set,
                        PhotolysisOptions op) {
  for (auto& react : op->reactions()) {
    for (auto& [name, _] : react.reactants()) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name,
                  " not found in species list");
      vapor_set.insert(name);
    }
    for (auto& [name, _] : react.products()) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name,
                  " not found in species list");
      vapor_set.insert(name);
    }
  }
}

//! Load cross-section from KINETICS7 format file
static std::tuple<std::vector<double>, std::vector<double>,
                  std::vector<Composition>>
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

//! Load cross-section from YAML inline data
static std::tuple<std::vector<double>, std::vector<double>,
                  std::vector<Composition>>
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

PhotolysisOptions PhotolysisOptionsImpl::from_yaml(
    const YAML::Node& root,
    std::shared_ptr<PhotolysisOptionsImpl> derived_type_ptr) {
  auto options =
      derived_type_ptr ? derived_type_ptr : PhotolysisOptionsImpl::create();

  std::vector<double> all_temperatures;

  for (auto const& rxn_node : root) {
    TORCH_CHECK(rxn_node["type"], "Reaction type not specified");

    if (rxn_node["type"].as<std::string>() != options->name()) {
      continue;
    }

    TORCH_CHECK(rxn_node["equation"],
                "'equation' is not defined in the reaction");

    std::string equation = rxn_node["equation"].as<std::string>();
    options->reactions().push_back(Reaction(equation));

    // Parse branches: first is always photoabsorption
    std::vector<std::string> branch_strs;
    std::vector<Composition> branch_comps;
    auto& rxn = options->reactions().back();

    std::string absorb_str;
    for (auto const& [sp, coeff] : rxn.reactants()) {
      absorb_str += sp + ":" + std::to_string((int)coeff) + " ";
    }
    branch_strs.push_back(absorb_str);

    if (rxn_node["branches"]) {
      for (auto const& branch :
           rxn_node["branches"].as<std::vector<std::string>>()) {
        branch_strs.push_back(branch);
      }
    } else if (rxn.products() != rxn.reactants()) {
      std::string prod_str;
      for (auto const& [sp, coeff] : rxn.products()) {
        prod_str += sp + ":" + std::to_string((int)coeff) + " ";
      }
      branch_strs.push_back(prod_str);
    }

    options->branch_names().push_back(branch_strs);

    // Parse cross-section data
    if (rxn_node["cross-section"]) {
      for (auto const& cs_node : rxn_node["cross-section"]) {
        std::string format = cs_node["format"].as<std::string>("YAML");

        std::vector<double> temp_range = {0., 1000.};
        if (cs_node["temperature-range"]) {
          temp_range = cs_node["temperature-range"].as<std::vector<double>>();
        }

        if (all_temperatures.empty()) {
          all_temperatures = temp_range;
        }

        std::vector<double> wave, xs;
        std::vector<Composition> br;

        if (format == "KINETICS7") {
          auto filename = cs_node["filename"].as<std::string>();
          std::tie(wave, xs, br) = load_xsection_kin7(filename, branch_strs);
        } else if (format == "YAML") {
          std::tie(wave, xs, br) =
              load_xsection_yaml(cs_node["data"], branch_strs);
        } else {
          TORCH_CHECK(false, "Unknown cross-section format: ", format);
        }

        if (options->wavelength().empty()) {
          options->wavelength() = wave;
        }

        for (auto v : xs) {
          options->cross_section().push_back(v);
        }

        for (auto const& s : branch_strs) {
          branch_comps.push_back(parse_comp_string(s));
        }
      }
    }

    options->branches().push_back(branch_comps);
  }

  if (!all_temperatures.empty()) {
    options->temperature() = all_temperatures;
  }

  return options;
}

PhotolysisImpl::PhotolysisImpl(PhotolysisOptions const& options_)
    : options(options_) {
  reset();
}

void PhotolysisImpl::reset() {
  _nreaction = options->reactions().size();

  if (_nreaction == 0) return;

  wavelength = register_buffer(
      "wavelength", torch::tensor(options->wavelength(), torch::kFloat64));

  if (!options->temperature().empty()) {
    temp_grid = register_buffer(
        "temp_grid", torch::tensor(options->temperature(), torch::kFloat64));
  } else {
    temp_grid = register_buffer("temp_grid",
                                torch::tensor({0., 1000.}, torch::kFloat64));
  }

  int nwave = options->wavelength().size();
  int ntemp = options->temperature().size();
  if (ntemp == 0) ntemp = 2;

  int xs_offset = 0;
  for (int r = 0; r < _nreaction; r++) {
    int nbranch = options->branches()[r].size();
    if (nbranch == 0) nbranch = 1;
    _nbranches.push_back(nbranch);

    int xs_size = nwave * nbranch;
    std::vector<double> xs_data(xs_size, 0.0);

    if (xs_offset + xs_size <= (int)options->cross_section().size()) {
      std::copy(options->cross_section().begin() + xs_offset,
                options->cross_section().begin() + xs_offset + xs_size,
                xs_data.begin());
    }

    auto xs_tensor =
        torch::tensor(xs_data, torch::kFloat64).view({nwave, nbranch});
    cross_section.push_back(
        register_buffer("cross_section_" + std::to_string(r), xs_tensor));

    xs_offset += xs_size;

    // Build stoichiometry tensor for branches
    int nspecies = species_names.size();
    auto stoich_tensor = torch::zeros({nbranch, nspecies}, torch::kFloat64);

    for (int b = 0; b < nbranch; b++) {
      if (b < (int)options->branches()[r].size()) {
        auto const& branch = options->branches()[r][b];
        for (auto const& [sp, coeff] : branch) {
          auto it = std::find(species_names.begin(), species_names.end(), sp);
          if (it != species_names.end()) {
            int sp_idx = it - species_names.begin();
            stoich_tensor[b][sp_idx] = coeff;
          }
        }
      }
    }

    branch_stoich.push_back(
        register_buffer("branch_stoich_" + std::to_string(r), stoich_tensor));
  }
}

void PhotolysisImpl::pretty_print(std::ostream& os) const {
  os << "Photolysis Rate Module:\n";
  os << "  Reactions: " << _nreaction << "\n";
  for (int r = 0; r < _nreaction; r++) {
    os << "  [" << r + 1 << "] " << options->reactions()[r].equation()
       << " (branches: " << _nbranches[r] << ")\n";
  }
}

torch::Tensor PhotolysisImpl::interp_cross_section(int rxn_idx,
                                                   torch::Tensor wave,
                                                   torch::Tensor temp) {
  TORCH_CHECK(rxn_idx >= 0 && rxn_idx < _nreaction,
              "Invalid reaction index: ", rxn_idx);
  return interpn({wave}, {wavelength}, cross_section[rxn_idx]);
}

torch::Tensor PhotolysisImpl::get_effective_stoich(int rxn_idx,
                                                   torch::Tensor wave,
                                                   torch::Tensor aflux,
                                                   torch::Tensor temp) {
  TORCH_CHECK(rxn_idx >= 0 && rxn_idx < _nreaction,
              "Invalid reaction index: ", rxn_idx);

  auto xs = interp_cross_section(rxn_idx, wave, temp);
  auto aflux_exp = aflux.unsqueeze(-1);
  auto branch_rate = torch::trapezoid(xs * aflux_exp, wave, 0);
  auto total_rate = branch_rate.sum() + 1e-30;
  auto branch_frac = branch_rate / total_rate;

  return (branch_frac.unsqueeze(-1) * branch_stoich[rxn_idx]).sum(0);
}

torch::Tensor PhotolysisImpl::forward(
    torch::Tensor T, torch::Tensor P, torch::Tensor C,
    std::map<std::string, torch::Tensor> const& other) {
  if (_nreaction == 0) {
    return torch::empty({0}, T.options());
  }

  TORCH_CHECK(other.count("wavelength"),
              "Photolysis requires 'wavelength' in other");
  TORCH_CHECK(other.count("actinic_flux"),
              "Photolysis requires 'actinic_flux' in other");

  auto wave = other.at("wavelength");
  auto aflux = other.at("actinic_flux");

  auto out_shape = T.sizes().vec();
  out_shape.push_back(_nreaction);
  auto result = torch::zeros(out_shape, T.options());

  for (int r = 0; r < _nreaction; r++) {
    auto xs = interp_cross_section(r, wave, T);

    // Sum dissociation branches (skip index 0 which is photoabsorption)
    torch::Tensor xs_diss;
    if (_nbranches[r] > 1) {
      xs_diss = xs.narrow(1, 1, _nbranches[r] - 1).sum(1);
    } else {
      xs_diss = xs.select(1, 0);
    }

    torch::Tensor rate;
    if (aflux.dim() == 1) {
      rate = torch::trapezoid(xs_diss * aflux, wave, 0);
      result.select(-1, r).fill_(rate.item<double>());
    } else if (aflux.dim() == 2) {
      auto xs_exp = xs_diss.unsqueeze(-1);
      rate = torch::trapezoid(xs_exp * aflux, wave, 0);
      result.select(-1, r).copy_(rate);
    } else {
      auto xs_exp = xs_diss;
      for (int d = 1; d < aflux.dim(); d++) {
        xs_exp = xs_exp.unsqueeze(-1);
      }
      rate = torch::trapezoid(xs_exp * aflux, wave, 0);
      result.select(-1, r).copy_(rate);
    }
  }

  return result;
}

}  // namespace kintera
