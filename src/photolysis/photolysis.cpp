//! @file photolysis.cpp
//! @brief Photolysis rate module implementation

#include "photolysis.hpp"

#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

#include <kintera/math/interpolation.hpp>
#include <kintera/units/units.hpp>
#include <kintera/utils/find_resource.hpp>
#include <kintera/utils/parse_comp_string.hpp>

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
  bool global_temp_grid_initialized = false;

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

    for (auto const& s : branch_strs) {
      branch_comps.push_back(parse_comp_string(s));
    }

    // Parse cross-section data
    if (rxn_node["cross-section"]) {
      auto cs_nodes = rxn_node["cross-section"];
      bool require_temperature = cs_nodes.size() > 1;
      std::vector<double> reaction_temperatures;

      for (auto const& cs_node : cs_nodes) {
        std::string format = cs_node["format"].as<std::string>("YAML");

        if (cs_node["temperature"]) {
          reaction_temperatures.push_back(cs_node["temperature"].as<double>());
        } else {
          TORCH_CHECK(!require_temperature,
                      "photolysis reactions with multiple cross-section blocks "
                      "must define 'temperature' for each block");
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
      }

      if (require_temperature) {
        if (!global_temp_grid_initialized) {
          options->temperature() = reaction_temperatures;
          global_temp_grid_initialized = true;
        } else {
          TORCH_CHECK(options->temperature() == reaction_temperatures,
                      "All multi-temperature photolysis reactions must use "
                      "the same temperature grid");
        }
      }

      options->cross_section_nslabs().push_back(cs_nodes.size());
    }

    options->branches().push_back(branch_comps);
  }

  return options;
}

PhotolysisImpl::PhotolysisImpl(PhotolysisOptions const& options_)
    : options(options_) {
  reset();
}

void PhotolysisImpl::reset() {
  _nreaction = options->reactions().size();
  _nbranches.clear();
  cross_section.clear();
  branch_stoich.clear();

  if (_nreaction == 0) return;

  wavelength = register_buffer(
      "wavelength", torch::tensor(options->wavelength(), torch::kFloat64));
  auto global_temps = options->temperature();
  if (global_temps.empty()) {
    global_temps = {0.0, 1000.0};
  }
  temp_grid = register_buffer("temp_grid",
                              torch::tensor(global_temps, torch::kFloat64));

  int nwave = options->wavelength().size();
  int ntemp = temp_grid.size(0);
  int total_single_size = 0;
  int total_full_size = 0;
  for (int r = 0; r < _nreaction; ++r) {
    int nbranch = options->branches()[r].size();
    if (nbranch == 0) nbranch = 1;
    total_single_size += nwave * nbranch;
    total_full_size += ntemp * nwave * nbranch;
  }
  bool has_explicit_slab_counts =
      options->cross_section_nslabs().size() == static_cast<size_t>(_nreaction);
  bool infer_all_full =
      !has_explicit_slab_counts && options->temperature().size() > 1 &&
      static_cast<int>(options->cross_section().size()) == total_full_size;

  int xs_offset = 0;
  for (int r = 0; r < _nreaction; r++) {
    int nbranch = options->branches()[r].size();
    if (nbranch == 0) nbranch = 1;
    _nbranches.push_back(nbranch);
    int xs_size_single = nwave * nbranch;
    int xs_size_full = ntemp * xs_size_single;
    int nslabs = 1;
    if (has_explicit_slab_counts) {
      nslabs = options->cross_section_nslabs()[r];
    } else if (infer_all_full) {
      nslabs = ntemp;
    }

    TORCH_CHECK(nslabs == 1 || nslabs == ntemp, "Reaction ", r,
                " must provide either one cross-section slab or one slab for "
                "every temperature in the shared photolysis grid");
    std::vector<double> xs_data(xs_size_full, 0.0);

    if (nslabs == ntemp) {
      TORCH_CHECK(
          xs_offset + xs_size_full <= (int)options->cross_section().size(),
          "Insufficient multi-temperature cross-section data for reaction ", r);
      std::copy(options->cross_section().begin() + xs_offset,
                options->cross_section().begin() + xs_offset + xs_size_full,
                xs_data.begin());
      xs_offset += xs_size_full;
    } else {
      TORCH_CHECK(
          xs_offset + xs_size_single <= (int)options->cross_section().size(),
          "Insufficient cross-section data for reaction ", r);
      std::vector<double> xs_single(xs_size_single, 0.0);
      std::copy(options->cross_section().begin() + xs_offset,
                options->cross_section().begin() + xs_offset + xs_size_single,
                xs_single.begin());
      for (int t = 0; t < ntemp; ++t) {
        std::copy(xs_single.begin(), xs_single.end(),
                  xs_data.begin() + t * xs_size_single);
      }
      xs_offset += xs_size_single;
    }

    auto xs_tensor =
        torch::tensor(xs_data, torch::kFloat64).view({ntemp, nwave, nbranch});
    cross_section.push_back(
        register_buffer("cross_section_" + std::to_string(r), xs_tensor));

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
  auto temp_shape =
      temp.numel() == 1 ? std::vector<int64_t>{} : temp.sizes().vec();
  auto query_shape = temp_shape;
  query_shape.push_back(wave.size(0));

  std::vector<int64_t> wave_view_shape(query_shape.size(), 1);
  wave_view_shape.back() = wave.size(0);
  auto wave_query = wave.view(wave_view_shape).expand(query_shape).contiguous();

  torch::Tensor temp_query;
  if (temp.numel() == 1) {
    temp_query = torch::full(query_shape, temp.item<double>(), temp.options());
  } else {
    temp_query = temp.unsqueeze(-1).expand(query_shape).contiguous();
  }

  return interpn({temp_query, wave_query}, {temp_grid, wavelength},
                 cross_section[rxn_idx]);
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

torch::Tensor PhotolysisImpl::forward(torch::Tensor T, torch::Tensor wave,
                                      torch::Tensor actinic_flux) {
  if (_nreaction == 0) {
    return torch::empty({0}, T.options());
  }

  auto out_shape = T.sizes().vec();
  out_shape.push_back(_nreaction);

  // Find max branches for padding
  int max_branches = 0;
  for (int r = 0; r < _nreaction; r++) {
    max_branches = std::max(max_branches, _nbranches[r]);
  }

  // Process all reactions: compute cross-sections
  std::vector<torch::Tensor> xs_diss_list;
  for (int r = 0; r < _nreaction; r++) {
    auto xs = interp_cross_section(r, wave, T);  // (..., nwave, nbranch)

    // Sum dissociation branches (skip index 0 which is photoabsorption)
    torch::Tensor xs_diss;
    if (_nbranches[r] > 1) {
      xs_diss = xs.narrow(-1, 1, _nbranches[r] - 1).sum(-1);
    } else {
      xs_diss = xs.select(-1, 0);
    }
    xs_diss_list.push_back(xs_diss);  // (..., nwave)
  }

  // Stack all xs_diss: (..., nwave, nreaction)
  auto xs_diss_stacked = torch::stack(xs_diss_list, -1);

  // Vectorized integration for all reactions
  // Handle different aflux dimensions
  if (actinic_flux.dim() == 1) {
    // aflux: (nwave,), xs_diss_stacked: (..., nwave, nreaction)
    auto integrand = xs_diss_stacked *
                     actinic_flux.unsqueeze(-1);  // (..., nwave, nreaction)
    auto rates = torch::trapezoid(integrand, wave, -2);  // (..., nreaction)
    return rates.view(out_shape);
  } else if (actinic_flux.dim() == 2) {
    // aflux: (nwave, nspatial) or similar, xs_diss_stacked: (..., nwave,
    // nreaction) Need to broadcast properly
    auto aflux_exp = actinic_flux.unsqueeze(-1);  // (nwave, nspatial, 1)
    auto integrand = xs_diss_stacked.unsqueeze(-2) *
                     aflux_exp;  // (..., nwave, nspatial, nreaction)
    auto rates =
        torch::trapezoid(integrand, wave, -3);  // (..., nspatial, nreaction)
    return rates.view(out_shape);
  } else {
    // Higher dimensional aflux: need to handle broadcasting
    // Expand xs_diss_stacked to match aflux dimensions
    auto xs_exp = xs_diss_stacked;
    for (int d = 1; d < actinic_flux.dim(); d++) {
      xs_exp = xs_exp.unsqueeze(-2);
    }
    // Broadcast: xs_exp (..., nwave, 1, ..., 1, nreaction), aflux (..., nwave,
    // ...)
    auto integrand = xs_exp * actinic_flux.unsqueeze(-1);
    // Find wavelength dimension (should be -aflux.dim() or similar)
    int wave_dim =
        xs_diss_stacked.dim() - 1;  // Last dimension before nreaction
    auto rates = torch::trapezoid(integrand, wave, wave_dim);
    return rates.view(out_shape);
  }
}

}  // namespace kintera
