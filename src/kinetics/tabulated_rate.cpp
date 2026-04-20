// C/C++
#include <fstream>
#include <sstream>

// yaml
#include <yaml-cpp/yaml.h>

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/math/interpolation.hpp>
#include <kintera/units/units.hpp>
#include <kintera/utils/find_resource.hpp>

#include "tabulated_rate.hpp"

namespace kintera {

extern std::vector<std::string> species_names;

// ============ File I/O ============

RateTable read_rate_table(std::string const& filepath) {
  std::ifstream fin(filepath);
  TORCH_CHECK(fin.is_open(), "Cannot open rate table file: ", filepath);

  RateTable table;
  std::string line;

  // Helper: read next non-comment, non-empty line
  auto next_line = [&]() {
    while (std::getline(fin, line)) {
      // trim leading whitespace
      auto pos = line.find_first_not_of(" \t");
      if (pos == std::string::npos) continue;
      line = line.substr(pos);
      if (line.empty() || line[0] == '#') continue;
      return true;
    }
    return false;
  };

  // Read number of dimensions
  TORCH_CHECK(next_line(), "Failed to read ndim from ", filepath);
  int ndim = std::stoi(line);
  TORCH_CHECK(ndim >= 1, "ndim must be >= 1, got ", ndim);

  // Read each dimension's coordinates
  int total_values = 1;
  for (int d = 0; d < ndim; ++d) {
    TORCH_CHECK(next_line(), "Failed to read dimension ", d, " size");
    int ncoord = std::stoi(line);
    TORCH_CHECK(ncoord >= 2, "Dimension ", d, " must have >= 2 points, got ",
                ncoord);

    std::vector<double> coords;
    coords.reserve(ncoord);
    while (static_cast<int>(coords.size()) < ncoord) {
      TORCH_CHECK(next_line(), "Failed to read coordinates for dimension ", d);
      std::istringstream iss(line);
      double val;
      while (iss >> val) {
        coords.push_back(val);
      }
    }
    TORCH_CHECK(static_cast<int>(coords.size()) == ncoord, "Expected ", ncoord,
                " coordinates for dimension ", d, ", got ", coords.size());

    table.coords.push_back(torch::tensor(coords, torch::kFloat64));
    total_values *= ncoord;
  }

  // Read total number of values
  TORCH_CHECK(next_line(), "Failed to read total number of values");
  int nvalues = std::stoi(line);
  TORCH_CHECK(nvalues == total_values, "Expected ", total_values,
              " values but header says ", nvalues);

  // Read values
  std::vector<double> values;
  values.reserve(nvalues);
  while (static_cast<int>(values.size()) < nvalues) {
    TORCH_CHECK(next_line(), "Failed to read values");
    std::istringstream iss(line);
    double val;
    while (iss >> val) {
      values.push_back(val);
    }
  }
  TORCH_CHECK(static_cast<int>(values.size()) == nvalues, "Expected ", nvalues,
              " values, got ", values.size());

  // Reshape values tensor to (n0, n1, ..., nd, 1) — trailing 1 for interpn
  std::vector<int64_t> shape;
  for (auto const& c : table.coords) {
    shape.push_back(c.size(0));
  }
  shape.push_back(1);  // nval dimension for interpn
  table.values = torch::tensor(values, torch::kFloat64).reshape(shape);

  return table;
}

// ============ Options ============

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set,
                        TabulatedRateOptions op) {
  for (auto& react : op->reactions()) {
    for (auto& [name, _] : react.reactants()) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      if (it == species_names.end()) continue;  // skip background/bath species
      vapor_set.insert(name);
    }
    for (auto& [name, _] : react.products()) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      if (it == species_names.end()) continue;  // skip background/bath species
      vapor_set.insert(name);
    }
  }
}

TabulatedRateOptions TabulatedRateOptionsImpl::from_yaml(
    const YAML::Node& root) {
  auto options = TabulatedRateOptionsImpl::create();

  for (auto const& rxn_node : root) {
    TORCH_CHECK(rxn_node["type"], "Reaction type not specified");

    if (rxn_node["type"].as<std::string>() != options->name()) {
      continue;  // skip non-tabulated reactions
    }

    TORCH_CHECK(rxn_node["equation"],
                "'equation' is not defined in the reaction");
    std::string equation = rxn_node["equation"].as<std::string>();
    options->reactions().push_back(Reaction(equation));

    TORCH_CHECK(rxn_node["rate-constant"],
                "'rate-constant' is not defined in the reaction");
    auto rc_node = rxn_node["rate-constant"];

    // Read table file path
    TORCH_CHECK(
        rc_node["file"],
        "'file' is not defined in tabulated rate-constant for: ", equation);
    std::string table_file = rc_node["file"].as<std::string>();
    options->files().push_back(table_file);

    // Override log-interpolation per reaction if specified
    if (rc_node["log-interpolation"]) {
      options->log_interpolation(rc_node["log-interpolation"].as<bool>());
    }

    // Try to find and load the table file
    std::string full_path;
    try {
      full_path = find_resource(table_file);
    } catch (...) {
      full_path = table_file;  // use as-is if not found in search paths
    }

    auto table = read_rate_table(full_path);
    options->tables().push_back(std::move(table));
  }

  return options;
}

// ============ Module Implementation ============

TabulatedRateImpl::TabulatedRateImpl(TabulatedRateOptions const& options_)
    : options(options_) {
  reset();
}

void TabulatedRateImpl::reset() {
  nreaction_ = options->reactions().size();
  if (nreaction_ == 0) return;

  // Load tables from files if not already loaded
  if (options->tables().empty() && !options->files().empty()) {
    TORCH_CHECK(static_cast<int>(options->files().size()) == nreaction_,
                "Number of table files (", options->files().size(),
                ") must match number of reactions (", nreaction_, ")");
    for (auto const& file : options->files()) {
      std::string full_path;
      try {
        full_path = find_resource(file);
      } catch (...) {
        full_path = file;
      }
      options->tables().push_back(read_rate_table(full_path));
    }
  }

  TORCH_CHECK(static_cast<int>(options->tables().size()) == nreaction_,
              "Number of tables (", options->tables().size(),
              ") must match number of reactions (", nreaction_, ")");

  coords_cache_.clear();
  values_cache_.clear();

  ndim_ = options->tables()[0].coords.size();

  for (int r = 0; r < nreaction_; ++r) {
    auto const& table = options->tables()[r];
    TORCH_CHECK(static_cast<int>(table.coords.size()) == ndim_,
                "All tabulated reactions must have the same number of "
                "dimensions. Reaction ",
                r, " has ", table.coords.size(), " dims but expected ", ndim_);

    // Register coordinate buffers
    std::vector<torch::Tensor> coord_refs;
    for (int d = 0; d < ndim_; ++d) {
      auto name = fmt::format("coords_{}_{}", r, d);
      auto buf = register_buffer(name, table.coords[d].clone());
      coord_refs.push_back(buf);
    }
    coords_cache_.push_back(coord_refs);

    // Register values buffer
    // If log-interpolation, store log10 of values (clamp to avoid log(0))
    torch::Tensor vals = table.values.clone();
    if (options->log_interpolation()) {
      vals = torch::log10(vals.clamp_min(1e-300));
    }
    auto vname = fmt::format("values_{}", r);
    auto vbuf = register_buffer(vname, vals);
    values_cache_.push_back(vbuf);
  }
}

void TabulatedRateImpl::pretty_print(std::ostream& os) const {
  os << "TabulatedRate(" << nreaction_ << " reactions, " << ndim_ << "D tables";
  if (options->log_interpolation()) os << ", log-interp";
  os << ")";
}

torch::Tensor TabulatedRateImpl::forward(
    torch::Tensor T, torch::Tensor P, torch::Tensor C,
    std::map<std::string, torch::Tensor> const& other) {
  if (nreaction_ == 0) {
    return torch::empty({0}, T.options());
  }

  // Build query coordinates
  // For log interpolation, query in log10 space
  auto log_interp = options->log_interpolation();

  // Determine the query tensors for each dimension
  // Convention: dim0 = temperature, dim1 = pressure
  auto query_T = log_interp ? torch::log10(T.clamp_min(1e-30)) : T;
  auto query_P = log_interp ? torch::log10(P.clamp_min(1e-30)) : P;

  // Expand T if not yet expanded to match P shape
  auto temp = T.sizes() == P.sizes() ? query_T : query_T;

  // Compute rate for each reaction and stack
  auto base_shape = P.sizes().vec();
  base_shape.push_back(nreaction_);
  auto result = torch::empty(base_shape, T.options());

  for (int r = 0; r < nreaction_; ++r) {
    auto const& coords = coords_cache_[r];
    auto const& values = values_cache_[r];

    // Build query vector
    std::vector<torch::Tensor> query;
    if (ndim_ >= 1) {
      // Flatten T for interpn (expects (nbatch, ndim) or flat)
      query.push_back(temp.reshape({-1}));
    }
    if (ndim_ >= 2) {
      query.push_back(query_P.reshape({-1}));
    }
    // For ndim > 2, additional dimensions from 'other' would go here

    // Coordinate arrays (already in log10 if log_interp, stored that way)
    std::vector<torch::Tensor> coord_arrays;
    for (int d = 0; d < ndim_; ++d) {
      if (log_interp) {
        coord_arrays.push_back(torch::log10(coords[d].clamp_min(1e-30)));
      } else {
        coord_arrays.push_back(coords[d]);
      }
    }

    // Interpolate
    auto interp_result = interpn(query, coord_arrays, values);
    // interpn returns (nbatch, 1), squeeze the nval dim
    interp_result = interp_result.squeeze(-1);

    // Convert back from log10 if needed
    if (log_interp) {
      interp_result = torch::pow(10.0, interp_result);
    }

    // Reshape back to original spatial dimensions
    result.select(-1, r) = interp_result.reshape(P.sizes());
  }

  return result;
}

}  // namespace kintera
