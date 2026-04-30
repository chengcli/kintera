#pragma once

// C/C++
#include <array>
#include <string>
#include <vector>

// kintera
#include <kintera/utils/user_funcs.hpp>

// arg
#include "add_arg.h"

namespace at {
class Tensor;
class TensorOptions;
}  // namespace at

namespace YAML {
class Node;
}  // namespace YAML

namespace kintera {

using Nasa9CoeffArray = std::array<double, 9>;
using Nasa9CoeffTable = std::vector<Nasa9CoeffArray>;

void init_species_from_yaml(std::string filename);
void init_species_from_yaml(YAML::Node const& config);
void init_species_from_kinetics_base(std::string const& master_input_path);

struct SpeciesThermoImpl {
  static std::shared_ptr<SpeciesThermoImpl> create() {
    return std::make_shared<SpeciesThermoImpl>();
  }

  virtual ~SpeciesThermoImpl() = default;

  //! \return species names
  std::vector<std::string> species() const;

  at::Tensor narrow_copy(at::Tensor data,
                         std::shared_ptr<SpeciesThermoImpl> const& other) const;
  void accumulate(at::Tensor& data, at::Tensor const& other_data,
                  std::shared_ptr<SpeciesThermoImpl> const& other) const;
  bool has_nasa9() const;
  at::Tensor nasa9_coeffs_low_tensor(
      at::TensorOptions const& options = at::TensorOptions()) const;
  at::Tensor nasa9_coeffs_high_tensor(
      at::TensorOptions const& options = at::TensorOptions()) const;
  at::Tensor nasa9_Tmid_tensor(
      at::TensorOptions const& options = at::TensorOptions()) const;

  ADD_ARG(std::vector<int>, vapor_ids);
  ADD_ARG(std::vector<int>, cloud_ids);

  ADD_ARG(std::vector<double>, cref_R);
  ADD_ARG(std::vector<double>, uref_R);
  ADD_ARG(std::vector<double>, sref_R);

  ADD_ARG(std::vector<std::string>, intEng_R_extra);
  ADD_ARG(std::vector<std::string>, cp_R_extra);

  //! only used for gas species, the rests are no-ops
  ADD_ARG(std::vector<std::string>, entropy_R_extra);

  //! This variable is funny. Because compressibility factor only applies to
  //! gas and we need extra enthalpy functions for cloud species, so we combined
  //! compressibility factor and extra enthalpy functions into one variable
  //! called czh, which has the size of nspcies
  ADD_ARG(std::vector<std::string>, czh);

  //! Similarly, the derivative of compressibility factor with respect to
  //! concentration is stored here, with first 'ngas' entries being
  //! valid numbers. The rests are no-ops.
  ADD_ARG(std::vector<std::string>, czh_ddC);

  //! NASA-9 low-temperature coefficients, one 9-coefficient record per species.
  ADD_ARG(Nasa9CoeffTable, nasa9_low);
  //! NASA-9 high-temperature coefficients, one 9-coefficient record per
  //! species.
  ADD_ARG(Nasa9CoeffTable, nasa9_high);
  //! NASA-9 range mid-point temperature [K], one value per species.
  ADD_ARG(std::vector<double>, nasa9_Tmid);
};
using SpeciesThermo = std::shared_ptr<SpeciesThermoImpl>;

void populate_thermo(SpeciesThermo thermo);

void check_dimensions(SpeciesThermo const& thermo);

SpeciesThermo merge_thermo(SpeciesThermo const& thermo1,
                           SpeciesThermo const& thermo2);

extern std::vector<std::string> species_names;
extern std::vector<double> species_weights;
extern std::vector<double> species_cref_R;
extern std::vector<double> species_uref_R;
extern std::vector<double> species_sref_R;
extern bool species_initialized;

}  // namespace kintera

#undef ADD_ARG
