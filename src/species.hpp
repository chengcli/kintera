#pragma once

// C/C++
#include <string>

// kintera
#include <kintera/utils/func2.hpp>
#include <kintera/utils/func3.hpp>

// arg
#include "add_arg.h"

namespace kintera {

void init_species_from_yaml(std::string filename);

struct SpeciesThermo {
  virtual ~SpeciesThermo() = default;

  //! \return species names
  std::vector<std::string> species() const;

  ADD_ARG(std::vector<int>, vapor_ids);
  ADD_ARG(std::vector<int>, cloud_ids);

  ADD_ARG(std::vector<double>, cref_R);
  ADD_ARG(std::vector<double>, uref_R);
  ADD_ARG(std::vector<double>, sref_R);

  ADD_ARG(std::vector<user_func2>, intEng_R_extra);
  ADD_ARG(std::vector<user_func2>, cv_R_extra);
  ADD_ARG(std::vector<user_func2>, cp_R_extra);

  //! only used for gas species, the rests are no-ops
  ADD_ARG(std::vector<user_func3>, entropy_R_extra);

  //! This variable is funny. Because compressibility factor only applies to
  //! gas and we need extra enthalpy functions for cloud species, so we combined
  //! compressibility factor and extra enthalpy functions into one variable
  //! called czh, which has the size of nspcies
  ADD_ARG(std::vector<user_func2>, czh);

  //! Similarly, the derivative of compressibility factor with respect to
  //! concentration is stored here, with first 'ngas' entries being
  //! valid numbers. The rests are no-ops.
  ADD_ARG(std::vector<user_func2>, czh_ddC);
};

}  // namespace kintera

#undef ADD_ARG
