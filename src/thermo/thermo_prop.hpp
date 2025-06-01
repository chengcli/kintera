// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>

// kintera
#include <kintera/utils/func1.hpp>

namespace kintera {

/*!
 * \brief Calculate thermodynamic properties based on temperature and
 * concentrations
 *
 * \param[in] temp temperature in Kelvin.
 * \param[in] conc concentrations of species in mol/m^3.
 * \param[in] nspecies number of species.
 * \param[in] offset thermodynamic property offsets for each species, used in
 * the calculation.
 * \param[in] first_derivative first derivative of thermodynamic properties with
 * respect to temperature.
 * \param[in] extra user-defined functions for additional temperature dependent
 * dependencies.
 */
template <typename T>
T thermo_prop(T temp, T const *conc, int nspecies, T const *offset,
              T const *first_derivative, user_func1 const *extra) {
  T prop = 0.;
  for (int i = 0; i < nspecies; i++) {
    T propi = offset[i] + first_derivative[i] * temp;
    if (extra[i]) {
      propi += extra[i](temp);
    }
    prop += conc[i] * propi;
  }
  return prop;
}

}  // namespace kintera
