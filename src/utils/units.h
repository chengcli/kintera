/**
 * @file Units.h
 * Header for unit conversion utilities, which are used to translate
 * user input from input files (See @ref inputGroup and
 * class @link Cantera::Units Units@endlink).
 */

// This file is part of Cantera. See License.txt in the top-level directory or
// at https://cantera.org/license.txt for license and copyright information.

#ifndef CT_UNITS_H
#define CT_UNITS_H

#include <any>
#include <map>
#include <string>
#include <vector>

namespace kintera {
/*!
 * String representations of units can be written using multiplication,
 * division, and exponentiation. Spaces are ignored. Positive, negative, and
 * decimal exponents are permitted. Examples:
 *
 *     kg*m/s^2
 *     J/kmol
 *     m*s^-2
 *     J/kg/K
 *
 * Metric prefixes are recognized for all units, such as nm, hPa, mg, EJ, mL,
 * kcal.
 *
 * Special functions for converting activation energies allow these values to be
 * expressed as either energy per quantity, energy (for example, eV), or
 * temperature by applying a factor of the Avogadro number or the gas constant
 * where needed.
 */
class UnitSystem {
 public:
  double m_mass_factor = 1.0;
  double m_length_factor = 1.0;
  double m_time_factor = 1.0;
  double m_pressure_factor = 1.0;
  double m_energy_factor = 1.0;
  double m_activation_energy_factor = 1.0;
  double m_quantity_factor = 1.0;
  bool m_explicit_activation_energy = false;
};

}  // namespace kintera

#endif
