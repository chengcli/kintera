#pragma once

// C/C++
#include <map>
#include <string>
#include <vector>

// arg
#include <kintera/add_arg.h>

namespace kintera {

//! A representation of the units associated with a dimensional quantity.
/*!
 * Used for converting quantities between unit systems and checking for
 * dimensional consistency. Units objects are mainly used within UnitSystem
 * class to convert values from a user-specified Unit system to kintera's
 * base units (SI + mol).
 */
class Units {
 public:
  //! Create a Units object with the specified dimensions.
  explicit Units(double factor = 1.0, double mass = 0, double length = 0,
                 double time = 0, double temperature = 0, double current = 0,
                 double quantity = 0);

  //! Create an object with the specified dimensions
  //! \param units        A string representation of the units. See UnitSystem
  //!                     for a description of the formatting options.
  //! \param force_unity  ensure that conversion factor is equal to one
  explicit Units(const std::string& units, bool force_unity = false);

  //! Returns `true` if the specified Units are dimensionally consistent
  bool convertible(const Units& other) const;

  //! Multiply two Units objects, combining their conversion factors and
  //! dimensions
  Units& operator*=(const Units& other);

  //! Provide a string representation of these Units
  //! \param skip_unity  do not print '1' if conversion factor is equal to one
  std::string str(bool skip_unity = true) const;

  //! Raise these Units to a power, changing both the conversion factor and
  //! the dimensions of these Units.
  Units pow(double exponent) const;

  bool operator==(const Units& other) const;

  //! Return dimension of primary unit component
  //! ("mass", "length", "time", "temperature", "current", or "quantity")
  double dimension(const std::string& primary) const;

  ADD_ARG(double, factor) = 1.;

 private:
  double _mass_dim = 0.0;
  double _length_dim = 0.0;
  double _time_dim = 0.0;
  double _temperature_dim = 0.0;
  double _current_dim = 0.0;
  double _quantity_dim = 0.0;
  double _pressure_dim =
      0.0;  //!< pseudo-dimension to track explicit pressure units
  double _energy_dim =
      0.0;  //!< pseudo-dimension to track explicit energy units

  friend class UnitSystem;
};

//! Unit conversion utility
/*!
 * Provides functions for converting dimensional values from a given unit
 * system. The main use is for converting values specified in input files to
 * Cantera's native unit system, which is SI units except for the use of kmol as
 * the base unit of quantity, that is, kilogram, meter, second, kelvin, ampere,
 * and kmol.
 *
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
 */

class UnitSystem {
 public:
  UnitSystem() = default;
  UnitSystem(std::initializer_list<std::string> units) { set_defaults(units); }

  std::map<std::string, std::string> defaults() const;
  void set_defaults(std::initializer_list<std::string> units);

  //! Convert `value` from the units of `src` to the units of `dest`.
  double convert(double value, const std::string& src,
                 const std::string& dest) const;
  double convert(double value, const Units& src, const Units& dest) const;

  //! Convert `value` to the specified `dest` units from the appropriate units
  //! for this unit system (defined by `setDefaults`)
  double convert_to(double value, const std::string& dest) const;
  double convert_to(double value, const Units& dest) const;

  //! Convert `value` from the specified `src` units to units appropriate for
  //! this unit system (defined by `setDefaults`)
  double convert_from(double value, const std::string& src) const;
  double convert_from(double value, const Units& src) const;

 private:
  //! Factor to convert mass from this unit system to kg
  double _mass_factor = 1.0;

  //! Factor to convert length from this unit system to meters
  double _length_factor = 1.0;

  //! Factor to convert time from this unit system to seconds
  double _time_factor = 1.0;

  //! Factor to convert pressure from this unit system to Pa
  double _pressure_factor = 1.0;

  //! Factor to convert energy from this unit system to J
  double _energy_factor = 1.0;

  //! Factor to convert quantity from this unit system to mol
  double _quantity_factor = 1.0;

  //! Map of dimensions (mass, length, etc.) to names of specified default
  //! units
  std::map<std::string, std::string> _defaults = {
      {"mass", "kg"},       {"length", "m"},    {"time", "s"},
      {"quantity", "mol"},  {"pressure", "Pa"}, {"energy", "J"},
      {"temperature", "K"}, {"current", "A"},
  };
};

}  // namespace kintera

#undef ADD_ARG
