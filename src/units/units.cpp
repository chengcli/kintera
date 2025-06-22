// C/C++
#include <regex>

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/constants.h>

#include <kintera/utils/fp_value.hpp>
#include <kintera/utils/trim_copy.hpp>

#include "units.hpp"

#define SmallNumber 1e-100

namespace kintera {

const std::map<std::string, Units> knownUnits{
    {"", Units(1.0)},
    {"1", Units(1.0)},

    // Mass [M]
    {"kg", Units(1.0, 1, 0, 0)},
    {"g", Units(1e-3, 1, 0, 0)},

    // Length [L]
    {"m", Units(1.0, 0, 1)},
    {"micron", Units(1e-6, 0, 1)},
    {"Å", Units(1e-10, 0, 1)},
    {"angstrom", Units(1e-10, 0, 1)},

    // Time [T]
    {"s", Units(1.0, 0, 0, 1)},
    {"min", Units(60, 0, 0, 1)},
    {"hr", Units(3600, 0, 0, 1)},

    // Temperature [K]
    {"K", Units(1.0, 0, 0, 0, 1)},

    // Current [A]
    {"A", Units(1.0, 0, 0, 0, 0, 1)},

    // Quantity [Q]
    {"mol", Units(1., 0, 0, 0, 0, 0, 1)},
    {"mole", Units(1., 0, 0, 0, 0, 0, 1)},
    {"kmol", Units(1e3, 0, 0, 0, 0, 0, 1)},
    {"molec", Units(1.0 / constants::Avogadro, 0, 0, 0, 0, 0, 1)},
    {"molecule", Units(1.0 / constants::Avogadro, 0, 0, 0, 0, 0, 1)},

    // Energy [M*L^2/T^2]
    {"J", Units(1.0, 1, 2, -2)},
    {"cal", Units(4.184, 1, 2, -2)},
    {"erg", Units(1e-7, 1, 2, -2)},
    {"eV", Units(constants::ElectronCharge, 1, 2, -2)},

    // Force [M*L/T^2]
    {"N", Units(1.0, 1, 1, -2)},
    {"dyn", Units(1e-5, 1, 1, -2)},

    // Pressure [M/L/T^2]
    {"Pa", Units(1.0, 1, -1, -2)},
    {"atm", Units(constants::OneAtm, 1, -1, -2)},
    {"bar", Units(1.0e5, 1, -1, -2)},
    {"dyn/cm^2", Units(0.1, 1, -1, -2)},

    // Volume [L^3]
    {"m^3", Units(1.0, 0, 3, 0)},
    {"liter", Units(0.001, 0, 3, 0)},
    {"L", Units(0.001, 0, 3, 0)},
    {"l", Units(0.001, 0, 3, 0)},
    {"cc", Units(1.0e-6, 0, 3, 0)},

    // Other electrical units
    {"ohm", Units(1.0, 1, 2, -3, 0, -2)},    // kg*m^2/s^3/A^2
    {"V", Units(1.0, 1, 2, -3, 0, -1)},      // kg*m^2/s^3/A
    {"coulomb", Units(1.0, 0, 0, 1, 0, 1)},  // A*s
};

const std::map<std::string, double> prefixes{
    {"Y", 1e24},  {"Z", 1e21},  {"E", 1e18},  {"P", 1e15}, {"T", 1e12},
    {"G", 1e9},   {"M", 1e6},   {"k", 1e3},   {"h", 1e2},  {"d", 1e-1},
    {"c", 1e-2},  {"m", 1e-3},  {"u", 1e-6},  {"n", 1e-9}, {"p", 1e-12},
    {"f", 1e-15}, {"a", 1e-18}, {"z", 1e-21}, {"y", 1e-24}};

Units::Units(double factor_, double mass, double length, double time,
             double temperature, double current, double quantity)
    : _mass_dim(mass),
      _length_dim(length),
      _time_dim(time),
      _temperature_dim(temperature),
      _current_dim(current),
      _quantity_dim(quantity) {
  factor(factor_);
  if (mass != 0 && length == -mass && time == -2 * mass && temperature == 0 &&
      current == 0 && quantity == 0) {
    // Dimension looks like Pa^n
    _pressure_dim = mass;
  } else if (mass != 0 && length == 2 * mass && time == -2 * mass &&
             temperature == 0 && current == 0 && quantity == 0) {
    // Dimension looks like J^n
    _energy_dim = mass;
  }
}

Units::Units(const std::string& name, bool force_unity) {
  size_t start = 0;

  // Determine factor
  static std::regex regexp("[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)");
  std::smatch matched;
  std::regex_search(name, matched, regexp);
  if (matched.size()) {
    std::string factor_str = *matched.begin();
    if (name.find(factor_str) == 0) {
      factor(fp_value_check(factor_str));
      start = factor_str.size();
    }
  }

  while (true) {
    // Split into groups of the form 'unit^exponent'
    size_t stop = name.find_first_of("*/", start);
    size_t caret = name.find('^', start);
    if (caret > stop) {
      // No caret in this group
      caret = -1;
    }
    std::string unit =
        trim_copy(name.substr(start, std::min(caret, stop) - start));

    double exponent = 1.0;
    if (caret != -1) {
      exponent = fp_value_check(name.substr(caret + 1, stop - caret - 1));
    }
    if (start != 0 && name[start - 1] == '/') {
      // This unit is in the denominator
      exponent = -exponent;
    }

    if (knownUnits.find(unit) != knownUnits.end()) {
      // Incorporate the unit defined by the current group
      *this *= knownUnits.at(unit).pow(exponent);
    } else {
      // See if the unit looks like a prefix + base unit
      std::string prefix = unit.substr(0, 1);
      std::string suffix = unit.substr(1);
      if (prefixes.find(prefix) != prefixes.end() &&
          knownUnits.find(suffix) != knownUnits.end()) {
        Units u = knownUnits.at(suffix);
        u.factor() *= prefixes.at(prefix);
        *this *= u.pow(exponent);
      } else {
        throw std::invalid_argument(
            fmt::format("Unknown unit '{}' in unit string '{}'", unit, name));
      }
    }

    start = stop + 1;
    if (stop == -1) {
      break;
    }
  }

  if (force_unity && (std::abs(factor() - 1.) > SmallNumber)) {
    throw std::invalid_argument(fmt::format(
        "Units string '{}' does not represent unity conversion factor", name));
  }
}

bool Units::convertible(const Units& other) const {
  return (_mass_dim == other._mass_dim && _length_dim == other._length_dim &&
          _time_dim == other._time_dim &&
          _temperature_dim == other._temperature_dim &&
          _current_dim == other._current_dim &&
          _quantity_dim == other._quantity_dim);
}

Units& Units::operator*=(const Units& other) {
  factor() *= other.factor();
  _mass_dim += other._mass_dim;
  _length_dim += other._length_dim;
  _time_dim += other._time_dim;
  _temperature_dim += other._temperature_dim;
  _current_dim += other._current_dim;
  _quantity_dim += other._quantity_dim;
  _pressure_dim += other._pressure_dim;
  _energy_dim += other._energy_dim;
  return *this;
}

Units Units::pow(double exponent) const {
  return Units(std::pow(factor(), exponent), _mass_dim * exponent,
               _length_dim * exponent, _time_dim * exponent,
               _temperature_dim * exponent, _current_dim * exponent,
               _quantity_dim * exponent);
}

std::string Units::str(bool skip_unity) const {
  std::map<std::string, double> dims{
      {"kg", _mass_dim},       {"m", _length_dim},  {"s", _time_dim},
      {"K", _temperature_dim}, {"A", _current_dim}, {"mol", _quantity_dim},
  };

  std::string num = "";
  std::string den = "";
  for (auto const& [dimension, exponent] : dims) {
    int rounded = (int)round(exponent);
    if (exponent == 0.) {
      // skip
    } else if (exponent == 1.) {
      num.append(fmt::format(" * {}", dimension));
    } else if (exponent == -1.) {
      den.append(fmt::format(" / {}", dimension));
    } else if (exponent == rounded && rounded > 0) {
      num.append(fmt::format(" * {}^{}", dimension, rounded));
    } else if (exponent == rounded) {
      den.append(fmt::format(" / {}^{}", dimension, -rounded));
    } else if (exponent > 0) {
      num.append(fmt::format(" * {}^{}", dimension, exponent));
    } else {
      den.append(fmt::format(" / {}^{}", dimension, -exponent));
    }
  }

  if (skip_unity && (std::abs(factor() - 1.) < SmallNumber)) {
    if (num.size()) {
      return fmt::format("{}{}", num.substr(3), den);
    }
    // print '1' as the numerator is empty
    return fmt::format("1{}", den);
  }

  std::string factor_str;
  if (factor() == round(factor())) {
    // ensure that fmt::format does not round to integer
    factor_str = fmt::format("{:.1f}", factor());
  } else {
    factor_str = fmt::format("{}", factor());
  }

  if (num.size()) {
    // concatenate factor, numerator and denominator (skipping leading '*')
    return fmt::format("{} {}{}", factor_str, num.substr(3), den);
  }

  return fmt::format("{}{}", factor_str, den);
}

bool Units::operator==(const Units& other) const {
  return factor() == other.factor() && _mass_dim == other._mass_dim &&
         _length_dim == other._length_dim && _time_dim == other._time_dim &&
         _temperature_dim == other._temperature_dim &&
         _current_dim == other._current_dim &&
         _quantity_dim == other._quantity_dim &&
         _pressure_dim == other._pressure_dim &&
         _energy_dim == other._energy_dim;
}

double Units::dimension(const std::string& primary) const {
  if (primary == "mass") {
    return _mass_dim;
  } else if (primary == "length") {
    return _length_dim;
  } else if (primary == "time") {
    return _time_dim;
  } else if (primary == "temperature") {
    return _temperature_dim;
  } else if (primary == "current") {
    return _current_dim;
  } else if (primary == "quantity") {
    return _quantity_dim;
  } else {
    throw std::invalid_argument(
        fmt::format("Unknown primary unit dimension '{}'", primary));
  }
}

std::map<std::string, std::string> UnitSystem::defaults() const {
  // Unit system defaults
  std::map<std::string, std::string> units{
      {"mass", "kg"},
      {"length", "m"},
      {"time", "s"},
      {"quantity", "kmol"},
      {"pressure", "Pa"},
      {"energy", "J"},
      {"temperature", "K"},
      {"current", "A"},
      {"activation-energy", "J / kmol"},
  };

  // Overwrite entries that have conversion factors
  for (const auto& [dimension, default_unit] : _defaults) {
    units[dimension] = default_unit;
  }

  return units;
}

void UnitSystem::set_defaults(std::initializer_list<std::string> units) {
  for (const auto& name : units) {
    auto unit = Units(name);
    if (unit.convertible(knownUnits.at("kg"))) {
      _mass_factor = unit.factor();
      _defaults["mass"] = name;
    } else if (unit.convertible(knownUnits.at("m"))) {
      _length_factor = unit.factor();
      _defaults["length"] = name;
    } else if (unit.convertible(knownUnits.at("s"))) {
      _time_factor = unit.factor();
      _defaults["time"] = name;
    } else if (unit.convertible(knownUnits.at("kmol"))) {
      _quantity_factor = unit.factor();
      _defaults["quantity"] = name;
    } else if (unit.convertible(knownUnits.at("Pa"))) {
      _pressure_factor = unit.factor();
      _defaults["pressure"] = name;
    } else if (unit.convertible(knownUnits.at("J"))) {
      _energy_factor = unit.factor();
      _defaults["energy"] = name;
    } else if (unit.convertible(knownUnits.at("K"))) {
      // Do nothing -- no other scales are supported for temperature
      if (unit.factor() != 1.) {
        throw std::invalid_argument(
            fmt::format("Unit '{}' is not supported for temperature", name));
      }
    } else if (unit.convertible(knownUnits.at("A"))) {
      // Do nothing -- no other scales are supported for current
      if (unit.factor() != 1.) {
        throw std::invalid_argument(
            fmt::format("Unit '{}' is not supported for current", name));
      }
    } else {
      throw std::invalid_argument(fmt::format(
          "Unit '{}' is not convertible to any basic dimension", name));
    }
  }
}

double UnitSystem::convert(double value, const std::string& src,
                           const std::string& dest) const {
  return convert(value, Units(src), Units(dest));
}

double UnitSystem::convert(double value, const Units& src,
                           const Units& dest) const {
  if (!src.convertible(dest)) {
    throw std::invalid_argument(
        fmt::format("Cannot convert from '{}' to '{}'", src.str(), dest.str()));
  }
  return value * src.factor() / dest.factor();
}

double UnitSystem::convert_to(double value, const std::string& dest) const {
  return convert_to(value, Units(dest));
}

double UnitSystem::convert_to(double value, const Units& dest) const {
  return value / dest.factor() *
         pow(_mass_factor,
             dest._mass_dim - dest._pressure_dim - dest._energy_dim) *
         pow(_length_factor,
             dest._length_dim + dest._pressure_dim - 2 * dest._energy_dim) *
         pow(_time_factor,
             dest._time_dim + 2 * dest._pressure_dim + 2 * dest._energy_dim) *
         pow(_quantity_factor, dest._quantity_dim) *
         pow(_pressure_factor, dest._pressure_dim) *
         pow(_energy_factor, dest._energy_dim);
}

double UnitSystem::convert_from(double value, const std::string& dest) const {
  return convert_from(value, Units(dest));
}

double UnitSystem::convert_from(double value, const Units& src) const {
  return value * src.factor() *
         pow(_mass_factor,
             -src._mass_dim + src._pressure_dim + src._energy_dim) *
         pow(_length_factor,
             -src._length_dim - src._pressure_dim + 2 * src._energy_dim) *
         pow(_time_factor,
             -src._time_dim - 2 * src._pressure_dim - 2 * src._energy_dim) *
         pow(_quantity_factor, -src._quantity_dim) *
         pow(_pressure_factor, -src._pressure_dim) *
         pow(_energy_factor, -src._energy_dim);
}

}  // namespace kintera
