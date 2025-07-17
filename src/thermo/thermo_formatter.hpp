#pragma once

// C/C++
#include <sstream>

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/kintera_formatter.hpp>

#include "thermo.hpp"

template <>
struct fmt::formatter<kintera::NucleationOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::NucleationOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    auto r = p.reactions();
    for (size_t i = 0; i < r.size(); ++i) {
      ss << fmt::format("R{}: {}", i + 1, r[i]) << ", ";
      ss << fmt::format("Tmin= {:.2f}, Tmax= {:.2f}", p.minT()[i], p.maxT()[i]);
      ss << "\n";
    }

    return fmt::format_to(ctx.out(), "{}", ss.str());
  }
};

template <>
struct fmt::formatter<kintera::ThermoOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::ThermoOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    p.report(ss);
    ss << fmt::format("{}", static_cast<kintera::SpeciesThermo>(p));
    ss << fmt::format("{}", p.nucleation());
    return fmt::format_to(ctx.out(), ss.str());
  }
};
