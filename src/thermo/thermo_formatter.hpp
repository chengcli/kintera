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
    ss << "Nucleation options:\n";
    p->report(ss);

    return fmt::format_to(ctx.out(), "{}", ss.str());
  }
};

template <>
struct fmt::formatter<kintera::ThermoOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::ThermoOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    p->report(ss);
    ss << fmt::format("{}",
                      std::static_pointer_cast<kintera::SpeciesThermoImpl>(p));
    ss << fmt::format("{}", p->nucleation());
    return fmt::format_to(ctx.out(), ss.str());
  }
};
