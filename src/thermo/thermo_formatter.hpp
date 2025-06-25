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
    std::ostringstream reactions;
    auto r = p.reactions();
    for (size_t i = 0; i < r.size(); ++i) {
      reactions << fmt::format("R{}: {}", i + 1, r[i]);
      if (i != r.size() - 1) {
        reactions << ", ";
      }
      reactions << fmt::format("Tmin= {:.2f}, Tmax= {:.2f}", p.minT()[i],
                               p.maxT()[i]);
      if (i != r.size() - 1) reactions << ";\n";
    }

    return fmt::format_to(ctx.out(), "{}", reactions.str());
  }
};

template <>
struct fmt::formatter<kintera::ThermoOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::ThermoOptions& p, FormatContext& ctx) const {
    std::ostringstream reactions;
    auto r = p.reactions();
    for (size_t i = 0; i < r.size(); ++i) {
      reactions << fmt::format("R{}: {}", i + 1, r[i]);
      if (i != r.size() - 1) reactions << ";\n";
    }

    return fmt::format_to(ctx.out(),
                          "species= (\n{}\n);\nTref= {}; Pref= "
                          "{};\nreactions= (\n{}\n)",
                          static_cast<kintera::SpeciesThermo>(p), p.Tref(),
                          p.Pref(), reactions.str());
  }
};
