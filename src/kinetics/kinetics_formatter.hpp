#pragma once

// C/C++
#include <sstream>

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/kintera_formatter.hpp>

#include "kinetics.hpp"

template <>
struct fmt::formatter<kintera::ArrheniusOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::ArrheniusOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    auto r = p.reactions();

    if (r.size() == 0) {
      return fmt::format_to(ctx.out(), "--\n");
    }

    for (size_t i = 0; i < r.size(); ++i) {
      ss << fmt::format("R{}: {}, ", i + 1, r[i]);
      ss << fmt::format("A= {:.2e}, b= {:.2f}, Ea_R= {:.2f}, E4_R= {:.2f}",
                        p.A()[i], p.b()[i], p.Ea_R()[i], p.E4_R()[i]);
      ss << "\n";
    }

    return fmt::format_to(ctx.out(), "{}", ss.str());
  }
};

template <>
struct fmt::formatter<kintera::EvaporationOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::EvaporationOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    auto r = p.reactions();

    if (r.size() == 0) {
      return fmt::format_to(ctx.out(), "--\n");
    }

    for (size_t i = 0; i < r.size(); ++i) {
      ss << fmt::format("R{}: {}, ", i + 1, r[i]);
      ss << fmt::format(
          "diff_c= {:.2f}, diff_T= {:.2f}, diff_P= {:.2f}, "
          "vm= {:.2f}, diamter= {:.2f}",
          p.diff_c()[i], p.diff_T()[i], p.diff_P()[i], p.vm()[i],
          p.diameter()[i]);
      ss << "\n";
    }

    return fmt::format_to(ctx.out(), "{}", ss.str());
  }
};

template <>
struct fmt::formatter<kintera::CoagulationOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::CoagulationOptions& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "{}",
                          static_cast<kintera::ArrheniusOptions>(p));
  }
};

template <>
struct fmt::formatter<kintera::KineticsOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::KineticsOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    p.report(ss);

    ss << fmt::format("{}", static_cast<kintera::SpeciesThermo>(p));
    ss << "Arrhenius Reactions:\n";
    ss << fmt::format("{}", p.arrhenius());

    ss << "Coagulation Reactions:\n";
    ss << fmt::format("{}", p.coagulation());

    ss << "Evaporation Reactions:\n";
    ss << fmt::format("{}", p.evaporation());

    return fmt::format_to(ctx.out(), ss.str());
  }
};
