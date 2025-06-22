#pragma once

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
    std::ostringstream reactions;
    auto r = p.reactions();

    if (r.size() == 0) {
      return fmt::format_to(ctx.out(), "--");
    }

    for (size_t i = 0; i < r.size(); ++i) {
      reactions << fmt::format("R{}: {}, ", i + 1, r[i]);
      if (i != r.size() - 1) {
        reactions << ", ";
      }
      reactions << fmt::format(
          "A= {:.2e}, b= {:.2f}, Ea_R= {:.2f}, E4_R= {:.2f}", p.A()[i],
          p.b()[i], p.Ea_R()[i], p.E4_R()[i]);
      if (i != r.size() - 1) reactions << ";\n";
    }

    return fmt::format_to(ctx.out(), "{}", reactions.str());
  }
};

template <>
struct fmt::formatter<kintera::EvaporationOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::EvaporationOptions& p, FormatContext& ctx) const {
    std::ostringstream reactions;
    auto r = p.reactions();

    if (r.size() == 0) {
      return fmt::format_to(ctx.out(), "--");
    }

    for (size_t i = 0; i < r.size(); ++i) {
      reactions << fmt::format("R{}: {}, ", i + 1, r[i]);
      if (i != r.size() - 1) {
        reactions << ", ";
      }
      reactions << fmt::format(
          "diff_c= {:.2f}, diff_T= {:.2f}, diff_P= {:.2f}, "
          "vm= {:.2f}, diamter= {:.2f}",
          p.diff_c()[i], p.diff_T()[i], p.diff_P()[i], p.vm()[i],
          p.diameter()[i]);
      if (i != r.size() - 1) reactions << ";\n";
    }

    return fmt::format_to(ctx.out(), "{}", reactions.str());
  }
};

template <>
struct fmt::formatter<kintera::KineticsOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::KineticsOptions& p, FormatContext& ctx) const {
    std::ostringstream reactions;
    auto r = p.reactions();
    for (size_t i = 0; i < r.size(); ++i) {
      reactions << fmt::format("R{}: {}", i + 1, r[i]);
      if (i != r.size() - 1) reactions << ";\n";
    }

    return fmt::format_to(
        ctx.out(),
        "species= (\n{}\n); Tref= {}; Pref= {};\nreactions= (\n{}\n)",
        static_cast<kintera::SpeciesThermo>(p), p.Tref(), p.Pref(),
        reactions.str());
  }
};
