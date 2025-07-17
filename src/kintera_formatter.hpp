#pragma once

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/utils/format.hpp>

#include "reaction.hpp"
#include "species.hpp"

template <>
struct fmt::formatter<kintera::Composition> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::Composition& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "{}", kintera::to_string(p));
  }
};

template <>
struct fmt::formatter<kintera::Reaction> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::Reaction& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "{}", p.equation());
  }
};

template <>
struct fmt::formatter<kintera::SpeciesThermo> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::SpeciesThermo& p, FormatContext& ctx) const {
    std::stringstream ss;
    ss << "* vapors = " << fmt::format("{}", p.vapor_ids()) << "\n";
    ss << "* clouds = " << fmt::format("{}", p.cloud_ids()) << "\n";
    ss << "* cref_R = " << fmt::format("{}", p.cref_R()) << "\n";
    ss << "* uref_R = " << fmt::format("{}", p.uref_R()) << "\n";
    ss << "* sref_R = " << fmt::format("{}", p.sref_R()) << "\n";

    return fmt::format_to(ctx.out(), ss.str());
  }
};

template <>
struct fmt::formatter<std::vector<kintera::Reaction>> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const std::vector<kintera::Reaction>& vec,
              FormatContext& ctx) const {
    std::string result = "[\n";
    for (size_t i = 0; i < vec.size(); ++i) {
      result += fmt::format("{}", vec[i]);
      if (i < vec.size() - 1) {
        result += "\n";
      }
    }
    result += "]";
    return fmt::format_to(ctx.out(), "{}", result);
  }
};
