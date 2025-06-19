#pragma once

// fmt
#include <fmt/format.h>

// kintera
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
    std::ostringstream vapors;
    for (size_t i = 0; i < p.vapor_ids().size(); ++i) {
      vapors << p.vapor_ids()[i];
      if (i != p.vapor_ids().size() - 1) {
        vapors << ", ";
      }
    }

    std::ostringstream clouds;
    for (size_t i = 0; i < p.cloud_ids().size(); ++i) {
      clouds << p.cloud_ids()[i];
      if (i != p.cloud_ids().size() - 1) {
        clouds << ", ";
      }
    }

    std::ostringstream cref;
    for (size_t i = 0; i < p.cref_R().size(); ++i) {
      cref << p.cref_R()[i];
      if (i != p.cref_R().size() - 1) {
        cref << ", ";
      }
    }

    std::ostringstream uref;
    for (size_t i = 0; i < p.uref_R().size(); ++i) {
      uref << p.uref_R()[i];
      if (i != p.uref_R().size() - 1) {
        uref << ", ";
      }
    }

    std::ostringstream sref;
    for (size_t i = 0; i < p.sref_R().size(); ++i) {
      sref << p.sref_R()[i];
      if (i != p.sref_R().size() - 1) {
        sref << ", ";
      }
    }

    return fmt::format_to(
        ctx.out(),
        "vapors= ({});\nclouds= ({});\ncv_R= ({});\nu0_R= ({});\ns0_R= ({})",
        vapors.str(), clouds.str(), cref.str(), uref.str(), sref.str());
  }
};
