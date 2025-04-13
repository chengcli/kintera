#pragma once

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/kintera_formatter.hpp>

#include "thermodynamics.hpp"

template <>
struct fmt::formatter<kintera::Nucleation> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::Nucleation& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "({}; min_tem = {}; max_tem = {})",
                          p.reaction(), p.min_tem(), p.max_tem());
  }
};

template <>
struct fmt::formatter<kintera::CondensationOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::CondensationOptions& p, FormatContext& ctx) {
    std::ostringstream reactions;
    for (size_t i = 0; i < p.react().size(); ++i) {
      reactions << fmt::format("R{}: {}", i + 1, p.react()[i]);
      if (i != p.react().size() - 1) {
        reactions << "; ";
      }
    }

    std::ostringstream species;
    for (size_t i = 0; i < p.species().size(); ++i) {
      species << p.species()[i];
      if (i != p.species().size() - 1) {
        species << ", ";
      }
    }

    return fmt::format_to(ctx.out(), "(react = ({}); species = ({}))",
                          reactions.str(), species.str());
  }
};

template <>
struct fmt::formatter<kintera::ThermodynamicsOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::ThermodynamicsOptions& p, FormatContext& ctx) {
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

    return fmt::format_to(ctx.out(),
                          "(Rd = {}; gammad = {}; vapors = ({}); clouds = "
                          "({}); cond = {})",
                          p.Rd(), p.gammad(), vapors.str(), clouds.str(),
                          p.cond());
  }
};
