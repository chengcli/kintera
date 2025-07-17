#pragma once

// fmt
#include <fmt/format.h>

template <typename A>
struct fmt::formatter<std::vector<A>> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const std::vector<A>& vec, FormatContext& ctx) const {
    std::string result = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      result += fmt::format("{}", vec[i]);
      if (i < vec.size() - 1) {
        result += ", ";
      }
    }
    result += "]";
    return fmt::format_to(ctx.out(), "{}", result);
  }
};
