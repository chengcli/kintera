#pragma once

// C/C++
#include <string>
#include <unordered_map>

using user_func3 = double (*)(double, double, double);

inline std::unordered_map<std::string, user_func3>& get_user_func3() {
  static std::unordered_map<std::string, user_func3> f3map;
  return f3map;
}

struct Func3Registrar {
  Func3Registrar(const std::string& name, user_func3 func) {
    get_user_func3()[name] = func;
  }
};
