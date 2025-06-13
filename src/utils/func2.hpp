#pragma once

// C/C++
#include <string>
#include <unordered_map>

typedef double (*user_func2)(double temp, double conc);

inline std::unordered_map<std::string, user_func2>& get_user_func2() {
  static std::unordered_map<std::string, user_func2> f1map;
  return f1map;
}

struct Func2Registrar {
  Func2Registrar(const std::string& name, user_func2 func) {
    get_user_func2()[name] = func;
  }
};
