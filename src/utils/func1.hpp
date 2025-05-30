#pragma once

// C/C++
#include <string>
#include <unordered_map>

// kintera
#include <kintera/thermo/thermo.h>

std::unordered_map<std::string, user_func1>& get_user_func1() {
  static std::unordered_map<std::string, user_func1> f1map;
  return f1map;
}

struct Func1Registrar {
  Func1Registrar(const std::string& name, user_func1 func) {
    get_user_func1()[name] = func;
  }
};
