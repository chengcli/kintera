#pragma once

// C/C++
#include <string>
#include <unordered_map>

typedef double (*user_func1)(double temp);

std::unordered_map<std::string, user_func1>& get_user_func1();

struct Func1Registrar {
  Func1Registrar(const std::string& name, user_func1 func) {
    get_user_func1()[name] = func;
  }
};
