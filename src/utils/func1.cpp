#include "func1.hpp"

std::unordered_map<std::string, user_func1>& get_user_func1() {
  static std::unordered_map<std::string, user_func1> f1map;
  return f1map;
}
