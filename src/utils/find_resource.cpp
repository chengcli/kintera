// C/C++
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_set>

// kintera
#include <configure.h>

#include "find_resource.hpp"

namespace kintera {

// const char* search_paths = "";
char search_paths[65536] = ".";
static std::mutex dir_mutex;

#ifdef WINDOWS
char pathsep = ';';
#else
char pathsep = ':';
#endif

namespace {

std::string expand_user_path(std::string const& dir) {
  std::string d = stripnonprint(dir);
  if (d.find("~/") == 0 || d.find("~\\") == 0) {
    char* home = getenv("HOME");  // POSIX systems
    if (!home) {
      home = getenv("USERPROFILE");  // Windows systems
    }
    if (home) {
      d = home + d.substr(1, std::string::npos);
    }
  }
  return d;
}

std::vector<std::string> python_site_package_resource_dirs(
    std::filesystem::path const& root) {
  std::vector<std::string> dirs;
  std::error_code ec;
  if (root.empty() || !std::filesystem::is_directory(root, ec)) {
    return dirs;
  }

  for (auto const& entry : std::filesystem::directory_iterator(root, ec)) {
    if (ec || !entry.is_directory()) continue;
    auto name = entry.path().filename().string();
    if (name.rfind("python", 0) != 0) continue;
    dirs.push_back(
        (entry.path() / "site-packages" / "kintera" / "data").string());
  }
  return dirs;
}

std::vector<std::string> virtualenv_resource_directories() {
  std::vector<std::string> dirs;
  for (char const* env_name : {"VIRTUAL_ENV", "CONDA_PREFIX"}) {
    auto env_root = std::getenv(env_name);
    if (!env_root || std::strlen(env_root) == 0) continue;

    auto root = std::filesystem::path(env_root);
    auto lib_dirs = python_site_package_resource_dirs(root / "lib");
    dirs.insert(dirs.end(), lib_dirs.begin(), lib_dirs.end());
    auto lib64_dirs = python_site_package_resource_dirs(root / "lib64");
    dirs.insert(dirs.end(), lib64_dirs.begin(), lib64_dirs.end());
  }
  return dirs;
}

std::vector<std::string> normalize_resource_directories(
    std::vector<std::string> const& dirs) {
  std::vector<std::string> normalized;
  std::unordered_set<std::string> seen;

  auto add_dir = [&](std::string const& dir) {
    auto expanded = expand_user_path(dir);
    if (!expanded.empty() && seen.insert(expanded).second) {
      normalized.push_back(expanded);
    }
  };

  for (auto const& dir : dirs) {
    add_dir(dir);
  }
  for (auto const& dir : virtualenv_resource_directories()) {
    add_dir(dir);
  }
  add_dir(std::string(KINTERA_ROOT_DIR) + "/data");
  return normalized;
}

}  // namespace

std::string stripnonprint(std::string const& s) {
  std::string ss = "";
  for (size_t i = 0; i < s.size(); i++) {
    if (isprint(s[i])) {
      ss += s[i];
    }
  }
  return ss;
}

char* serialize_search_paths(std::vector<std::string> const& dirs) {
  auto normalized = normalize_resource_directories(dirs);
  std::string s = "";
  for (size_t i = 0; i < normalized.size(); i++) {
    s += normalized[i];
    if (i + 1 < normalized.size()) {
      s += pathsep;
    }
  }
  strncpy(search_paths, s.c_str(), 65536);
  return search_paths;
}

std::vector<std::string> deserialize_search_paths(char const* p) {
  std::vector<std::string> dirs;
  std::string s(p);
  size_t start = 0;
  size_t end = s.find(pathsep);
  while (end != std::string::npos) {
    dirs.push_back(s.substr(start, end - start));
    start = end + 1;
    end = s.find(pathsep, start);
  }
  auto tail = s.substr(start, end);
  if (!tail.empty()) {
    dirs.push_back(tail);
  }
  return normalize_resource_directories(dirs);
}

std::vector<std::string> default_resource_directories() {
  return normalize_resource_directories({".", "data"});
}

std::vector<std::string> current_resource_directories() {
  return deserialize_search_paths(search_paths);
}

std::string describe_resource_directories() {
  std::ostringstream oss;
  auto dirs = current_resource_directories();
  for (auto const& dir : dirs) {
    oss << "\n'" << dir << "'";
  }
  return oss.str();
}

void set_default_directories() {
  serialize_search_paths(default_resource_directories());
}

void add_resource_directory(std::string const& dir, bool prepend) {
  std::unique_lock<std::mutex> dirLock(dir_mutex);
  auto input_dirs = current_resource_directories();
  std::string d = expand_user_path(dir);

  // Remove any existing entry for this directory
  auto iter = std::find(input_dirs.begin(), input_dirs.end(), d);
  if (iter != input_dirs.end()) {
    input_dirs.erase(iter);
  }

  if (prepend) {
    // Insert this directory at the beginning of the search path
    input_dirs.insert(input_dirs.begin(), d);
  } else {
    // Append this directory to the end of the search path
    input_dirs.push_back(d);
  }

  serialize_search_paths(input_dirs);
}

std::string find_resource(std::string const& name) {
  std::unique_lock<std::mutex> dirLock(dir_mutex);
  std::string::size_type islash = name.find('/');
  std::string::size_type ibslash = name.find('\\');
  std::string::size_type icolon = name.find(':');

  std::vector<std::string> dirs = current_resource_directories();

  // Expand "~/" to user's home directory, if possible
  if (name.find("~/") == 0 || name.find("~\\") == 0) {
    char* home = getenv("HOME");  // POSIX systems
    if (!home) {
      home = getenv("USERPROFILE");  // Windows systems
    }
    if (home) {
      std::string full_name = home + name.substr(1, std::string::npos);
      std::ifstream fin(full_name);
      if (fin) {
        return full_name;
      } else {
        std::string msg = "\nkintera::find_resource::" + name + "not found";
        throw std::runtime_error(msg.c_str());
      }
    }
  }

  // If this is an absolute path, just look for the file there
  if (islash == 0 || ibslash == 0 ||
      (icolon == 1 && (ibslash == 2 || islash == 2))) {
    std::ifstream fin(name);
    if (fin) {
      return name;
    } else {
      std::string msg = "\nkintera::find_resource::" + name + "not found";
      throw std::runtime_error(msg.c_str());
    }
  }

  // Search the data directories for the input file, and return
  // the full path if a match is found
  size_t nd_ = dirs.size();
  for (size_t i = 0; i < nd_; i++) {
    std::string full_name = dirs[i] + "/" + name;
    std::ifstream fin(full_name);
    if (fin) {
      return full_name;
    }
  }
  std::string msg = "\nResource " + name + " not found in director";
  msg += (nd_ == 1 ? "y " : "ies ");
  msg += describe_resource_directories();
  msg += "\n\n";
  msg += "To fix this problem, either:\n";
  msg += "    a) move the missing files into the local directory;\n";
  msg += "    b) define -DMYPATH= during build\n";
  throw std::runtime_error(msg);
}
}  // namespace kintera
