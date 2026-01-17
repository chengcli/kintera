// torch
#include <torch/serialize.h>

// kintera
#include "serialize.hpp"

namespace kintera {

void save_tensors(const std::map<std::string, torch::Tensor>& tensor_map,
                  const std::string& filename) {
  torch::serialize::OutputArchive archive;
  for (const auto& pair : tensor_map) {
    archive.write(pair.first, pair.second);
  }
  archive.save_to(filename);
}

void load_tensors(std::map<std::string, torch::Tensor>& tensor_map,
                  const std::string& filename) {
  torch::serialize::InputArchive archive;
  archive.load_from(filename);
  for (auto& pair : tensor_map) {
    try {
      archive.read(pair.first, pair.second);
    } catch (const c10::Error& e) {
      // skip missing tensors
    }
  }
}

}  // namespace kintera
