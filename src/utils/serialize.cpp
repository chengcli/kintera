// torch
#include <torch/serialize.h>
#include <torch/script.h>

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
  // get keys
  torch::jit::Module m = torch::jit::load(filename);

  for (const auto& p : m.named_parameters(/*recurse=*/true)) {
    tensor_map[p.name] = p.value;
  }

  for (const auto& b : m.named_buffers(/*recurse=*/true)) {
    tensor_map[p.name] = p.value;
  }

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
