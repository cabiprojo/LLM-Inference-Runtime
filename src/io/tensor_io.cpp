#include "tensor_io.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>

std::unordered_map<std::string, Tensor> load_tensors(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Could not open tensor file: " + path);

    int32_t num_tensors = 0;
    in.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));

    std::unordered_map<std::string, Tensor> tensors;
    tensors.reserve(static_cast<size_t>(num_tensors));

    for (int32_t t = 0; t < num_tensors; ++t) {
        int32_t name_len = 0;
        in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(static_cast<size_t>(name_len), '\0');
        in.read(name.data(), name_len);

        int32_t ndim = 0;
        in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

        Tensor tensor;
        tensor.shape.resize(static_cast<size_t>(ndim));
        in.read(reinterpret_cast<char*>(tensor.shape.data()), ndim * sizeof(int32_t));

        size_t numel = 1;
        for (int dim : tensor.shape) numel *= static_cast<size_t>(dim);

        tensor.data.resize(numel);
        in.read(reinterpret_cast<char*>(tensor.data.data()), numel * sizeof(float));

        if (!in) throw std::runtime_error("Unexpected EOF while reading tensor: " + name);

        tensors.emplace(std::move(name), std::move(tensor));
    }

    return tensors;
}

void save_tensors(const std::string& path, const std::unordered_map<std::string, Tensor>& tensors) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Could not open file for writing: " + path);

    int32_t num_tensors = static_cast<int32_t>(tensors.size());
    out.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));

    for (const auto& [name, tensor] : tensors) {
        int32_t name_len = static_cast<int32_t>(name.size());
        out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        out.write(name.data(), name_len);

        int32_t ndim = static_cast<int32_t>(tensor.shape.size());
        out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        out.write(reinterpret_cast<const char*>(tensor.shape.data()), ndim * sizeof(int32_t));

        out.write(reinterpret_cast<const char*>(tensor.data.data()),
                   static_cast<std::streamsize>(tensor.data.size() * sizeof(float)));
    }
}
