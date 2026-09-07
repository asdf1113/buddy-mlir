//===- DeepSeekR1RaxSession.cpp - DeepSeek RAX resource session -----------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/DeepSeekR1RaxSession.h"

#include "buddy/runtime/core/RaxExecutionSession.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace buddy {
namespace runtime {
namespace {

struct MappedRegion {
  void *data = nullptr;
  size_t bytes = 0;

  ~MappedRegion() {
    if (data)
      munmap(data, bytes);
  }
};

std::runtime_error systemError(const std::string &message) {
  return std::runtime_error(message + ": " + std::strerror(errno));
}

std::unique_ptr<MappedRegion> mapAnonymous(size_t bytes) {
  if (bytes == 0)
    throw std::runtime_error(
        "DeepSeekR1RaxSession: cannot allocate an empty buffer");
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
  flags |= MAP_NORESERVE;
#endif
  void *data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
  if (data == MAP_FAILED)
    throw systemError("DeepSeekR1RaxSession: mmap buffer failed");
  auto region = std::make_unique<MappedRegion>();
  region->data = data;
  region->bytes = bytes;
  return region;
}

std::unique_ptr<MappedRegion> mapFile(const std::string &path) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0)
    throw systemError("DeepSeekR1RaxSession: cannot open parameter pack " +
                      path);

  struct stat status{};
  if (fstat(fd, &status) != 0) {
    const int savedErrno = errno;
    close(fd);
    errno = savedErrno;
    throw systemError("DeepSeekR1RaxSession: cannot stat parameter pack " +
                      path);
  }
  if (status.st_size <= 0) {
    close(fd);
    throw std::runtime_error("DeepSeekR1RaxSession: parameter pack is empty: " +
                             path);
  }
  if (static_cast<uintmax_t>(status.st_size) >
      static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
    close(fd);
    throw std::runtime_error(
        "DeepSeekR1RaxSession: parameter pack is too large: " + path);
  }

  const size_t bytes = static_cast<size_t>(status.st_size);
  void *data = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  const int savedErrno = errno;
  close(fd);
  if (data == MAP_FAILED) {
    errno = savedErrno;
    throw systemError("DeepSeekR1RaxSession: mmap parameter pack failed for " +
                      path);
  }

  auto region = std::make_unique<MappedRegion>();
  region->data = data;
  region->bytes = bytes;
  return region;
}

size_t elementBytes(rhal::rax::DType dtype) {
  switch (dtype) {
  // The current RAX schema has no i1 value. DeepSeek attention-mask buffers
  // therefore arrive as Invalid, while their generated shim ABI uses bool.
  case rhal::rax::DType_Invalid:
  case rhal::rax::DType_I8:
  case rhal::rax::DType_U8:
    return 1;
  case rhal::rax::DType_I16:
  case rhal::rax::DType_U16:
  case rhal::rax::DType_F16:
  case rhal::rax::DType_BF16:
    return 2;
  case rhal::rax::DType_I32:
  case rhal::rax::DType_U32:
  case rhal::rax::DType_F32:
    return 4;
  case rhal::rax::DType_I64:
  case rhal::rax::DType_U64:
  case rhal::rax::DType_F64:
    return 8;
  default:
    throw std::runtime_error("DeepSeekR1RaxSession: unsupported buffer dtype");
  }
}

size_t bufferBytes(const ModelManifest::RaxBuffer &buffer) {
  size_t elements = 1;
  for (int64_t dimension : buffer.shape) {
    if (dimension <= 0)
      throw std::runtime_error("DeepSeekR1RaxSession: buffer " + buffer.name +
                               " has a non-static shape");
    if (elements >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(dimension))
      throw std::runtime_error("DeepSeekR1RaxSession: buffer " + buffer.name +
                               " size overflows size_t");
    elements *= static_cast<size_t>(dimension);
  }
  const size_t scalarBytes = elementBytes(buffer.dtype);
  if (elements > std::numeric_limits<size_t>::max() / scalarBytes)
    throw std::runtime_error("DeepSeekR1RaxSession: buffer " + buffer.name +
                             " size overflows size_t");
  return elements * scalarBytes;
}

} // namespace

struct DeepSeekR1RaxSession::Impl {
  explicit Impl(const std::string &raxPath) : execution(raxPath) {
    initialize();
  }

  Impl(const std::string &raxPath, Communicator &communicator)
      : execution(raxPath, communicator) {
    initialize();
  }

  void initialize() {
    const ModelManifest &manifest = execution.manifest();
    for (const auto &buffer : manifest.buffers) {
      if (buffers.count(buffer.id) || bufferNames.count(buffer.name))
        throw std::runtime_error(
            "DeepSeekR1RaxSession: duplicate buffer resource");
      auto storage = mapAnonymous(bufferBytes(buffer));
      execution.bindBuffer(buffer.id, storage->data);
      bufferNames.emplace(buffer.name, buffer.id);
      buffers.emplace(buffer.id, std::move(storage));
    }
    for (const auto &constant : manifest.constants) {
      if (constantNames.count(constant.name))
        throw std::runtime_error(
            "DeepSeekR1RaxSession: duplicate constant resource");
      constantNames.emplace(constant.name, constant.id);
      if (constant.storage == rhal::rax::ConstantStorage_External)
        bindParameterPack(constant.id, constant.path);
    }
  }

  uint32_t bufferId(const std::string &name) const {
    auto found = bufferNames.find(name);
    if (found == bufferNames.end())
      throw std::runtime_error("DeepSeekR1RaxSession: unknown buffer " + name);
    return found->second;
  }

  uint32_t constantId(const std::string &name) const {
    auto found = constantNames.find(name);
    if (found == constantNames.end())
      throw std::runtime_error("DeepSeekR1RaxSession: unknown constant " +
                               name);
    return found->second;
  }

  MappedRegion &buffer(uint32_t id) {
    auto found = buffers.find(id);
    if (found == buffers.end())
      throw std::runtime_error("DeepSeekR1RaxSession: unknown buffer ID " +
                               std::to_string(id));
    return *found->second;
  }

  const MappedRegion &buffer(uint32_t id) const {
    auto found = buffers.find(id);
    if (found == buffers.end())
      throw std::runtime_error("DeepSeekR1RaxSession: unknown buffer ID " +
                               std::to_string(id));
    return *found->second;
  }

  bool isRuntimeInput(uint32_t id) const {
    for (const auto &function : execution.manifest().functions)
      for (uint32_t input : function.inputs)
        if (input == id)
          return true;
    return false;
  }

  void copyBuffer(uint32_t id, const void *data, size_t bytes) {
    MappedRegion &destination = buffer(id);
    if (bytes > destination.bytes)
      throw std::runtime_error(
          "DeepSeekR1RaxSession: source is larger than buffer ID " +
          std::to_string(id));
    if (bytes != 0 && !data)
      throw std::runtime_error("DeepSeekR1RaxSession: null buffer source");
    if (bytes != 0)
      std::memcpy(destination.data, data, bytes);
  }

  const ModelManifest::RaxFunction *function(const std::string &name) const {
    for (const auto &candidate : execution.manifest().functions)
      if (candidate.name == name)
        return &candidate;
    return nullptr;
  }

  void copyBuffer(uint32_t sourceId, uint32_t destinationId) {
    const MappedRegion &source = buffer(sourceId);
    MappedRegion &destination = buffer(destinationId);
    if (source.bytes != destination.bytes)
      throw std::runtime_error(
          "DeepSeekR1RaxSession: KV cache buffer sizes do not match");
    std::memcpy(destination.data, source.data, source.bytes);
  }

  size_t deepSeekKVLayerCount() const {
    const auto *prefill = function("forward_prefill");
    const auto *decode = function("forward_decode");
    if (!prefill || !decode || decode->inputs.empty() ||
        (decode->inputs.size() - 1) % 3 != 0)
      return 0;
    const size_t layers = (decode->inputs.size() - 1) / 3;
    if (prefill->outputs.size() != 2 * layers + 1 ||
        decode->outputs.size() != 3 * layers + 1)
      return 0;
    return layers;
  }

  void copyPrefillKVCacheToDecodeInputs() {
    const size_t layers = deepSeekKVLayerCount();
    if (layers == 0)
      return;
    const auto *prefill = function("forward_prefill");
    const auto *decode = function("forward_decode");
    for (size_t layer = 0; layer < layers; ++layer) {
      copyBuffer(prefill->outputs[2 * layer], decode->inputs[3 * layer + 2]);
      copyBuffer(prefill->outputs[2 * layer + 1],
                 decode->inputs[3 * layer + 3]);
    }
  }

  void copyDecodeKVCacheToInputs() {
    const size_t layers = deepSeekKVLayerCount();
    if (layers == 0)
      return;
    const auto *decode = function("forward_decode");
    for (size_t layer = 0; layer < layers; ++layer) {
      copyBuffer(decode->outputs[3 * layer + 1], decode->inputs[3 * layer + 2]);
      copyBuffer(decode->outputs[3 * layer + 2], decode->inputs[3 * layer + 3]);
    }
  }

  void bindParameterPack(uint32_t id, const std::string &path) {
    bool known = false;
    for (const auto &constant : execution.manifest().constants) {
      if (constant.id == id) {
        known = true;
        break;
      }
    }
    if (!known)
      throw std::runtime_error("DeepSeekR1RaxSession: unknown constant ID " +
                               std::to_string(id));
    auto mapping = mapFile(path);
    execution.bindConstant(id, mapping->data);
    constants[id] = std::move(mapping);
  }

  RaxExecutionSession execution;
  std::unordered_map<uint32_t, std::unique_ptr<MappedRegion>> buffers;
  std::unordered_map<uint32_t, std::unique_ptr<MappedRegion>> constants;
  std::unordered_map<std::string, uint32_t> bufferNames;
  std::unordered_map<std::string, uint32_t> constantNames;
};

DeepSeekR1RaxSession::DeepSeekR1RaxSession(const std::string &raxPath)
    : impl_(std::make_unique<Impl>(raxPath)) {}

DeepSeekR1RaxSession::DeepSeekR1RaxSession(const std::string &raxPath,
                                           Communicator &communicator)
    : impl_(std::make_unique<Impl>(raxPath, communicator)) {}

DeepSeekR1RaxSession::~DeepSeekR1RaxSession() = default;

void DeepSeekR1RaxSession::bindParameterPack(uint32_t id,
                                             const std::string &path) {
  impl_->bindParameterPack(id, path);
}

void DeepSeekR1RaxSession::bindParameterPack(const std::string &name,
                                             const std::string &path) {
  bindParameterPack(impl_->constantId(name), path);
}

void DeepSeekR1RaxSession::bindRuntimeInput(uint32_t id, const void *data,
                                            size_t bytes) {
  if (!impl_->isRuntimeInput(id))
    throw std::runtime_error("DeepSeekR1RaxSession: buffer ID " +
                             std::to_string(id) + " is not a forward input");
  impl_->copyBuffer(id, data, bytes);
}

void DeepSeekR1RaxSession::bindRuntimeInput(const std::string &name,
                                            const void *data, size_t bytes) {
  bindRuntimeInput(impl_->bufferId(name), data, bytes);
}

void DeepSeekR1RaxSession::bindKVCacheBuffer(uint32_t id, const void *data,
                                             size_t bytes) {
  impl_->copyBuffer(id, data, bytes);
}

void DeepSeekR1RaxSession::bindKVCacheBuffer(const std::string &name,
                                             const void *data, size_t bytes) {
  bindKVCacheBuffer(impl_->bufferId(name), data, bytes);
}

void DeepSeekR1RaxSession::forwardPrefill() {
  impl_->execution.execute("forward_prefill");
  impl_->copyPrefillKVCacheToDecodeInputs();
}

void DeepSeekR1RaxSession::forwardDecode() {
  impl_->execution.execute("forward_decode");
  impl_->copyDecodeKVCacheToInputs();
}

void *DeepSeekR1RaxSession::bufferData(uint32_t id) {
  return impl_->buffer(id).data;
}

void *DeepSeekR1RaxSession::bufferData(const std::string &name) {
  return bufferData(impl_->bufferId(name));
}

size_t DeepSeekR1RaxSession::bufferSize(uint32_t id) const {
  return impl_->buffer(id).bytes;
}

size_t DeepSeekR1RaxSession::bufferSize(const std::string &name) const {
  return bufferSize(impl_->bufferId(name));
}

const ModelManifest &DeepSeekR1RaxSession::manifest() const {
  return impl_->execution.manifest();
}

} // namespace runtime
} // namespace buddy
