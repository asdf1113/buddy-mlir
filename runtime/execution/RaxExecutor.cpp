//===- RaxExecutor.cpp - Execute host RAX functions -----------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/core/RaxExecutor.h"

#include <dlfcn.h>

#include <stdexcept>
#include <vector>

namespace buddy {
namespace runtime {

RaxExecutor::RaxExecutor(const ModelManifest &manifest) : manifest_(manifest) {}

RaxExecutor::~RaxExecutor() {
  for (const auto &library : libraryHandles_)
    dlclose(library.second);
}

void RaxExecutor::bindBuffer(uint32_t id, void *abiPtr) {
  buffers_[id] = abiPtr;
}

void RaxExecutor::bindConstant(uint32_t id, void *abiPtr) {
  constants_[id] = abiPtr;
}

RaxHostEntryFn RaxExecutor::resolveEntry(uint32_t codeObjectId) {
  auto cachedEntry = entries_.find(codeObjectId);
  if (cachedEntry != entries_.end())
    return cachedEntry->second;

  const ModelManifest::ResolvedCodeObject *codeObject = nullptr;
  for (const auto &candidate : manifest_.codeObjects) {
    if (candidate.id == codeObjectId) {
      codeObject = &candidate;
      break;
    }
  }
  if (!codeObject)
    throw std::runtime_error("RaxExecutor: unknown code object ID " +
                             std::to_string(codeObjectId));
  if (codeObject->kind != rhal::rax::CodeObjectKind_HostSharedLib)
    throw std::runtime_error("RaxExecutor: code object " +
                             std::to_string(codeObjectId) +
                             " is not a HostSharedLib");
  if (codeObject->path.empty())
    throw std::runtime_error("RaxExecutor: code object " +
                             std::to_string(codeObjectId) +
                             " has no shared-library path");
  if (codeObject->entrySymbol.empty())
    throw std::runtime_error("RaxExecutor: code object " +
                             std::to_string(codeObjectId) +
                             " has no entry symbol");

  void *handle = nullptr;
  auto cachedLibrary = libraryHandles_.find(codeObject->path);
  if (cachedLibrary != libraryHandles_.end()) {
    handle = cachedLibrary->second;
  } else {
    handle = dlopen(codeObject->path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      const char *error = dlerror();
      throw std::runtime_error("RaxExecutor: dlopen failed for " +
                               codeObject->path + ": " +
                               (error ? error : "unknown error"));
    }
    libraryHandles_.emplace(codeObject->path, handle);
  }

  dlerror();
  void *symbol = dlsym(handle, codeObject->entrySymbol.c_str());
  const char *error = dlerror();
  if (error || !symbol)
    throw std::runtime_error("RaxExecutor: dlsym failed for " +
                             codeObject->entrySymbol + ": " +
                             (error ? error : "symbol is null"));

  auto entry = reinterpret_cast<RaxHostEntryFn>(symbol);
  entries_.emplace(codeObjectId, entry);
  return entry;
}

void RaxExecutor::execute(const std::string &functionName) {
  const ModelManifest::RaxFunction *function = nullptr;
  for (const auto &candidate : manifest_.functions) {
    if (candidate.name == functionName) {
      function = &candidate;
      break;
    }
  }
  if (!function)
    throw std::runtime_error("RaxExecutor: unknown function " + functionName);

  for (const auto &op : function->ops) {
    if (op.kind == rhal::rax::OpKind_Barrier)
      continue;
    if (op.kind != rhal::rax::OpKind_Dispatch)
      throw std::runtime_error("RaxExecutor: unsupported operation kind " +
                               std::to_string(static_cast<int>(op.kind)));

    std::vector<void *> callArgs;
    callArgs.reserve(op.arguments.size());
    for (const auto &argument : op.arguments) {
      if (argument.kind == ModelManifest::RaxDispatchArgument::Kind::Buffer) {
        auto binding = buffers_.find(argument.resourceId);
        if (binding == buffers_.end())
          throw std::runtime_error("RaxExecutor: unbound buffer ID " +
                                   std::to_string(argument.resourceId));
        callArgs.push_back(binding->second);
      } else {
        auto binding = constants_.find(argument.resourceId);
        if (binding == constants_.end())
          throw std::runtime_error("RaxExecutor: unbound constant ID " +
                                   std::to_string(argument.resourceId));
        callArgs.push_back(binding->second);
      }
    }
    resolveEntry(op.codeObjectId)(callArgs.data());
  }
}

} // namespace runtime
} // namespace buddy
