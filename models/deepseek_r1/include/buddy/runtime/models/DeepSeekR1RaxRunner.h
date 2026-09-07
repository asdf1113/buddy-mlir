//===- DeepSeekR1RaxRunner.h - DeepSeek RAX execution entry ----*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_MODELS_DEEPSEEKR1RAXRUNNER_H
#define BUDDY_RUNTIME_MODELS_DEEPSEEKR1RAXRUNNER_H

#include <string>

namespace buddy {
namespace runtime {

/// Execute one prefill and one decode step from rank-local DeepSeek RAX files.
/// The input path locates the artifact directory; each process opens
/// rank<mpi-rank>.rax from that directory.
void runDeepSeekR1Rax(const std::string &raxPath, int tensorParallelSize);

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_DEEPSEEKR1RAXRUNNER_H
