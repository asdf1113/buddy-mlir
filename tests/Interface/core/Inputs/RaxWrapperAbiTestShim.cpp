//===- RaxWrapperAbiTestShim.cpp - Focused DeepSeek TP RAX bridge ---------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>

template <typename T, int Rank> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

using MemRef1DF32 = MemRefDescriptor<float, 1>;
using MemRef3DF32 = MemRefDescriptor<float, 3>;

extern "C" void _mlir_ciface_forward_decode_layer0_seg1(MemRef1DF32 *parameters,
                                                        MemRef3DF32 *input0,
                                                        MemRef3DF32 *input1,
                                                        MemRef3DF32 *output0,
                                                        MemRef3DF32 *output1);

static MemRef1DF32 makeParameters(void *payload) {
  auto *data = static_cast<float *>(payload);
  return {data, data, 0, {1121961536}, {1}};
}

static MemRef3DF32 makeActivation(void *payload) {
  auto *data = static_cast<float *>(payload);
  return {data, data, 0, {1, 1, 1536}, {1536, 1536, 1}};
}

extern "C" void rax_forward_decode_layer0_seg1(void **args) {
  MemRef1DF32 parameters = makeParameters(args[0]);
  MemRef3DF32 input0 = makeActivation(args[1]);
  MemRef3DF32 input1 = makeActivation(args[2]);
  MemRef3DF32 output0 = makeActivation(args[3]);
  MemRef3DF32 output1 = makeActivation(args[4]);
  _mlir_ciface_forward_decode_layer0_seg1(&parameters, &input0, &input1,
                                          &output0, &output1);
}
