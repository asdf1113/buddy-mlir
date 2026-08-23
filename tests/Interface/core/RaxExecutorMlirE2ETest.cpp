// RUN: rax-pack %rax_executor_mlir_e2e_rhal -o %t.rax
// RUN: buddy-rax-executor-mlir-e2e-test %t.rax

#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/core/RaxExecutor.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using MemRef1D = StridedMemRefType<float, 1>;

static MemRef1D makeDescriptor(float *data, int64_t size) {
  return {data, data, 0, {size}, {1}};
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: buddy-rax-executor-mlir-e2e-test <input.rax>\n";
    return 1;
  }

  try {
    std::array<float, 2> params = {2.0f, 3.0f};
    std::array<float, 4> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::array<float, 4> scratch = {};
    std::array<float, 4> output = {};
    MemRef1D paramsDescriptor = makeDescriptor(params.data(), params.size());
    MemRef1D inputDescriptor = makeDescriptor(input.data(), input.size());
    MemRef1D scratchDescriptor = makeDescriptor(scratch.data(), scratch.size());
    MemRef1D outputDescriptor = makeDescriptor(output.data(), output.size());

    auto manifest = buddy::runtime::ModelManifest::loadFromRax(argv[1]);
    buddy::runtime::RaxExecutor executor(manifest);
    executor.bindConstant(7, &paramsDescriptor);
    executor.bindBuffer(1, &inputDescriptor);
    executor.bindBuffer(2, &scratchDescriptor);
    executor.bindBuffer(3, &outputDescriptor);
    executor.execute("forward");

    constexpr std::array<float, 4> expected = {5.0f, 7.0f, 9.0f, 11.0f};
    for (size_t i = 0; i < output.size(); ++i) {
      if (std::fabs(output[i] - expected[i]) > 1.0e-6f)
        throw std::runtime_error("unexpected MLIR pipeline output");
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
