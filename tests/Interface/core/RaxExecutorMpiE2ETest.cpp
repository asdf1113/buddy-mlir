// REQUIRES: mpi
// RUN: rax-pack %rax_executor_mpi_e2e_rank0_rhal -o %t.rank0.rax
// RUN: rax-pack %rax_executor_mpi_e2e_rank1_rhal -o %t.rank1.rax
// RUN: %mpiexec %mpi_numproc_flag 2 %mpi_preflags \
// RUN:   buddy-rax-executor-mpi-e2e-test %mpi_postflags \
// RUN:   %t.rank0.rax %t.rank1.rax

#include "buddy/runtime/communication/MpiCommunicator.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/core/RaxExecutor.h"
#include "buddy/runtime/rax/RAX.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using MemRef1D = StridedMemRefType<float, 1>;
using RaxOperation = buddy::runtime::ModelManifest::RaxOperation;

namespace {

MemRef1D makeDescriptor(float *data, int64_t size) {
  return {data, data, 0, {size}, {1}};
}

void requireValues(const std::array<float, 2> &actual,
                   const std::array<float, 2> &expected,
                   const std::string &name) {
  for (size_t i = 0; i < actual.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > 1.0e-6f)
      throw std::runtime_error(name + " mismatch at element " +
                               std::to_string(i));
  }
}

void requireDispatch(const RaxOperation &op, uint32_t codeObjectId) {
  if (op.kind != rhal::rax::OpKind_Dispatch || op.codeObjectId != codeObjectId)
    throw std::runtime_error("unexpected Dispatch schedule");
}

void requireBroadcast(const RaxOperation &op) {
  if (op.kind != rhal::rax::OpKind_Collective ||
      op.collectiveKind != rhal::rax::CollectiveKind_Broadcast ||
      op.root != 0 || op.collectiveOperands.size() != 2 ||
      op.collectiveOperands[0].inputBufferId != 1 ||
      op.collectiveOperands[0].outputBufferId != 1 ||
      op.collectiveOperands[1].inputBufferId != 2 ||
      op.collectiveOperands[1].outputBufferId != 2)
    throw std::runtime_error("unexpected Broadcast schedule");
}

void requireAllReduce(const RaxOperation &op) {
  if (op.kind != rhal::rax::OpKind_Collective ||
      op.collectiveKind != rhal::rax::CollectiveKind_AllReduce ||
      op.reductionKind != rhal::rax::ReductionKind_Sum ||
      op.collectiveOperands.size() != 1 ||
      op.collectiveOperands[0].inputBufferId != 4 ||
      op.collectiveOperands[0].outputBufferId != 4)
    throw std::runtime_error("unexpected AllReduce schedule");
}

void requireSchedule(const buddy::runtime::ModelManifest &manifest, int rank) {
  if (manifest.functions.size() != 1 || manifest.functions[0].name != "forward")
    throw std::runtime_error("unexpected RAX function");
  const std::vector<RaxOperation> &ops = manifest.functions[0].ops;
  const size_t expectedSize = rank == 0 ? 5 : 4;
  if (ops.size() != expectedSize)
    throw std::runtime_error("unexpected rank-local operation count");

  size_t index = 0;
  if (rank == 0)
    requireDispatch(ops[index++], 10);
  requireBroadcast(ops[index++]);
  requireDispatch(ops[index++], 11);
  requireAllReduce(ops[index++]);
  requireDispatch(ops[index], 12);
}

} // namespace

int main(int argc, char **argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "MPI_Init failed\n";
    return 1;
  }

  int result = 0;
  try {
    if (argc != 3)
      throw std::runtime_error(
          "usage: buddy-rax-executor-mpi-e2e-test <rank0.rax> <rank1.rax>");

    buddy::runtime::MpiCommunicator communicator(MPI_COMM_WORLD);
    if (communicator.size() != 2)
      throw std::runtime_error("MPI E2E test requires exactly 2 ranks");
    const int rank = communicator.rank();

    std::array<float, 2> scale = {};
    std::array<float, 2> bias = {};
    std::array<float, 2> input = rank == 0 ? std::array<float, 2>{1.0f, 2.0f}
                                           : std::array<float, 2>{10.0f, 20.0f};
    std::array<float, 2> partial = {};
    std::array<float, 2> output = {};

    MemRef1D scaleDescriptor = makeDescriptor(scale.data(), scale.size());
    MemRef1D biasDescriptor = makeDescriptor(bias.data(), bias.size());
    MemRef1D inputDescriptor = makeDescriptor(input.data(), input.size());
    MemRef1D partialDescriptor = makeDescriptor(partial.data(), partial.size());
    MemRef1D outputDescriptor = makeDescriptor(output.data(), output.size());

    auto manifest = buddy::runtime::ModelManifest::loadFromRax(argv[rank + 1]);
    requireSchedule(manifest, rank);
    buddy::runtime::RaxExecutor executor(manifest, communicator);
    executor.bindBuffer(1, &scaleDescriptor);
    executor.bindBuffer(2, &biasDescriptor);
    executor.bindBuffer(3, &inputDescriptor);
    executor.bindBuffer(4, &partialDescriptor);
    executor.bindBuffer(5, &outputDescriptor);
    executor.execute("forward");

    requireValues(scale, {2.0f, 3.0f}, "Broadcast scale");
    requireValues(bias, {1.0f, 1.0f}, "Broadcast bias");
    requireValues(partial, {24.0f, 68.0f}, "AllReduce partial");
    requireValues(output, {25.0f, 69.0f}, "final output");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    result = 1;
  }

  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cerr << "MPI_Finalize failed\n";
    result = 1;
  }
  return result;
}
