// REQUIRES: mpi
// RUN: rax-pack %S/Inputs/RaxVariableCollectivesMpiE2E.rhal.mlir -o %t.rax
// RUN: rax-inspect %t.rax | FileCheck %s --check-prefix=RAX
// RUN: %mpiexec %mpi_numproc_flag 2 %mpi_preflags \
// RUN:   buddy-rax-variable-collectives-mpi-e2e-test %mpi_postflags %t.rax \
// RUN:   | FileCheck %s --check-prefix=SUCCESS --match-full-lines

// clang-format off
// RAX: code_objects: 0
// RAX-LABEL: functions: 2
// RAX-NEXT: @all_gatherv
// RAX-NEXT: [0] Collective kind=AllGatherV operands=[1->2 recv_counts=[2,2] displacements=[0,2]]
// RAX-NEXT: @reduce_scatter
// RAX-NEXT: [0] Collective kind=ReduceScatter reduction=Sum operands=[3->4 recv_counts=[2,2]]
// SUCCESS: Stage 2F.3 MPI variable collectives passed
// clang-format on

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

template <size_t Size>
void requireValues(const std::array<float, Size> &actual,
                   const std::array<float, Size> &expected,
                   const std::string &name) {
  for (size_t i = 0; i < Size; ++i) {
    if (std::fabs(actual[i] - expected[i]) > 1.0e-6f)
      throw std::runtime_error(name + " mismatch at element " +
                               std::to_string(i));
  }
}

void requireAllGatherV(const RaxOperation &op) {
  if (op.kind != rhal::rax::OpKind_Collective ||
      op.collectiveKind != rhal::rax::CollectiveKind_AllGatherV ||
      op.collectiveOperands.size() != 1 ||
      op.collectiveOperands[0].inputBufferId != 1 ||
      op.collectiveOperands[0].outputBufferId != 2 ||
      op.collectiveOperands[0].recvCounts != std::vector<int64_t>({2, 2}) ||
      op.collectiveOperands[0].displacements != std::vector<int64_t>({0, 2}))
    throw std::runtime_error("unexpected AllGatherV schedule");
}

void requireReduceScatter(const RaxOperation &op) {
  if (op.kind != rhal::rax::OpKind_Collective ||
      op.collectiveKind != rhal::rax::CollectiveKind_ReduceScatter ||
      op.reductionKind != rhal::rax::ReductionKind_Sum ||
      op.collectiveOperands.size() != 1 ||
      op.collectiveOperands[0].inputBufferId != 3 ||
      op.collectiveOperands[0].outputBufferId != 4 ||
      op.collectiveOperands[0].recvCounts != std::vector<int64_t>({2, 2}) ||
      !op.collectiveOperands[0].displacements.empty())
    throw std::runtime_error("unexpected ReduceScatter schedule");
}

void requireSchedule(const buddy::runtime::ModelManifest &manifest) {
  if (manifest.functions.size() != 2 ||
      manifest.functions[0].name != "all_gatherv" ||
      manifest.functions[0].ops.size() != 1 ||
      manifest.functions[1].name != "reduce_scatter" ||
      manifest.functions[1].ops.size() != 1)
    throw std::runtime_error("unexpected RAX function schedule");
  requireAllGatherV(manifest.functions[0].ops[0]);
  requireReduceScatter(manifest.functions[1].ops[0]);
}

} // namespace

int main(int argc, char **argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "MPI_Init failed\n";
    return 1;
  }

  int rank = -1;
  int localResult = 0;
  try {
    if (argc != 2)
      throw std::runtime_error(
          "usage: buddy-rax-variable-collectives-mpi-e2e-test <input.rax>");

    buddy::runtime::MpiCommunicator communicator(MPI_COMM_WORLD);
    if (communicator.size() != 2)
      throw std::runtime_error("MPI E2E test requires exactly 2 ranks");
    rank = communicator.rank();

    std::array<float, 2> allGatherVInput =
        rank == 0 ? std::array<float, 2>{1.0f, 2.0f}
                  : std::array<float, 2>{3.0f, 4.0f};
    std::array<float, 4> allGatherVOutput = {};
    std::array<float, 4> reduceScatterInput =
        rank == 0 ? std::array<float, 4>{1.0f, 2.0f, 3.0f, 4.0f}
                  : std::array<float, 4>{10.0f, 20.0f, 30.0f, 40.0f};
    std::array<float, 2> reduceScatterOutput = {};

    MemRef1D allGatherVInputDescriptor =
        makeDescriptor(allGatherVInput.data(), allGatherVInput.size());
    MemRef1D allGatherVOutputDescriptor =
        makeDescriptor(allGatherVOutput.data(), allGatherVOutput.size());
    MemRef1D reduceScatterInputDescriptor =
        makeDescriptor(reduceScatterInput.data(), reduceScatterInput.size());
    MemRef1D reduceScatterOutputDescriptor =
        makeDescriptor(reduceScatterOutput.data(), reduceScatterOutput.size());

    auto manifest = buddy::runtime::ModelManifest::loadFromRax(argv[1]);
    requireSchedule(manifest);
    buddy::runtime::RaxExecutor executor(manifest, communicator);
    executor.bindBuffer(1, &allGatherVInputDescriptor);
    executor.bindBuffer(2, &allGatherVOutputDescriptor);
    executor.bindBuffer(3, &reduceScatterInputDescriptor);
    executor.bindBuffer(4, &reduceScatterOutputDescriptor);

    executor.execute("all_gatherv");
    requireValues(allGatherVOutput, {1.0f, 2.0f, 3.0f, 4.0f},
                  "AllGatherV output");

    executor.execute("reduce_scatter");
    const std::array<float, 2> expectedReduceScatter =
        rank == 0 ? std::array<float, 2>{11.0f, 22.0f}
                  : std::array<float, 2>{33.0f, 44.0f};
    requireValues(reduceScatterOutput, expectedReduceScatter,
                  "ReduceScatter output");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    localResult = 1;
  }

  int result = localResult;
  if (MPI_Allreduce(&localResult, &result, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS) {
    std::cerr << "MPI result synchronization failed\n";
    result = 1;
  }
  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cerr << "MPI_Finalize failed\n";
    result = 1;
  }
  if (result == 0 && rank == 0)
    std::cout << "Stage 2F.3 MPI variable collectives passed\n";
  return result;
}
