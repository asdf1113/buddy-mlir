// REQUIRES: mpi, python-packages
// RUN: rax-inspect %rax_tp_frontend_rank0_rax | FileCheck %s --check-prefix=RAX
// RUN: %mpiexec %mpi_numproc_flag 2 %mpi_preflags \
// RUN:   buddy-rax-tp-frontend-mpi-e2e-test %mpi_postflags \
// RUN:   %rax_tp_frontend_rank0_rax %rax_tp_frontend_rank1_rax \
// RUN:   %rax_tp_frontend_rank0_pack %rax_tp_frontend_rank1_pack
// RAX: @forward
// RAX: Dispatch
// RAX: Collective
// RAX: Dispatch

#include "buddy/runtime/communication/MpiCommunicator.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/core/RaxExecutor.h"
#include "buddy/runtime/rax/RAX.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using MemRef1D = StridedMemRefType<float, 1>;
using MemRef2D = StridedMemRefType<float, 2>;
using RaxOperation = buddy::runtime::ModelManifest::RaxOperation;

namespace {

MemRef1D make1DDescriptor(float *data, int64_t size) {
  return {data, data, 0, {size}, {1}};
}

MemRef2D make2DDescriptor(float *data, int64_t rows, int64_t columns) {
  return {data, data, 0, {rows, columns}, {columns, 1}};
}

std::array<float, 6> loadParameterPack(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("cannot open rank parameter pack " + path);
  if (file.tellg() != static_cast<std::streamoff>(6 * sizeof(float)))
    throw std::runtime_error("rank parameter pack has unexpected size");
  file.seekg(0);
  std::array<float, 6> values{};
  file.read(reinterpret_cast<char *>(values.data()),
            static_cast<std::streamsize>(6 * sizeof(float)));
  if (!file)
    throw std::runtime_error("cannot read rank parameter pack " + path);
  return values;
}

void requireValues(const float *actual, const std::array<float, 2> &expected,
                   const std::string &name) {
  for (size_t i = 0; i < expected.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > 1.0e-6f)
      throw std::runtime_error(name + " mismatch at element " +
                               std::to_string(i));
  }
}

void requirePack(const std::array<float, 6> &actual, int rank) {
  const std::array<float, 6> expected =
      rank == 0 ? std::array<float, 6>{1, 0, 0, 1, 10, 20}
                : std::array<float, 6>{1, 1, 2, 1, 10, 20};
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i])
      throw std::runtime_error("rank parameter pack mismatch at element " +
                               std::to_string(i));
  }
}

void requireSchedule(const buddy::runtime::ModelManifest &manifest) {
  if (manifest.functions.size() != 1 || manifest.functions[0].name != "forward")
    throw std::runtime_error("unexpected Stage 2E RAX function");
  const std::vector<RaxOperation> &ops = manifest.functions[0].ops;
  if (ops.size() != 3 || ops[0].kind != rhal::rax::OpKind_Dispatch ||
      ops[0].codeObjectId != 10 ||
      ops[1].kind != rhal::rax::OpKind_Collective ||
      ops[1].collectiveKind != rhal::rax::CollectiveKind_AllReduce ||
      ops[1].reductionKind != rhal::rax::ReductionKind_Sum ||
      ops[1].collectiveOperands.size() != 1 ||
      ops[1].collectiveOperands[0].inputBufferId != 2 ||
      ops[1].collectiveOperands[0].outputBufferId != 2 ||
      ops[2].kind != rhal::rax::OpKind_Dispatch || ops[2].codeObjectId != 11)
    throw std::runtime_error("unexpected Stage 2E RAX schedule");
}

} // namespace

int main(int argc, char **argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "MPI_Init failed\n";
    return 1;
  }

  int result = 0;
  try {
    if (argc != 5)
      throw std::runtime_error(
          "usage: buddy-rax-tp-frontend-mpi-e2e-test <rank0.rax> "
          "<rank1.rax> <rank0.data> <rank1.data>");

    buddy::runtime::MpiCommunicator communicator(MPI_COMM_WORLD);
    if (communicator.size() != 2)
      throw std::runtime_error("Stage 2E test requires exactly two ranks");
    const int rank = communicator.rank();

    std::array<float, 6> parameters = loadParameterPack(argv[rank + 3]);
    requirePack(parameters, rank);
    std::array<float, 2> input =
        rank == 0 ? std::array<float, 2>{1, 2} : std::array<float, 2>{3, 4};
    std::array<float, 2> boundaryStorage{};
    std::array<float, 2> outputStorage{};
    MemRef1D parameterDescriptor =
        make1DDescriptor(parameters.data(), parameters.size());
    MemRef2D inputDescriptor = make2DDescriptor(input.data(), 1, 2);
    MemRef2D boundaryDescriptor =
        make2DDescriptor(boundaryStorage.data(), 1, 2);
    MemRef2D outputDescriptor = make2DDescriptor(outputStorage.data(), 1, 2);

    auto manifest = buddy::runtime::ModelManifest::loadFromRax(argv[rank + 1]);
    requireSchedule(manifest);
    buddy::runtime::RaxExecutor executor(manifest, communicator);
    executor.bindConstant(7, &parameterDescriptor);
    executor.bindBuffer(1, &inputDescriptor);
    executor.bindBuffer(2, &boundaryDescriptor);
    executor.bindBuffer(3, &outputDescriptor);
    executor.execute("forward");

    requireValues(boundaryDescriptor.data + boundaryDescriptor.offset, {12, 9},
                  "AllReduce boundary");
    requireValues(outputDescriptor.data + outputDescriptor.offset, {22, 29},
                  "final output");
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
