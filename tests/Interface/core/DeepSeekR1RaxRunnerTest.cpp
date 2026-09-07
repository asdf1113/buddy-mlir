// REQUIRES: mpi
// RUN: buddy-deepseek-r1-rax-runner-test --prepare %t
// RUN: buddy-deepseek-r1-rax-runner-test --check-mpi-lifecycle \
// RUN:   %t/rank0.rax 2>&1 | FileCheck %s --check-prefix=LIFECYCLE
// RUN: buddy-deepseek-r1-rax-runner-test --run-singleton \
// RUN:   %t/rank0.rax 2>&1 | FileCheck %s --check-prefix=SINGLETON
// RUN: %mpiexec %mpi_numproc_flag 2 %mpi_preflags \
// RUN:   buddy-deepseek-r1-rax-runner-test %mpi_postflags \
// RUN:   --run %t/rank0.rax 2>&1 | FileCheck %s

// LIFECYCLE: Stage 5D.1 owned MPI lifecycle passed
// SINGLETON: Stage 5D.1 singleton MPI/session wiring passed
// CHECK-DAG: DeepSeek RAX rank 0/2: {{.*}}rank0.rax
// CHECK-DAG: DeepSeek RAX rank 1/2: {{.*}}rank1.rax
// CHECK-DAG: Stage 5D.1 rank0 prefill/decode passed
// CHECK-DAG: Stage 5D.1 rank1 prefill/decode passed

#include "buddy/runtime/models/DeepSeekR1RaxRunner.h"
#include "buddy/runtime/communication/MpiCommunicator.h"
#include "buddy/runtime/models/DeepSeekR1RaxSession.h"
#include "buddy/runtime/rax/RAX.h"

#include "flatbuffers/flatbuffers.h"

#include <mpi.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rhal::rax;

namespace {

void writeRankRax(const std::filesystem::path &path, int rank) {
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<int64_t> shape = {1};
  const std::vector<int64_t> strides = {1};
  auto type = CreateTensorType(
      builder, DType_F32, CreateShape(builder, builder.CreateVector(shape)),
      Layout_RowMajor, builder.CreateVector(strides));

  const std::vector<flatbuffers::Offset<BufferBinding>> buffers = {
      CreateBufferBinding(builder, 1, builder.CreateString("synthetic_input"),
                          type, MemorySpace_Host, 0)};

  const std::string libraryUri = "file:" DEEPSEEK_RAX_RUNNER_TEST_LIBRARY;
  auto prefillCodeObject = CreateCodeObject(
      builder, 10, builder.CreateString("prefill"),
      CodeObjectKind_HostSharedLib, 0, builder.CreateString(libraryUri),
      builder.CreateString(rank == 0 ? "stage5d_rank0_prefill"
                                     : "stage5d_rank1_prefill"),
      builder.CreateString("cpu"), 0);
  auto decodeCodeObject = CreateCodeObject(
      builder, 11, builder.CreateString("decode"), CodeObjectKind_HostSharedLib,
      0, builder.CreateString(libraryUri),
      builder.CreateString(rank == 0 ? "stage5d_rank0_decode"
                                     : "stage5d_rank1_decode"),
      builder.CreateString("cpu"), 0);

  const std::vector<flatbuffers::Offset<Arg>> arguments = {
      CreateArg(builder, 1, 0, 0, 0)};
  auto prefillDispatch =
      CreateDispatchOp(builder, 10, builder.CreateVector(arguments), 0, 0);
  const std::vector<flatbuffers::Offset<Op>> prefillOperations = {
      CreateOp(builder, OpKind_Dispatch, prefillDispatch, 0, 0, 0, 0, 0, 0)};

  auto decodeDispatch =
      CreateDispatchOp(builder, 11, builder.CreateVector(arguments), 0, 0);
  const std::vector<flatbuffers::Offset<Op>> decodeOperations = {
      CreateOp(builder, OpKind_Dispatch, decodeDispatch, 0, 0, 0, 0, 0, 0)};
  const std::vector<uint32_t> functionBuffers = {1};
  auto prefill =
      CreateFunction(builder, builder.CreateString("forward_prefill"),
                     builder.CreateVector(functionBuffers),
                     builder.CreateVector(functionBuffers), 0,
                     builder.CreateVector(prefillOperations), 0);
  auto decode = CreateFunction(builder, builder.CreateString("forward_decode"),
                               builder.CreateVector(functionBuffers),
                               builder.CreateVector(functionBuffers), 0,
                               builder.CreateVector(decodeOperations), 0);

  const std::vector<flatbuffers::Offset<CodeObject>> codeObjects = {
      prefillCodeObject, decodeCodeObject};
  const std::vector<flatbuffers::Offset<Function>> functions = {prefill,
                                                                decode};
  auto module = CreateModule(
      builder, builder.CreateString("RAX"), CreateVersion(builder, 0, 1, 0),
      Endianness_Little, 0, 0, builder.CreateVector(buffers), 0,
      builder.CreateVector(codeObjects), builder.CreateVector(functions));
  builder.Finish(module, "RAX0");

  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(builder.GetBufferPointer()),
               builder.GetSize());
  if (!output)
    throw std::runtime_error("failed to write " + path.string());
}

void checkOwnedMpiLifecycle(const std::string &raxPath) {
  try {
    buddy::runtime::runDeepSeekR1Rax(raxPath, 2);
    throw std::runtime_error("expected MPI world-size mismatch");
  } catch (const std::runtime_error &error) {
    if (std::string(error.what()).find("MPI world size 1") == std::string::npos)
      throw;
  }

  int finalized = 0;
  if (MPI_Finalized(&finalized) != MPI_SUCCESS || !finalized)
    throw std::runtime_error("runner did not finalize its owned MPI state");
  std::cout << "Stage 5D.1 owned MPI lifecycle passed\n";
}

void runSingletonSession(const std::string &raxPath) {
  if (MPI_Init(nullptr, nullptr) != MPI_SUCCESS)
    throw std::runtime_error("MPI_Init failed");

  try {
    buddy::runtime::MpiCommunicator communicator(MPI_COMM_WORLD);
    if (communicator.rank() != 0 || communicator.size() != 1)
      throw std::runtime_error("expected a singleton MPI world");
    buddy::runtime::DeepSeekR1RaxSession session(raxPath, communicator);
    session.forwardPrefill();
    session.forwardDecode();
    const auto *value =
        static_cast<const float *>(session.bufferData("synthetic_input"));
    if (*value != 1.0f)
      throw std::runtime_error("singleton session did not invoke prefill");
  } catch (...) {
    MPI_Finalize();
    throw;
  }

  if (MPI_Finalize() != MPI_SUCCESS)
    throw std::runtime_error("MPI_Finalize failed");
  std::cout << "Stage 5D.1 singleton MPI/session wiring passed\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error(
          "usage: buddy-deepseek-r1-rax-runner-test --prepare <directory> | "
          "--check-mpi-lifecycle <rank0.rax> | "
          "--run-singleton <rank0.rax> | "
          "--run <rank0.rax>");

    if (std::string(argv[1]) == "--prepare") {
      const std::filesystem::path directory = argv[2];
      std::filesystem::create_directories(directory);
      writeRankRax(directory / "rank0.rax", 0);
      writeRankRax(directory / "rank1.rax", 1);
      return 0;
    }
    if (std::string(argv[1]) == "--check-mpi-lifecycle") {
      checkOwnedMpiLifecycle(argv[2]);
      return 0;
    }
    if (std::string(argv[1]) == "--run-singleton") {
      runSingletonSession(argv[2]);
      return 0;
    }
    if (std::string(argv[1]) != "--run")
      throw std::runtime_error(
          "expected --prepare, --check-mpi-lifecycle, --run-singleton, or "
          "--run");

    buddy::runtime::runDeepSeekR1Rax(argv[2], 2);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
