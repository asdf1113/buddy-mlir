// RUN: buddy-rax-wrapper-abi-test %t.rax

#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/core/RaxExecutor.h"
#include "buddy/runtime/rax/RAX.h"

#include "flatbuffers/flatbuffers.h"

#include <dlfcn.h>
#include <sys/mman.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rhal::rax;

template <typename T, int Rank> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

using MemRef1DF32 = MemRefDescriptor<float, 1>;
using MemRef3DF32 = MemRefDescriptor<float, 3>;
using WrapperFn = void (*)(MemRef1DF32 *, MemRef3DF32 *, MemRef3DF32 *,
                           MemRef3DF32 *, MemRef3DF32 *);

static constexpr int64_t kElements = 1536;
static constexpr int64_t kParameterElements = 1121961536;
static constexpr int64_t kBiasOffset = 236128768;

static MemRef1DF32 makeParameters(float *payload) {
  return {payload, payload, 0, {kParameterElements}, {1}};
}

static MemRef3DF32 makeActivation(float *payload) {
  return {payload, payload, 0, {1, 1, kElements}, {kElements, kElements, 1}};
}

static void writeTestRax(const std::string &path,
                         const std::string &libraryPath) {
  flatbuffers::FlatBufferBuilder builder;
  const std::string uri = "file:" + libraryPath;

  auto codeObject = CreateCodeObject(
      builder, 17, builder.CreateString("forward_decode_layer0_seg1"),
      CodeObjectKind_HostSharedLib, 0, builder.CreateString(uri),
      builder.CreateString("rax_forward_decode_layer0_seg1"),
      builder.CreateString("cpu"), 0);

  std::vector<flatbuffers::Offset<Arg>> arguments = {
      CreateArg(builder, 0, 7, 0, 0), CreateArg(builder, 1, 0, 0, 0),
      CreateArg(builder, 2, 0, 0, 0), CreateArg(builder, 3, 0, 0, 0),
      CreateArg(builder, 4, 0, 0, 0)};
  auto dispatch =
      CreateDispatchOp(builder, 17, builder.CreateVector(arguments), 0, 0);
  std::vector<flatbuffers::Offset<Op>> operations = {
      CreateOp(builder, OpKind_Dispatch, dispatch, 0, 0, 0, 0, 0)};
  std::vector<uint32_t> inputs = {1, 2};
  std::vector<uint32_t> outputs = {3, 4};
  auto function = CreateFunction(
      builder, builder.CreateString("forward_decode_layer0_seg1"),
      builder.CreateVector(inputs), builder.CreateVector(outputs), 0,
      builder.CreateVector(operations), 0);

  auto module = CreateModule(builder, builder.CreateString("RAX"),
                             CreateVersion(builder, 0, 1, 0), Endianness_Little,
                             0, 0, 0, 0, builder.CreateVector(&codeObject, 1),
                             builder.CreateVector(&function, 1));
  builder.Finish(module, "RAX0");

  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(builder.GetBufferPointer()),
               builder.GetSize());
  if (!output)
    throw std::runtime_error("failed to write Stage 5A test RAX file");
}

static void checkOutput(const std::vector<float> &actual,
                        const std::vector<float> &expected, const char *label) {
  for (int64_t i = 0; i < kElements; ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
        std::abs(actual[i] - expected[i]) > 1.0e-6f)
      throw std::runtime_error(std::string(label) + " mismatch at element " +
                               std::to_string(i));
  }
}

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) {
    std::cerr << "usage: buddy-rax-wrapper-abi-test <output.rax> "
                 "[real-artifact-library]\n";
    return 1;
  }
  const std::string libraryPath =
      argc == 3 ? argv[2] : RAX_WRAPPER_ABI_TEST_LIBRARY;
  const bool checkFixtureFormula = argc == 2;

  const size_t parameterBytes =
      static_cast<size_t>(kParameterElements) * sizeof(float);
  void *mapping = mmap(nullptr, parameterBytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (mapping == MAP_FAILED) {
    std::cerr << "failed to reserve sparse parameter-pack mapping\n";
    return 1;
  }

  try {
    auto *parameters = static_cast<float *>(mapping);
    std::vector<float> input0(kElements);
    std::vector<float> input1(kElements);
    std::vector<float> expected0(kElements);
    std::vector<float> expected1(kElements);
    std::vector<float> direct0(kElements);
    std::vector<float> direct1(kElements);
    std::vector<float> dispatched0(kElements,
                                   std::numeric_limits<float>::quiet_NaN());
    std::vector<float> dispatched1(kElements,
                                   std::numeric_limits<float>::quiet_NaN());

    for (int64_t i = 0; i < kElements; ++i) {
      input0[i] = static_cast<float>(i % 29) * 0.125f;
      input1[i] = static_cast<float>(i % 17) * -0.0625f;
      parameters[kBiasOffset + i] = static_cast<float>(i % 11) * 0.03125f;
      expected0[i] = input0[i] + input1[i] + parameters[kBiasOffset + i];
      expected1[i] = input0[i] - input1[i] + parameters[kBiasOffset + i];
    }

    void *handle = dlopen(libraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      throw std::runtime_error(std::string("direct dlopen failed: ") +
                               dlerror());
    auto direct = reinterpret_cast<WrapperFn>(
        dlsym(handle, "_mlir_ciface_forward_decode_layer0_seg1"));
    if (!direct) {
      dlclose(handle);
      throw std::runtime_error("direct wrapper symbol was not exported");
    }

    MemRef1DF32 parameterDescriptor = makeParameters(parameters);
    MemRef3DF32 input0Descriptor = makeActivation(input0.data());
    MemRef3DF32 input1Descriptor = makeActivation(input1.data());
    MemRef3DF32 direct0Descriptor = makeActivation(direct0.data());
    MemRef3DF32 direct1Descriptor = makeActivation(direct1.data());
    direct(&parameterDescriptor, &input0Descriptor, &input1Descriptor,
           &direct0Descriptor, &direct1Descriptor);
    dlclose(handle);

    if (checkFixtureFormula) {
      checkOutput(direct0, expected0, "direct output0");
      checkOutput(direct1, expected1, "direct output1");
    }

    writeTestRax(argv[1], libraryPath);
    auto manifest = buddy::runtime::ModelManifest::loadFromRax(argv[1]);
    buddy::runtime::RaxExecutor executor(manifest);
    executor.bindConstant(7, parameters);
    executor.bindBuffer(1, input0.data());
    executor.bindBuffer(2, input1.data());
    executor.bindBuffer(3, dispatched0.data());
    executor.bindBuffer(4, dispatched1.data());
    executor.execute("forward_decode_layer0_seg1");

    checkOutput(dispatched0, direct0, "RaxExecutor output0");
    checkOutput(dispatched1, direct1, "RaxExecutor output1");
    if (checkFixtureFormula) {
      checkOutput(dispatched0, expected0, "caller output0");
      checkOutput(dispatched1, expected1, "caller output1");
    }
  } catch (const std::exception &error) {
    munmap(mapping, parameterBytes);
    std::cerr << error.what() << '\n';
    return 1;
  }

  munmap(mapping, parameterBytes);
  return 0;
}
