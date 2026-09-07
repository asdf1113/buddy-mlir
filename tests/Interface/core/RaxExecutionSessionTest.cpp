// RUN: buddy-rax-execution-session-test %t.rax

#include "buddy/runtime/core/RaxExecutionSession.h"
#include "buddy/runtime/models/DeepSeekR1RaxSession.h"
#include "buddy/runtime/rax/RAX.h"

#include "flatbuffers/flatbuffers.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rhal::rax;

namespace {

void writeParameterPack(const std::string &path) {
  const int parameter = 5;
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(&parameter), sizeof(parameter));
  if (!output)
    throw std::runtime_error("failed to write test parameter pack");
}

void writeTestRax(const std::string &path, const std::string &parameterPath) {
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<int64_t> shape = {1};
  const std::vector<int64_t> strides = {1};
  auto type = CreateTensorType(
      builder, DType_I32, CreateShape(builder, builder.CreateVector(shape)),
      Layout_RowMajor, builder.CreateVector(strides));

  const std::vector<flatbuffers::Offset<BufferBinding>> buffers = {
      CreateBufferBinding(builder, 1,
                          builder.CreateString("forward_decode__input0"), type,
                          MemorySpace_Host, 0),
      CreateBufferBinding(builder, 2,
                          builder.CreateString("forward_decode__kv_cache0"),
                          type, MemorySpace_Host, 0),
      CreateBufferBinding(builder, 3,
                          builder.CreateString("forward_decode__output0"), type,
                          MemorySpace_Host, 0)};

  auto constant = CreateConstant(
      builder, 7, builder.CreateString("rank0_params_float32"), type,
      ConstantStorage_External, 0,
      builder.CreateString("file:" +
                           std::filesystem::absolute(parameterPath).string()),
      builder.CreateString(""), 0);
  const std::string libraryUri = "file:" RAX_EXECUTION_SESSION_TEST_LIBRARY;
  auto codeObject = CreateCodeObject(
      builder, 10, builder.CreateString("session_dispatch"),
      CodeObjectKind_HostSharedLib, 0, builder.CreateString(libraryUri),
      builder.CreateString("deepseek_session_kernel"),
      builder.CreateString("cpu"), 0);

  const std::vector<flatbuffers::Offset<Arg>> arguments = {
      CreateArg(builder, 0, 7, 0, 0), CreateArg(builder, 1, 0, 0, 0),
      CreateArg(builder, 2, 0, 0, 0), CreateArg(builder, 3, 0, 0, 0)};
  auto dispatch =
      CreateDispatchOp(builder, 10, builder.CreateVector(arguments), 0, 0);
  const std::vector<flatbuffers::Offset<Op>> operations = {
      CreateOp(builder, OpKind_Dispatch, dispatch, 0, 0, 0, 0, 0)};
  const std::vector<uint32_t> inputs = {1, 2};
  const std::vector<uint32_t> outputs = {3};
  auto prefill = CreateFunction(
      builder, builder.CreateString("forward_prefill"),
      builder.CreateVector(inputs), builder.CreateVector(outputs), 0,
      builder.CreateVector(operations), 0);
  auto decode = CreateFunction(builder, builder.CreateString("forward_decode"),
                               builder.CreateVector(inputs),
                               builder.CreateVector(outputs), 0,
                               builder.CreateVector(operations), 0);

  const std::vector<flatbuffers::Offset<Constant>> constants = {constant};
  const std::vector<flatbuffers::Offset<CodeObject>> codeObjects = {codeObject};
  const std::vector<flatbuffers::Offset<Function>> functions = {prefill,
                                                                decode};
  auto module = CreateModule(
      builder, builder.CreateString("RAX"), CreateVersion(builder, 0, 1, 0),
      Endianness_Little, 0, 0, builder.CreateVector(buffers),
      builder.CreateVector(constants), builder.CreateVector(codeObjects),
      builder.CreateVector(functions));
  builder.Finish(module, "RAX0");

  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(builder.GetBufferPointer()),
               builder.GetSize());
  if (!output)
    throw std::runtime_error("failed to write test RAX file");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: buddy-rax-execution-session-test <output.rax>\n";
    return 1;
  }

  try {
    const std::string parameterPath = std::string(argv[1]) + ".params";
    writeParameterPack(parameterPath);
    writeTestRax(argv[1], parameterPath);

    int parameter = 5;
    int input = 2;
    int kvCache = 3;
    int output = 0;
    buddy::runtime::RaxExecutionSession genericSession(argv[1]);
    genericSession.bindConstant(7, &parameter);
    genericSession.bindBuffer(1, &input);
    genericSession.bindBuffer(2, &kvCache);
    genericSession.bindBuffer(3, &output);
    genericSession.execute("forward_prefill");
    if (output != 10)
      throw std::runtime_error("generic session did not dispatch");

    buddy::runtime::DeepSeekR1RaxSession deepSeekSession(argv[1]);
    deepSeekSession.bindRuntimeInput("forward_decode__input0", &input,
                                     sizeof(input));
    deepSeekSession.bindKVCacheBuffer("forward_decode__kv_cache0", &kvCache,
                                      sizeof(kvCache));
    deepSeekSession.forwardDecode();
    const auto *ownedOutput = static_cast<const int *>(
        deepSeekSession.bufferData("forward_decode__output0"));
    if (*ownedOutput != 10)
      throw std::runtime_error("DeepSeek RAX session did not dispatch");
    if (deepSeekSession.bufferSize("forward_decode__kv_cache0") != sizeof(int))
      throw std::runtime_error("DeepSeek RAX session buffer size is wrong");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
