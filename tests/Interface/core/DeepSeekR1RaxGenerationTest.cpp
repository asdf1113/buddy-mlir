// RUN: buddy-deepseek-r1-rax-generation-test %t

#include "buddy/runtime/llm/TextGeneration.h"
#include "buddy/runtime/models/DeepSeekR1RaxSession.h"
#include "buddy/runtime/rax/RAX.h"

#include "flatbuffers/flatbuffers.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rhal::rax;

namespace {

flatbuffers::Offset<TensorType>
makeType(flatbuffers::FlatBufferBuilder &builder, DType dtype,
         const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t i = shape.size(); i > 1; --i)
    strides[i - 2] = strides[i - 1] * shape[i - 1];
  return CreateTensorType(builder, dtype,
                          CreateShape(builder, builder.CreateVector(shape)),
                          Layout_RowMajor, builder.CreateVector(strides));
}

void writeFixture(const std::filesystem::path &path) {
  flatbuffers::FlatBufferBuilder builder;
  auto tokenSequence = makeType(builder, DType_I64, {1, 10});
  auto token = makeType(builder, DType_I64, {1, 1});
  auto scalarI64 = makeType(builder, DType_I64, {1});
  auto scalarI32 = makeType(builder, DType_I32, {1});
  auto prefillLogits = makeType(builder, DType_F32, {1, 10, 4});
  auto decodeLogits = makeType(builder, DType_F32, {1, 1, 4});

  const std::vector<flatbuffers::Offset<BufferBinding>> buffers = {
      CreateBufferBinding(builder, 1,
                          builder.CreateString("forward_prefill__input0"),
                          tokenSequence, MemorySpace_Host, 0),
      CreateBufferBinding(builder, 2,
                          builder.CreateString("forward_prefill__output0"),
                          scalarI32, MemorySpace_Host, 0),
      CreateBufferBinding(builder, 3,
                          builder.CreateString("forward_prefill__output1"),
                          scalarI32, MemorySpace_Host, 0),
      CreateBufferBinding(builder, 4,
                          builder.CreateString("forward_prefill__output2"),
                          prefillLogits, MemorySpace_Host, 0),
      CreateBufferBinding(builder, 5,
                          builder.CreateString("forward_decode__input0"), token,
                          MemorySpace_Host, 0),
      CreateBufferBinding(builder, 6,
                          builder.CreateString("forward_decode__input1"),
                          scalarI64, MemorySpace_Host, 0),
      CreateBufferBinding(builder, 7,
                          builder.CreateString("forward_decode__input2"),
                          scalarI32, MemorySpace_Host, 0),
      CreateBufferBinding(builder, 8,
                          builder.CreateString("forward_decode__input3"),
                          scalarI32, MemorySpace_Host, 0),
      CreateBufferBinding(builder, 9,
                          builder.CreateString("forward_decode__output0"),
                          scalarI64, MemorySpace_Host, 0),
      CreateBufferBinding(builder, 10,
                          builder.CreateString("forward_decode__output1"),
                          scalarI32, MemorySpace_Host, 0),
      CreateBufferBinding(builder, 11,
                          builder.CreateString("forward_decode__output2"),
                          scalarI32, MemorySpace_Host, 0),
      CreateBufferBinding(builder, 12,
                          builder.CreateString("forward_decode__output3"),
                          decodeLogits, MemorySpace_Host, 0)};

  const std::string libraryUri = "file:" DEEPSEEK_RAX_GENERATION_TEST_LIBRARY;
  auto prefillCodeObject = CreateCodeObject(
      builder, 1, builder.CreateString("prefill"), CodeObjectKind_HostSharedLib,
      0, builder.CreateString(libraryUri),
      builder.CreateString("stage5d2_generation_prefill"),
      builder.CreateString("cpu"), 0);
  auto decodeCodeObject = CreateCodeObject(
      builder, 2, builder.CreateString("decode"), CodeObjectKind_HostSharedLib,
      0, builder.CreateString(libraryUri),
      builder.CreateString("stage5d2_generation_decode"),
      builder.CreateString("cpu"), 0);

  const std::vector<flatbuffers::Offset<Arg>> prefillArguments = {
      CreateArg(builder, 1, 0, 0, 0), CreateArg(builder, 2, 0, 0, 0),
      CreateArg(builder, 3, 0, 0, 0), CreateArg(builder, 4, 0, 0, 0)};
  auto prefillDispatch = CreateDispatchOp(
      builder, 1, builder.CreateVector(prefillArguments), 0, 0);
  const std::vector<flatbuffers::Offset<Op>> prefillOperations = {
      CreateOp(builder, OpKind_Dispatch, prefillDispatch, 0, 0, 0, 0, 0, 0)};

  const std::vector<flatbuffers::Offset<Arg>> decodeArguments = {
      CreateArg(builder, 5, 0, 0, 0),  CreateArg(builder, 6, 0, 0, 0),
      CreateArg(builder, 7, 0, 0, 0),  CreateArg(builder, 8, 0, 0, 0),
      CreateArg(builder, 9, 0, 0, 0),  CreateArg(builder, 10, 0, 0, 0),
      CreateArg(builder, 11, 0, 0, 0), CreateArg(builder, 12, 0, 0, 0)};
  auto decodeDispatch =
      CreateDispatchOp(builder, 2, builder.CreateVector(decodeArguments), 0, 0);
  const std::vector<flatbuffers::Offset<Op>> decodeOperations = {
      CreateOp(builder, OpKind_Dispatch, decodeDispatch, 0, 0, 0, 0, 0, 0)};

  const std::vector<uint32_t> prefillInputs = {1};
  const std::vector<uint32_t> prefillOutputs = {2, 3, 4};
  auto prefill = CreateFunction(
      builder, builder.CreateString("forward_prefill"),
      builder.CreateVector(prefillInputs), builder.CreateVector(prefillOutputs),
      0, builder.CreateVector(prefillOperations), 0);
  const std::vector<uint32_t> decodeInputs = {5, 6, 7, 8};
  const std::vector<uint32_t> decodeOutputs = {9, 10, 11, 12};
  auto decode = CreateFunction(builder, builder.CreateString("forward_decode"),
                               builder.CreateVector(decodeInputs),
                               builder.CreateVector(decodeOutputs), 0,
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
    throw std::runtime_error("failed to write generation fixture");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error(
          "usage: buddy-deepseek-r1-rax-generation-test <directory>");
    const std::filesystem::path directory = argv[1];
    std::filesystem::create_directories(directory);
    const auto raxPath = directory / "generation.rax";
    const auto vocabPath = directory / "vocab.txt";
    writeFixture(raxPath);
    {
      std::ofstream vocab(vocabPath);
      vocab << "zero\none\ntwo\nthree\n";
    }

    buddy::runtime::DeepSeekR1RaxSession raxSession(raxPath.string());
    buddy::runtime::LLMSession &session = raxSession;
    buddy::runtime::TextCodec codec;
    codec.tokenize = [](buddy::Text<size_t, 2> &tokens,
                        const std::string &vocab) {
      tokens.tokenizeDeepSeekR1(vocab, 10);
    };
    codec.detokenize = [](buddy::Text<size_t, 2> &) { return std::string(); };
    codec.maxTokenLen = 10;
    buddy::Sampler sampler({});
    const auto result =
        buddy::runtime::runGeneration("", session, vocabPath.string(), 8, {3},
                                      sampler, codec, true, false, false);

    if (result.generatedTokens != 2 || session.position() != 8)
      throw std::runtime_error("generation loop did not perform two decodes");
    if (*static_cast<const int *>(
            raxSession.bufferData("forward_decode__input2")) != 151649)
      throw std::runtime_error("persistent KV state was not reused by decode");
    std::cout << "Stage 5D.2 LLMSession generation contract passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
