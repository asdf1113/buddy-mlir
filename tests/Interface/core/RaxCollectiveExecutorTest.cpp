// RUN: buddy-rax-collective-executor-test %t.rax

#include "buddy/runtime/communication/Communicator.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/core/RaxExecutor.h"
#include "buddy/runtime/rax/RAX.h"

#include "flatbuffers/flatbuffers.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rhal::rax;

namespace {

struct Event {
  enum class Kind { Broadcast, AllReduce, AllGatherV, ReduceScatter } kind;
  const void *sendBuffer = nullptr;
  void *recvBuffer = nullptr;
  size_t amount = 0;
  int root = -1;
  std::vector<int64_t> recvCounts;
  std::vector<int64_t> displacements;
  buddy::runtime::DataType dataType = buddy::runtime::DataType::F32;
  buddy::runtime::ReductionOp reduction = buddy::runtime::ReductionOp::Sum;
};

class FakeCommunicator final : public buddy::runtime::Communicator {
public:
  int rank() const override { return 0; }
  int size() const override { return 2; }

  void broadcast(void *buffer, size_t bytes, int root) override {
    Event event;
    event.kind = Event::Kind::Broadcast;
    event.sendBuffer = buffer;
    event.recvBuffer = buffer;
    event.amount = bytes;
    event.root = root;
    events.push_back(std::move(event));
  }

  void allReduce(const void *sendBuffer, void *recvBuffer, size_t count,
                 buddy::runtime::DataType dataType,
                 buddy::runtime::ReductionOp reduction) override {
    Event event;
    event.kind = Event::Kind::AllReduce;
    event.sendBuffer = sendBuffer;
    event.recvBuffer = recvBuffer;
    event.amount = count;
    event.dataType = dataType;
    event.reduction = reduction;
    events.push_back(std::move(event));
    auto *data = static_cast<float *>(recvBuffer);
    for (size_t i = 0; i < count; ++i)
      data[i] *= 2.0f;
  }

  void allGatherV(const void *sendBuffer, size_t sendCount, void *recvBuffer,
                  const std::vector<int64_t> &recvCounts,
                  const std::vector<int64_t> &displacements,
                  buddy::runtime::DataType dataType) override {
    Event event;
    event.kind = Event::Kind::AllGatherV;
    event.sendBuffer = sendBuffer;
    event.recvBuffer = recvBuffer;
    event.amount = sendCount;
    event.recvCounts = recvCounts;
    event.displacements = displacements;
    event.dataType = dataType;
    events.push_back(std::move(event));
  }

  void reduceScatter(const void *sendBuffer, void *recvBuffer,
                     const std::vector<int64_t> &recvCounts,
                     buddy::runtime::DataType dataType,
                     buddy::runtime::ReductionOp reduction) override {
    Event event;
    event.kind = Event::Kind::ReduceScatter;
    event.sendBuffer = sendBuffer;
    event.recvBuffer = recvBuffer;
    event.recvCounts = recvCounts;
    event.dataType = dataType;
    event.reduction = reduction;
    events.push_back(std::move(event));
  }

  std::vector<Event> events;
};

template <typename Callable>
void expectRuntimeError(Callable &&callable, const char *message) {
  try {
    callable();
  } catch (const std::runtime_error &) {
    return;
  }
  throw std::runtime_error(message);
}

void writeTestRax(const std::string &path) {
  flatbuffers::FlatBufferBuilder builder;
  const std::string uri = "file:" RAX_EXECUTOR_TEST_LIBRARY;

  const std::vector<int64_t> f32Shape = {2, 2};
  const std::vector<int64_t> f32Strides = {2, 1};
  const std::vector<int64_t> f32Vector2Shape = {2};
  const std::vector<int64_t> f32Vector4Shape = {4};
  const std::vector<int64_t> vectorStrides = {1};
  const std::vector<int64_t> u16Shape = {3};
  const std::vector<int64_t> u16Strides = {1};
  const std::vector<flatbuffers::Offset<KV>> f32Attrs = {
      CreateKV(builder, builder.CreateString("role"),
               builder.CreateString("activation"))};
  auto f32Type = CreateTensorType(
      builder, DType_F32, CreateShape(builder, builder.CreateVector(f32Shape)),
      Layout_RowMajor, builder.CreateVector(f32Strides));
  auto f32Vector2Type = CreateTensorType(
      builder, DType_F32,
      CreateShape(builder, builder.CreateVector(f32Vector2Shape)),
      Layout_RowMajor, builder.CreateVector(vectorStrides));
  auto f32Vector4Type = CreateTensorType(
      builder, DType_F32,
      CreateShape(builder, builder.CreateVector(f32Vector4Shape)),
      Layout_RowMajor, builder.CreateVector(vectorStrides));
  auto u16Type = CreateTensorType(
      builder, DType_U16, CreateShape(builder, builder.CreateVector(u16Shape)),
      Layout_RowMajor, builder.CreateVector(u16Strides));
  const std::vector<flatbuffers::Offset<BufferBinding>> buffers = {
      CreateBufferBinding(builder, 1, builder.CreateString("f32_buffer"),
                          f32Type, MemorySpace_Host,
                          builder.CreateVector(f32Attrs)),
      CreateBufferBinding(builder, 2, builder.CreateString("u16_buffer"),
                          u16Type, MemorySpace_Host, 0),
      CreateBufferBinding(builder, 3,
                          builder.CreateString("all_gather_v_input"),
                          f32Vector2Type, MemorySpace_Host, 0),
      CreateBufferBinding(builder, 4,
                          builder.CreateString("all_gather_v_output"),
                          f32Vector4Type, MemorySpace_Host, 0),
      CreateBufferBinding(builder, 5,
                          builder.CreateString("reduce_scatter_input"),
                          f32Vector4Type, MemorySpace_Host, 0),
      CreateBufferBinding(builder, 6,
                          builder.CreateString("reduce_scatter_output"),
                          f32Vector2Type, MemorySpace_Host, 0)};

  const std::vector<flatbuffers::Offset<CollectiveOperand>> broadcastOperands =
      {CreateCollectiveOperand(builder, 1, 1),
       CreateCollectiveOperand(builder, 2, 2)};
  const std::vector<flatbuffers::Offset<CollectiveOperand>> reduceOperands = {
      CreateCollectiveOperand(builder, 1, 1)};
  auto broadcast = CreateCollectiveOp(builder, CollectiveKind_Broadcast,
                                      builder.CreateVector(broadcastOperands),
                                      ReductionKind_Invalid, 1, 0);
  auto allReduce = CreateCollectiveOp(builder, CollectiveKind_AllReduce,
                                      builder.CreateVector(reduceOperands),
                                      ReductionKind_Sum, -1, 0);
  const std::vector<flatbuffers::Offset<Op>> collectiveOps = {
      CreateOp(builder, OpKind_Collective, 0, 0, 0, 0, 0, 0, broadcast),
      CreateOp(builder, OpKind_Barrier, 0, 0, 0, CreateBarrierOp(builder, 0), 0,
               0, 0),
      CreateOp(builder, OpKind_Collective, 0, 0, 0, 0, 0, 0, allReduce)};
  const std::vector<uint32_t> functionBuffers = {1, 2};
  auto collectivesFunction =
      CreateFunction(builder, builder.CreateString("collectives"),
                     builder.CreateVector(functionBuffers),
                     builder.CreateVector(functionBuffers), 0,
                     builder.CreateVector(collectiveOps), 0);

  const std::vector<int64_t> recvCounts = {2, 2};
  const std::vector<int64_t> displacements = {0, 2};
  const auto recvCountsOffset = builder.CreateVector(recvCounts);
  const auto displacementsOffset = builder.CreateVector(displacements);
  const std::vector<flatbuffers::Offset<CollectiveOperand>> allGatherVOperands =
      {CreateCollectiveOperand(builder, 3, 4, recvCountsOffset,
                               displacementsOffset)};
  auto allGatherV = CreateCollectiveOp(builder, CollectiveKind_AllGatherV,
                                       builder.CreateVector(allGatherVOperands),
                                       ReductionKind_Invalid, -1, 0);
  const std::vector<flatbuffers::Offset<Op>> allGatherVOps = {
      CreateOp(builder, OpKind_Collective, 0, 0, 0, 0, 0, 0, allGatherV)};
  const std::vector<uint32_t> allGatherVInputs = {3};
  const std::vector<uint32_t> allGatherVOutputs = {4};
  auto allGatherVFunction =
      CreateFunction(builder, builder.CreateString("all_gather_v"),
                     builder.CreateVector(allGatherVInputs),
                     builder.CreateVector(allGatherVOutputs), 0,
                     builder.CreateVector(allGatherVOps), 0);

  const std::vector<flatbuffers::Offset<CollectiveOperand>>
      reduceScatterOperands = {
          CreateCollectiveOperand(builder, 5, 6, recvCountsOffset, 0)};
  auto reduceScatter = CreateCollectiveOp(
      builder, CollectiveKind_ReduceScatter,
      builder.CreateVector(reduceScatterOperands), ReductionKind_Sum, -1, 0);
  const std::vector<flatbuffers::Offset<Op>> reduceScatterOps = {
      CreateOp(builder, OpKind_Collective, 0, 0, 0, 0, 0, 0, reduceScatter)};
  const std::vector<uint32_t> reduceScatterInputs = {5};
  const std::vector<uint32_t> reduceScatterOutputs = {6};
  auto reduceScatterFunction =
      CreateFunction(builder, builder.CreateString("reduce_scatter"),
                     builder.CreateVector(reduceScatterInputs),
                     builder.CreateVector(reduceScatterOutputs), 0,
                     builder.CreateVector(reduceScatterOps), 0);

  auto producerCodeObject = CreateCodeObject(
      builder, 10, builder.CreateString("collective_producer"),
      CodeObjectKind_HostSharedLib, 0, builder.CreateString(uri),
      builder.CreateString("collective_producer"), builder.CreateString("cpu"),
      0);
  auto consumerCodeObject = CreateCodeObject(
      builder, 11, builder.CreateString("collective_consumer"),
      CodeObjectKind_HostSharedLib, 0, builder.CreateString(uri),
      builder.CreateString("collective_consumer"), builder.CreateString("cpu"),
      0);
  const std::vector<flatbuffers::Offset<Arg>> dispatchArguments = {
      CreateArg(builder, 1, 0, 0, 0)};
  auto producerDispatch = CreateDispatchOp(
      builder, 10, builder.CreateVector(dispatchArguments), 0, 0);
  auto sequenceAllReduce = CreateCollectiveOp(
      builder, CollectiveKind_AllReduce, builder.CreateVector(reduceOperands),
      ReductionKind_Sum, -1, 0);
  auto consumerDispatch = CreateDispatchOp(
      builder, 11, builder.CreateVector(dispatchArguments), 0, 0);
  const std::vector<flatbuffers::Offset<Op>> sequenceOps = {
      CreateOp(builder, OpKind_Dispatch, producerDispatch, 0, 0, 0, 0, 0),
      CreateOp(builder, OpKind_Collective, 0, 0, 0, 0, 0, 0, sequenceAllReduce),
      CreateOp(builder, OpKind_Dispatch, consumerDispatch, 0, 0, 0, 0, 0)};
  auto sequenceFunction = CreateFunction(
      builder, builder.CreateString("dispatch_collective_dispatch"),
      builder.CreateVector(functionBuffers),
      builder.CreateVector(functionBuffers), 0,
      builder.CreateVector(sequenceOps), 0);

  const std::vector<flatbuffers::Offset<CodeObject>> codeObjects = {
      producerCodeObject, consumerCodeObject};
  const std::vector<flatbuffers::Offset<Function>> functions = {
      collectivesFunction, allGatherVFunction, reduceScatterFunction,
      sequenceFunction};
  auto module = CreateModule(
      builder, builder.CreateString("RAX"), CreateVersion(builder, 0, 1, 0),
      Endianness_Little, 0, 0, builder.CreateVector(buffers), 0,
      builder.CreateVector(codeObjects), builder.CreateVector(functions));
  builder.Finish(module, "RAX0");

  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(builder.GetBufferPointer()),
               builder.GetSize());
  if (!output)
    throw std::runtime_error("failed to write test RAX file");
}

void checkManifest(const buddy::runtime::ModelManifest &manifest) {
  if (manifest.codeObjects.size() != 2)
    throw std::runtime_error("dispatch code objects were not parsed");
  if (manifest.buffers.size() != 6)
    throw std::runtime_error("buffer metadata was not parsed");
  const auto &buffer = manifest.buffers[0];
  if (buffer.id != 1 || buffer.name != "f32_buffer" ||
      buffer.dtype != DType_F32 ||
      buffer.shape != std::vector<int64_t>({2, 2}) ||
      buffer.strides != std::vector<int64_t>({2, 1}) ||
      buffer.layout != Layout_RowMajor ||
      buffer.memorySpace != MemorySpace_Host ||
      buffer.attrs.at("role") != "activation")
    throw std::runtime_error("buffer metadata was not preserved");

  if (manifest.functions.size() != 4 || manifest.functions[0].ops.size() != 3)
    throw std::runtime_error("collective operation sequence was not parsed");
  const auto &broadcast = manifest.functions[0].ops[0];
  if (broadcast.kind != OpKind_Collective ||
      broadcast.collectiveKind != CollectiveKind_Broadcast ||
      broadcast.reductionKind != ReductionKind_Invalid || broadcast.root != 1 ||
      broadcast.collectiveOperands.size() != 2 ||
      broadcast.collectiveOperands[0].inputBufferId != 1 ||
      broadcast.collectiveOperands[0].outputBufferId != 1 ||
      broadcast.collectiveOperands[1].inputBufferId != 2 ||
      broadcast.collectiveOperands[1].outputBufferId != 2)
    throw std::runtime_error("Broadcast payload was not preserved");
  const auto &allReduce = manifest.functions[0].ops[2];
  if (allReduce.collectiveKind != CollectiveKind_AllReduce ||
      allReduce.reductionKind != ReductionKind_Sum || allReduce.root != -1 ||
      allReduce.collectiveOperands.size() != 1)
    throw std::runtime_error("AllReduce payload was not preserved");

  const auto &allGatherV = manifest.functions[1].ops[0];
  if (allGatherV.collectiveKind != CollectiveKind_AllGatherV ||
      allGatherV.collectiveOperands.size() != 1 ||
      allGatherV.collectiveOperands[0].inputBufferId != 3 ||
      allGatherV.collectiveOperands[0].outputBufferId != 4 ||
      allGatherV.collectiveOperands[0].recvCounts !=
          std::vector<int64_t>({2, 2}) ||
      allGatherV.collectiveOperands[0].displacements !=
          std::vector<int64_t>({0, 2}))
    throw std::runtime_error("AllGatherV payload was not preserved");

  const auto &reduceScatter = manifest.functions[2].ops[0];
  if (reduceScatter.collectiveKind != CollectiveKind_ReduceScatter ||
      reduceScatter.reductionKind != ReductionKind_Sum ||
      reduceScatter.collectiveOperands.size() != 1 ||
      reduceScatter.collectiveOperands[0].inputBufferId != 5 ||
      reduceScatter.collectiveOperands[0].outputBufferId != 6 ||
      reduceScatter.collectiveOperands[0].recvCounts !=
          std::vector<int64_t>({2, 2}) ||
      !reduceScatter.collectiveOperands[0].displacements.empty())
    throw std::runtime_error("ReduceScatter payload was not preserved");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: buddy-rax-collective-executor-test <output.rax>\n";
    return 1;
  }

  try {
    writeTestRax(argv[1]);
    auto manifest = buddy::runtime::ModelManifest::loadFromRax(argv[1]);
    checkManifest(manifest);

    std::array<float, 16> f32Data = {};
    std::array<uint16_t, 24> u16Data = {};
    std::array<float, 16> allGatherVInputData = {};
    std::array<float, 16> allGatherVOutputData = {};
    std::array<float, 16> reduceScatterInputData = {};
    std::array<float, 16> reduceScatterOutputData = {};

    buddy::runtime::RaxExecutor noCommunicator(manifest);
    expectRuntimeError([&] { noCommunicator.execute("collectives"); },
                       "Collective without communicator did not fail");

    FakeCommunicator communicator;
    buddy::runtime::RaxExecutor executor(manifest, communicator);
    executor.bindBuffer(1, f32Data.data());
    executor.bindBuffer(2, u16Data.data());
    executor.bindBuffer(3, allGatherVInputData.data());
    executor.bindBuffer(4, allGatherVOutputData.data());
    executor.bindBuffer(5, reduceScatterInputData.data());
    executor.bindBuffer(6, reduceScatterOutputData.data());
    executor.execute("collectives");
    executor.execute("all_gather_v");
    executor.execute("reduce_scatter");

    if (communicator.events.size() != 5)
      throw std::runtime_error("unexpected communicator call count");
    const Event &first = communicator.events[0];
    const Event &second = communicator.events[1];
    const Event &third = communicator.events[2];
    const Event &fourth = communicator.events[3];
    const Event &fifth = communicator.events[4];
    if (first.kind != Event::Kind::Broadcast ||
        first.recvBuffer != f32Data.data() || first.amount != 16 ||
        first.root != 1 || second.kind != Event::Kind::Broadcast ||
        second.recvBuffer != u16Data.data() || second.amount != 6 ||
        second.root != 1)
      throw std::runtime_error(
          "Broadcast operands or raw payload pointers were incorrect");
    if (third.kind != Event::Kind::AllReduce ||
        third.sendBuffer != f32Data.data() ||
        third.recvBuffer != f32Data.data() || third.amount != 4 ||
        third.dataType != buddy::runtime::DataType::F32 ||
        third.reduction != buddy::runtime::ReductionOp::Sum)
      throw std::runtime_error("F32/Sum AllReduce call was incorrect");
    if (fourth.kind != Event::Kind::AllGatherV ||
        fourth.sendBuffer != allGatherVInputData.data() ||
        fourth.recvBuffer != allGatherVOutputData.data() ||
        fourth.sendBuffer == fourth.recvBuffer || fourth.amount != 2 ||
        fourth.recvCounts != std::vector<int64_t>({2, 2}) ||
        fourth.displacements != std::vector<int64_t>({0, 2}) ||
        fourth.dataType != buddy::runtime::DataType::F32)
      throw std::runtime_error("AllGatherV call was incorrect");
    if (fifth.kind != Event::Kind::ReduceScatter ||
        fifth.sendBuffer != reduceScatterInputData.data() ||
        fifth.recvBuffer != reduceScatterOutputData.data() ||
        fifth.sendBuffer == fifth.recvBuffer ||
        fifth.recvCounts != std::vector<int64_t>({2, 2}) ||
        fifth.dataType != buddy::runtime::DataType::F32 ||
        fifth.reduction != buddy::runtime::ReductionOp::Sum)
      throw std::runtime_error("ReduceScatter call was incorrect");

    std::array<float, 16> sequenceData = {};
    FakeCommunicator sequenceCommunicator;
    buddy::runtime::RaxExecutor sequenceExecutor(manifest,
                                                 sequenceCommunicator);
    sequenceExecutor.bindBuffer(1, sequenceData.data());
    sequenceExecutor.execute("dispatch_collective_dispatch");
    if (sequenceCommunicator.events.size() != 1 || sequenceData[0] != 10.0f ||
        sequenceData[1] != 4.0f || sequenceData[2] != 6.0f ||
        sequenceData[3] != 8.0f)
      throw std::runtime_error(
          "raw buffer dispatch/collective/dispatch sequence was incorrect");

    auto badAllGatherVCounts = manifest;
    badAllGatherVCounts.functions[1].ops[0].collectiveOperands[0].recvCounts = {
        2};
    FakeCommunicator badCountsCommunicator;
    buddy::runtime::RaxExecutor badCountsExecutor(badAllGatherVCounts,
                                                  badCountsCommunicator);
    badCountsExecutor.bindBuffer(3, allGatherVInputData.data());
    badCountsExecutor.bindBuffer(4, allGatherVOutputData.data());
    expectRuntimeError([&] { badCountsExecutor.execute("all_gather_v"); },
                       "AllGatherV accepted mismatched receive counts");

    auto smallAllGatherVOutput = manifest;
    smallAllGatherVOutput.buffers[3].shape = {3};
    std::array<float, 16> smallAllGatherVOutputData = {};
    FakeCommunicator smallOutputCommunicator;
    buddy::runtime::RaxExecutor smallOutputExecutor(smallAllGatherVOutput,
                                                    smallOutputCommunicator);
    smallOutputExecutor.bindBuffer(3, allGatherVInputData.data());
    smallOutputExecutor.bindBuffer(4, smallAllGatherVOutputData.data());
    expectRuntimeError([&] { smallOutputExecutor.execute("all_gather_v"); },
                       "AllGatherV accepted an undersized output buffer");

    auto shortReduceScatterInput = manifest;
    shortReduceScatterInput.buffers[4].shape = {3};
    std::array<float, 16> shortReduceScatterInputData = {};
    FakeCommunicator shortInputCommunicator;
    buddy::runtime::RaxExecutor shortInputExecutor(shortReduceScatterInput,
                                                   shortInputCommunicator);
    shortInputExecutor.bindBuffer(5, shortReduceScatterInputData.data());
    shortInputExecutor.bindBuffer(6, reduceScatterOutputData.data());
    expectRuntimeError([&] { shortInputExecutor.execute("reduce_scatter"); },
                       "ReduceScatter accepted a mismatched input count");

    auto shortReduceScatterOutput = manifest;
    shortReduceScatterOutput.buffers[5].shape = {1};
    std::array<float, 16> shortReduceScatterOutputData = {};
    FakeCommunicator shortOutputCommunicator;
    buddy::runtime::RaxExecutor shortOutputExecutor(shortReduceScatterOutput,
                                                    shortOutputCommunicator);
    shortOutputExecutor.bindBuffer(5, reduceScatterInputData.data());
    shortOutputExecutor.bindBuffer(6, shortReduceScatterOutputData.data());
    expectRuntimeError([&] { shortOutputExecutor.execute("reduce_scatter"); },
                       "ReduceScatter accepted a mismatched output count");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
