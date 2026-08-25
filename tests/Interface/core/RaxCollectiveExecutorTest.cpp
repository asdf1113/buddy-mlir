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

struct MemRef2DF32 {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

struct MemRef1DU16 {
  uint16_t *allocated;
  uint16_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
};

struct Event {
  enum class Kind { Broadcast, AllReduce } kind;
  const void *sendBuffer = nullptr;
  void *recvBuffer = nullptr;
  size_t amount = 0;
  int root = -1;
  buddy::runtime::DataType dataType = buddy::runtime::DataType::F32;
  buddy::runtime::ReductionOp reduction = buddy::runtime::ReductionOp::Sum;
};

class FakeCommunicator final : public buddy::runtime::Communicator {
public:
  int rank() const override { return 0; }
  int size() const override { return 2; }

  void broadcast(void *buffer, size_t bytes, int root) override {
    events.push_back({Event::Kind::Broadcast, buffer, buffer, bytes, root,
                      buddy::runtime::DataType::F32,
                      buddy::runtime::ReductionOp::Sum});
  }

  void allReduce(const void *sendBuffer, void *recvBuffer, size_t count,
                 buddy::runtime::DataType dataType,
                 buddy::runtime::ReductionOp reduction) override {
    events.push_back({Event::Kind::AllReduce, sendBuffer, recvBuffer, count, -1,
                      dataType, reduction});
  }

  std::vector<Event> events;
};

void writeTestRax(const std::string &path) {
  flatbuffers::FlatBufferBuilder builder;

  const std::vector<int64_t> f32Shape = {2, 2};
  const std::vector<int64_t> f32Strides = {2, 1};
  const std::vector<int64_t> u16Shape = {3};
  const std::vector<int64_t> u16Strides = {1};
  const std::vector<flatbuffers::Offset<KV>> f32Attrs = {
      CreateKV(builder, builder.CreateString("role"),
               builder.CreateString("activation"))};
  auto f32Type = CreateTensorType(
      builder, DType_F32, CreateShape(builder, builder.CreateVector(f32Shape)),
      Layout_RowMajor, builder.CreateVector(f32Strides));
  auto u16Type = CreateTensorType(
      builder, DType_U16, CreateShape(builder, builder.CreateVector(u16Shape)),
      Layout_RowMajor, builder.CreateVector(u16Strides));
  const std::vector<flatbuffers::Offset<BufferBinding>> buffers = {
      CreateBufferBinding(builder, 1, builder.CreateString("f32_buffer"),
                          f32Type, MemorySpace_Host,
                          builder.CreateVector(f32Attrs)),
      CreateBufferBinding(builder, 2, builder.CreateString("u16_buffer"),
                          u16Type, MemorySpace_Host, 0)};

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
  const std::vector<flatbuffers::Offset<Op>> ops = {
      CreateOp(builder, OpKind_Collective, 0, 0, 0, 0, 0, 0, broadcast),
      CreateOp(builder, OpKind_Barrier, 0, 0, 0, CreateBarrierOp(builder, 0), 0,
               0, 0),
      CreateOp(builder, OpKind_Collective, 0, 0, 0, 0, 0, 0, allReduce)};
  const std::vector<uint32_t> functionBuffers = {1, 2};
  auto function = CreateFunction(builder, builder.CreateString("collectives"),
                                 builder.CreateVector(functionBuffers),
                                 builder.CreateVector(functionBuffers), 0,
                                 builder.CreateVector(ops), 0);

  const std::vector<flatbuffers::Offset<CodeObject>> codeObjects = {
      CreateCodeObject(builder, 99, builder.CreateString("unused"),
                       CodeObjectKind_Unknown, 0, 0, 0, 0, 0)};
  const std::vector<flatbuffers::Offset<Function>> functions = {function};
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
  if (manifest.buffers.size() != 2)
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

  if (manifest.functions.size() != 1 || manifest.functions[0].ops.size() != 3)
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

    std::array<float, 5> f32Data = {};
    std::array<uint16_t, 4> u16Data = {};
    MemRef2DF32 f32Descriptor = {
        f32Data.data(), f32Data.data(), 1, {2, 2}, {2, 1}};
    MemRef1DU16 u16Descriptor = {u16Data.data(), u16Data.data(), 1, {3}, {1}};

    buddy::runtime::RaxExecutor noCommunicator(manifest);
    bool missingCommunicatorFailed = false;
    try {
      noCommunicator.execute("collectives");
    } catch (const std::runtime_error &) {
      missingCommunicatorFailed = true;
    }
    if (!missingCommunicatorFailed)
      throw std::runtime_error("Collective without communicator did not fail");

    FakeCommunicator communicator;
    buddy::runtime::RaxExecutor executor(manifest, communicator);
    executor.bindBuffer(1, &f32Descriptor);
    executor.bindBuffer(2, &u16Descriptor);
    executor.execute("collectives");

    if (communicator.events.size() != 3)
      throw std::runtime_error("unexpected communicator call count");
    const Event &first = communicator.events[0];
    const Event &second = communicator.events[1];
    const Event &third = communicator.events[2];
    if (first.kind != Event::Kind::Broadcast ||
        first.recvBuffer != f32Data.data() + 1 || first.amount != 16 ||
        first.root != 1 || second.kind != Event::Kind::Broadcast ||
        second.recvBuffer != u16Data.data() + 1 || second.amount != 6 ||
        second.root != 1)
      throw std::runtime_error(
          "Broadcast operands or ranked memref data were incorrect");
    if (third.kind != Event::Kind::AllReduce ||
        third.sendBuffer != f32Data.data() + 1 ||
        third.recvBuffer != f32Data.data() + 1 || third.amount != 4 ||
        third.dataType != buddy::runtime::DataType::F32 ||
        third.reduction != buddy::runtime::ReductionOp::Sum)
      throw std::runtime_error("F32/Sum AllReduce call was incorrect");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
