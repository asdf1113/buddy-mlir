//===- dis-main.cpp -----------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <string>

using namespace buddy;
double total_time = 0;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 1024;
constexpr size_t SubMaxTokenLength = 512;

constexpr size_t NUM_LAYERS = 56;
constexpr size_t HiddenSize = 128;
constexpr size_t HiddenSize0 = 1536;
constexpr size_t HeadNum = 2;

struct MemRefContainer0 {
  MemRef<float, 3> data;
  MemRef<int8_t, 4> mask;
  MemRef<float, 3> cos;
  MemRef<float, 3> sin;

  MemRefContainer0(MemRef<float, 3> m1, MemRef<int8_t, 4> m2,
                   MemRef<float, 3> m3, MemRef<float, 3> m4)
      : data(m1), mask(m2), cos(m3), sin(m4) {}
};

struct MemRefContainer2 {
  MemRef<float, 4> kcache;
  MemRef<float, 4> vcache;
  MemRef<float, 2> data;

  MemRefContainer2(MemRef<float, 4> m1, MemRef<float, 4> m2,
                   MemRef<float, 2> m3)
      : kcache(m1), vcache(m2), data(m3) {}
};

/// Declare DeepSeekR1 forward function.
extern "C" {
void _mlir_ciface_forward_prefill0(MemRefContainer0 *, MemRef<float, 1> *,
                                   Text<size_t, 2> *);
void _mlir_ciface_forward_prefill1(MemRef<float, 3> *, MemRef<float, 1> *,
                                   MemRef<float, 3> *);
void _mlir_ciface_forward_prefill2(MemRefContainer2 *, MemRef<float, 1> *,
                                   MemRef<int8_t, 4> *, MemRef<float, 3> *,
                                   MemRef<float, 3> *, MemRef<float, 3> *);
void _mlir_ciface_forward_prefill3(MemRef<float, 3> *, MemRef<float, 3> *,
                                   MemRef<float, 2> *);
void _mlir_ciface_forward_prefill5(MemRef<float, 2> *, MemRef<float, 1> *,
                                   MemRef<float, 3> *);
void _mlir_ciface_forward_prefill169(MemRef<float, 3> *, MemRef<float, 1> *,
                                     MemRef<float, 3> *);

void _mlir_ciface_forward_decode0(MemRefContainer0 *, MemRef<float, 1> *,
                                  MemRef<long long, 2> *,
                                  MemRef<long long, 1> *);
void _mlir_ciface_forward_decode1(MemRef<float, 3> *, MemRef<float, 1> *,
                                  MemRef<float, 3> *);
void _mlir_ciface_forward_decode2(MemRefContainer2 *, MemRef<float, 1> *,
                                  MemRef<long long, 1> *, MemRef<float, 4> *,
                                  MemRef<float, 4> *, MemRef<int8_t, 4> *,
                                  MemRef<float, 3> *, MemRef<float, 3> *,
                                  MemRef<float, 3> *);
void _mlir_ciface_forward_decode3(MemRef<float, 3> *, MemRef<float, 3> *,
                                  MemRef<float, 2> *);
void _mlir_ciface_forward_decode5(MemRef<float, 2> *, MemRef<float, 1> *,
                                  MemRef<float, 3> *);
void _mlir_ciface_forward_decode169(MemRef<float, 3> *, MemRef<float, 1> *,
                                    MemRef<float, 3> *);
}

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

/// Capture input message.
void getUserInput(std::string &inputStr) {
  std::cout << "\nPlease send a message:" << std::endl;
  std::cout << ">>> ";
  getline(std::cin, inputStr);
  std::cout << std::endl;
}

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

/// Print information for each iteration.
void printIterInfo(size_t iterIdx, std::string str, double time) {
  total_time += time;
  std::cout << "\033[32;1m[Iteration " << iterIdx << "] \033[0m";
  std::cout << "Token: " << str << " | "
            << "Time: " << time << "s" << std::endl;
}

/// Tokenize input data in the container.
void tokenizeInput(const std::string &vocabFile,
                   Text<size_t, 2> &inputContainer) {
  printLogLabel();
  std::cout << "Vocab file: " << std::filesystem::canonical(vocabFile)
            << std::endl;
  const auto buddyTokenizeStart = std::chrono::high_resolution_clock::now();
  inputContainer.tokenizeDeepSeekR1(vocabFile, MaxTokenLength);
  const auto buddyTokenizeEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> buddyTokenizeTime =
      buddyTokenizeEnd - buddyTokenizeStart;
  printLogLabel();
  std::cout << "Tokenize time: " << buddyTokenizeTime.count() << "ms"
            << std::endl;
}

/// Load parameters into data container.
void loadParameters(const std::string &paramFilePath,
                    MemRef<float, 1> &params) {
  const auto loadStart = std::chrono::high_resolution_clock::now();
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    std::cout << paramFilePath << std::endl;
    throw std::runtime_error("[Error] Failed to open params file!");
  }
  printLogLabel();
  std::cout << "Loading params..." << std::endl;
  printLogLabel();
  std::cout << "Params file: " << std::filesystem::canonical(paramFilePath)
            << std::endl;
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * (params.getSize()));
  if (paramFile.fail()) {
    throw std::runtime_error("Error occurred while reading params file!");
  }
  paramFile.close();
  const auto loadEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> loadTime =
      loadEnd - loadStart;
  printLogLabel();
  std::cout << "Params load time: " << (double)(loadTime.count()) / 1000
            << "s\n"
            << std::endl;
}

/// Find the index of the max value.
int findMaxIndex(const float *start, const float *end) {
  return std::distance(start, std::max_element(start, end));
}

// -----------------------------------------------------------------------------
// DeepSeekR1 Inference Main Entry
// -----------------------------------------------------------------------------

int main(int argc, char *argv[]) {

  /// Define directories of vacabulary and parameter file.
  std::string deepSeekR1Dir = DSR1TP_EXAMPLE_PATH;
  std::string deepSeekR1BuildDir = DSR1TP_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = deepSeekR1Dir + "/vocab.txt";

  // Common variables needed by all ranks
  int subSize = SubMaxTokenLength * HiddenSize0;
  int offset0 = subSize;
  int offset1 = subSize * 2;

  int rank, size;
  int generateLen = MaxTokenLength;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    /// Print the title of this example.
    const std::string title = "DeepSeekR1  Inference Powered by Buddy Compiler";
    std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

    //  变量 Rank 0 specific variables
    Text<size_t, 2> outputContainer;
    outputContainer.loadVocab(vocabDir);

    MemRef<float, 3> myMemRef1({1, MaxTokenLength, HiddenSize0});
    MemRef<int8_t, 4> myMemRef2({1, 1, MaxTokenLength, MaxTokenLength});
    MemRef<float, 3> myMemRef3({1, MaxTokenLength, HiddenSize});
    MemRef<float, 3> myMemRef4({1, MaxTokenLength, HiddenSize});
    MemRefContainer0 resultContainer(myMemRef1, myMemRef2, myMemRef3,
                                     myMemRef4);
    MemRefContainer0 *resultContainerPtr = &resultContainer;

    MemRef<float, 3> tmp3DMemRef(
        {1, MaxTokenLength,
         HiddenSize0}); //  MemRef<float, 3> tmp3DContainer({1, MaxTokenLength,
                        //  HiddenSize0});
    MemRef<float, 3> resultPrefill({1, MaxTokenLength, MaxVocabSize});
    MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);

    MemRef<long long, 1> cachePosition({1}, 0LL);

    MemRef<float, 3> myMemRef_decode1({1, 1, HiddenSize0});
    MemRef<int8_t, 4> myMemRef_decode2({1, 1, 1, MaxTokenLength});
    MemRef<float, 3> myMemRef_decode3({1, 1, HiddenSize});
    MemRef<float, 3> myMemRef_decode4({1, 1, HiddenSize});
    MemRefContainer0 resultContainerDecode(myMemRef_decode1, myMemRef_decode2,
                                           myMemRef_decode3, myMemRef_decode4);
    MemRefContainer0 *resultContainerDecodePtr = &resultContainerDecode;

    MemRef<float, 3> resultDecode({1, 1, MaxVocabSize});

    // 加载参数
    constexpr size_t param_size0 = 233373760;
    const std::string paramsDir0 =
        deepSeekR1BuildDir + "/subgraph0_prefill0_arg0.data";
    constexpr size_t param_size1 = 233375232;
    const std::string paramsDir1 =
        deepSeekR1BuildDir + "/subgraph0_prefill169_arg0.data";
    MemRef<float, 1> paramsContainer0({param_size0});
    loadParameters(paramsDir0, paramsContainer0);
    MemRef<float, 1> paramsContainer1({param_size1});
    loadParameters(paramsDir1, paramsContainer1);

    // /// Get user message.
    std::string inputStr;
    getUserInput(inputStr);
    Text<size_t, 2> inputContainerPrefill(inputStr);
    tokenizeInput(vocabDir, inputContainerPrefill);
    std::cout << "Input token count: " << inputContainerPrefill.getTokenCnt()
              << std::endl;
    MPI_Request send_req[2], recv_req[2];
    MPI_Request send_req_mha[6];

    float *inputPtr = nullptr;
    float *outputPtr = tmp3DMemRef.getData();
    float *outputPtrDecode = myMemRef_decode1.getData();

    double prefillTokensPerSec = 0.0;
    const auto inferenceStart = std::chrono::high_resolution_clock::now();

    std::cout << "\n\033[33;1m[Inference Start]\033[0m" << std::endl;
    _mlir_ciface_forward_prefill0(resultContainerPtr, &paramsContainer0,
                                  &inputContainerPrefill);
    inputPtr = resultContainerPtr->data.getData();

    MPI_Isend(inputPtr, subSize, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &send_req[0]);
    MPI_Isend(inputPtr + offset0, subSize, MPI_FLOAT, 2, 0, MPI_COMM_WORLD,
              &send_req[1]);
    MPI_Waitall(2, send_req, MPI_STATUSES_IGNORE);

    for (int m = 1; m < 4; m += 2) {
      MPI_Isend(resultContainerPtr->mask.getData(),
                MaxTokenLength * MaxTokenLength, MPI_INT8_T, m, 1,
                MPI_COMM_WORLD, &send_req_mha[0]);
      MPI_Isend(resultContainerPtr->cos.getData(), MaxTokenLength * HiddenSize,
                MPI_FLOAT, m, 2, MPI_COMM_WORLD, &send_req_mha[1]);
      MPI_Isend(resultContainerPtr->sin.getData(), MaxTokenLength * HiddenSize,
                MPI_FLOAT, m, 3, MPI_COMM_WORLD, &send_req_mha[2]);
      MPI_Isend(resultContainerPtr->mask.getData(),
                MaxTokenLength * MaxTokenLength, MPI_INT8_T, m + 1, 1,
                MPI_COMM_WORLD, &send_req_mha[3]);
      MPI_Isend(resultContainerPtr->cos.getData(), MaxTokenLength * HiddenSize,
                MPI_FLOAT, m + 1, 2, MPI_COMM_WORLD, &send_req_mha[4]);
      MPI_Isend(resultContainerPtr->sin.getData(), MaxTokenLength * HiddenSize,
                MPI_FLOAT, m + 1, 3, MPI_COMM_WORLD, &send_req_mha[5]);
      MPI_Waitall(6, send_req_mha, MPI_STATUSES_IGNORE);
    }

    MPI_Irecv(outputPtr, subSize, MPI_FLOAT, 3, 0, MPI_COMM_WORLD,
              &recv_req[0]);
    MPI_Irecv(outputPtr + offset0, subSize, MPI_FLOAT, 4, 0, MPI_COMM_WORLD,
              &recv_req[1]);
    MPI_Waitall(2, recv_req, MPI_STATUSES_IGNORE);

    _mlir_ciface_forward_prefill169(&resultPrefill, &paramsContainer1,
                                    &tmp3DMemRef);
    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

    int tokenIndex = inputContainerPrefill.getTokenCnt() - 1;
    const float *startPtr = resultPrefill.getData() + tokenIndex * MaxVocabSize;
    const float *endPtr = startPtr + MaxVocabSize;
    int maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = inputContainerPrefill.getStr(maxIndex);
    // Print the generated token and inference time.
    printIterInfo(0, tok, inferenceTime.count() / 1000);

    const double prefillSeconds = inferenceTime.count() / 1000.0;
    if (prefillSeconds > 0.0) {
      prefillTokensPerSec =
          static_cast<double>(MaxTokenLength) / prefillSeconds;
    }

    inputContainerDecode.getData()[0] = (long long)maxIndex;
    outputContainer.appendTokenIdx(maxIndex);
    cachePosition.getData()[0] = inputContainerPrefill.getTokenCnt();
    generateLen = MaxTokenLength - inputContainerPrefill.getTokenCnt();
    double decodeTimeAccumMs = 0.0;
    size_t decodeTokens = 0;

    MPI_Request send_req_decode[8];

    for (int i = 1; i <= generateLen; i++) {

      const auto inferenceStart = std::chrono::high_resolution_clock::now();
      // std::cout << "\n\033[33;1m[Decoding Iteration " << i << "]\033[0m"
      //           << std::endl;
      // std::cout << "inputContainerDecode: " <<
      // inputContainerDecode.getData()[0]
      //           << std::endl;
      _mlir_ciface_forward_decode0(resultContainerDecodePtr, &paramsContainer0,
                                   &inputContainerDecode, &cachePosition);
      inputPtr = resultContainerDecodePtr->data.getData();
      // std::cout << "Data after decode0: " << inputPtr[0] << ", " <<
      // inputPtr[1]
      //           << ", " << inputPtr[2] << std::endl;
      MPI_Isend(inputPtr, HiddenSize0, MPI_FLOAT, 1, 0, MPI_COMM_WORLD,
                &send_req[0]);
      MPI_Isend(inputPtr, HiddenSize0, MPI_FLOAT, 2, 0, MPI_COMM_WORLD,
                &send_req[1]);
      MPI_Waitall(2, send_req, MPI_STATUSES_IGNORE);
      for (int m = 1; m < 4; m += 2) {
        MPI_Isend(resultContainerDecodePtr->mask.getData(), MaxTokenLength,
                  MPI_INT8_T, m, 1, MPI_COMM_WORLD, &send_req_decode[0]);
        MPI_Isend(resultContainerDecodePtr->cos.getData(), HiddenSize,
                  MPI_FLOAT, m, 2, MPI_COMM_WORLD, &send_req_decode[1]);
        MPI_Isend(resultContainerDecodePtr->sin.getData(), HiddenSize,
                  MPI_FLOAT, m, 3, MPI_COMM_WORLD, &send_req_decode[2]);
        MPI_Isend(cachePosition.getData(), 1, MPI_LONG_LONG, m, 4,
                  MPI_COMM_WORLD, &send_req_decode[3]);
        MPI_Isend(resultContainerDecodePtr->mask.getData(), MaxTokenLength,
                  MPI_INT8_T, m + 1, 1, MPI_COMM_WORLD, &send_req_decode[4]);
        MPI_Isend(resultContainerDecodePtr->cos.getData(), HiddenSize,
                  MPI_FLOAT, m + 1, 2, MPI_COMM_WORLD, &send_req_decode[5]);
        MPI_Isend(resultContainerDecodePtr->sin.getData(), HiddenSize,
                  MPI_FLOAT, m + 1, 3, MPI_COMM_WORLD, &send_req_decode[6]);
        MPI_Isend(cachePosition.getData(), 1, MPI_LONG_LONG, m + 1, 4,
                  MPI_COMM_WORLD, &send_req_decode[7]);

        MPI_Waitall(8, send_req_decode, MPI_STATUSES_IGNORE);
      }

      MPI_Irecv(outputPtrDecode, HiddenSize0, MPI_FLOAT, 3, 0, MPI_COMM_WORLD,
                &recv_req[0]);
      MPI_Waitall(1, recv_req, MPI_STATUSES_IGNORE);
      // std::cout << "Received data for decoding iteration " << i << std::endl;
      // std::cout << "Data: " << outputPtrDecode[0] << ", " <<
      // outputPtrDecode[1]
      //           << ", " << outputPtrDecode[2] << std::endl;

      // std::cout << "myMemRef_decode1:" << &myMemRef_decode1.getData()[0]
      //           << std::endl;
      _mlir_ciface_forward_decode169(&resultDecode, &paramsContainer1,
                                     &myMemRef_decode1);
      // std::cout << "Completed forward_decode169 for iteration " << i
      //           << std::endl;
      // std::cout << "Data after decode169: " << resultDecode.getData()[0] <<
      // ", "
      //           << resultDecode.getData()[1] << ", "
      //           << resultDecode.getData()[2] << std::endl;
      const auto inferenceEnd = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double, std::milli> inferenceTime =
          inferenceEnd - inferenceStart;
      decodeTimeAccumMs += inferenceTime.count();
      decodeTokens += 1;

      const float *startPtr = resultDecode.getData();
      // std::cout << "Start pointer value: " << startPtr[0] << ", " <<
      // startPtr[1]
      //           << ", " << startPtr[2] << std::endl;
      const float *endPtr = startPtr + MaxVocabSize;
      // std::cout << "End pointer value: " << endPtr[-3] << ", " << endPtr[-2]
      //           << ", " << endPtr[-1] << std::endl;
      int maxIndex = findMaxIndex(startPtr, endPtr);
      // std::cout << "Max index: " << maxIndex << std::endl;
      std::string tok = inputContainerPrefill.getStr(maxIndex);
      // Print the generated token and inference time.
      printIterInfo(i, tok, inferenceTime.count() / 1000);
      if (maxIndex == 151643) {
        break;
      }

      inputContainerDecode.getData()[0] = maxIndex;
      outputContainer.appendTokenIdx(maxIndex);
      cachePosition.getData()[0] += 1;
    }

    double decodeSeconds = decodeTimeAccumMs / 1000.0;
    const double decodeTokensPerSec =
        decodeSeconds > 0.0 ? static_cast<double>(decodeTokens) / decodeSeconds
                            : 0.0;

    /// Print the final result
    std::cout << "\n\033[33;1m[Total time]\033[0m " << total_time << std::endl;
    std::cout << "\033[33;1m[Prefilling]\033[0m " << prefillTokensPerSec
              << " tokens/s" << std::endl;
    std::cout << "\033[33;1m[Decoding]\033[0m " << decodeTokensPerSec
              << " tokens/s" << std::endl;
    std::cout << "\033[33;1m[Input]\033[0m " << inputStr << std::endl;
    std::cout << "\033[33;1m[Output]\033[0m "
              << outputContainer.revertDeepSeekR1() << std::endl;

  } else if (rank == 1 || rank == 3) {

    // === RMSNorm ===
    // Rank 1 specific variables
    // MemRef<long long, 1> cachePosition({1}, 0LL);
    MemRef<float, 3> subResultContainer({1, SubMaxTokenLength, HiddenSize0});
    MemRef<float, 3> sub3DContainer({1, SubMaxTokenLength, HiddenSize0});
    MemRef<int8_t, 4> mhaMemRef4D({1, 1, MaxTokenLength, MaxTokenLength});
    MemRef<float, 3> mhaMemRef3D1({1, MaxTokenLength, HiddenSize});
    MemRef<float, 3> mhaMemRef3D2({1, MaxTokenLength, HiddenSize});
    MemRef<float, 3> tmp3DMemRef({1, MaxTokenLength, HiddenSize0});
    MemRef<float, 2> tmp2DContainer({MaxTokenLength, HiddenSize0});
    MemRef<float, 2> sub2DContainer({SubMaxTokenLength, HiddenSize0});
    std::vector<MemRef<float, 4>> kv0;
    kv0.reserve(56);
    for (int i = 0; i < 56; ++i) {
      kv0.emplace_back(std::vector<size_t>{1, 1, MaxTokenLength, HiddenSize});
    }
    MemRefContainer2 kvContainer0(kv0[0], kv0[1], tmp2DContainer);
    MemRefContainer2 *kvContainerPtr0 = &kvContainer0;

    //     // Get pointers once to avoid repeated calls
    float *subResultPtr = subResultContainer.getData();
    float *rmsPtr = sub3DContainer.getData();
    int8_t *mhaMemRef4DPtr = mhaMemRef4D.getData();
    float *mhaMemRef3D1Ptr = mhaMemRef3D1.getData();
    float *mhaMemRef3D2Ptr = mhaMemRef3D2.getData();
    float *mhaPtr = tmp3DMemRef.getData();
    float *mhaOutputPtr = tmp2DContainer.getData();
    float *sub2DPtr = sub2DContainer.getData();

    constexpr size_t paramSizeRMS = 1536;
    constexpr size_t paramSizeMHA = 2753536;
    constexpr size_t paramSizeMLP = 20643840;
    int id = rank / 2;
    int init = id * 84; // 42=6*7 6*14=84,
    int endVal = (id + 1) * 84;
    int times = 14;
    int source = 0;
    int dest2 = rank + 1;
    int nextRank = (rank == 1 ? 3 : 0);
    MPI_Request send_req[2], recv_req[2];
    MPI_Request mha_recv_req[3];
    MPI_Request mha_recv_decode[4];
    // bool mha_params_received = false;
    std::vector<std::string> paramsDirsRMS, paramsDirsRMS0;
    std::vector<std::string> paramsDirsMHA, paramsDirsMLP;
    std::vector<MemRef<float, 1>> paramsContainersRMS, paramsContainersRMS0;
    std::vector<MemRef<float, 1>> paramsContainersMHA, paramsContainersMLP;

    // Get Parameter file paths
    for (int i = init + 1; i < endVal; i += 6) {
      paramsDirsRMS.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i) + "_arg0" + ".data");
      paramsDirsMHA.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i + 1) + "_arg0" + ".data");
      paramsDirsRMS0.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                  std::to_string(i + 3) + "_arg0" + ".data");
      paramsDirsMLP.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i + 4) + "_arg0" + ".data");
    }

    // Load parameters after Bcast to avoid blocking rank 2 at MPI_Barrier
    for (int i = 0; i < times; i++) {
      // First RMS
      MemRef<float, 1> paramsContainerRMS({paramSizeRMS});
      loadParameters(paramsDirsRMS[i], paramsContainerRMS);
      paramsContainersRMS.push_back(paramsContainerRMS);
      // MHA
      MemRef<float, 1> paramsContainerMHA({paramSizeMHA});
      loadParameters(paramsDirsMHA[i], paramsContainerMHA);
      paramsContainersMHA.push_back(paramsContainerMHA);
      // Second RMS
      MemRef<float, 1> paramsContainerRMS0({paramSizeRMS});
      loadParameters(paramsDirsRMS0[i], paramsContainerRMS0);
      paramsContainersRMS0.push_back(paramsContainerRMS0);
      // MLP
      MemRef<float, 1> paramsContainerMLP({paramSizeMLP});
      loadParameters(paramsDirsMLP[i], paramsContainerMLP);
      paramsContainersMLP.push_back(paramsContainerMLP);
    }

    MPI_Irecv(mhaMemRef4DPtr, MaxTokenLength * MaxTokenLength, MPI_INT8_T,
              source, 1, MPI_COMM_WORLD, &mha_recv_req[0]);
    MPI_Irecv(mhaMemRef3D1Ptr, MaxTokenLength * HiddenSize0, MPI_FLOAT, source,
              2, MPI_COMM_WORLD, &mha_recv_req[1]);
    MPI_Irecv(mhaMemRef3D2Ptr, MaxTokenLength * HiddenSize0, MPI_FLOAT, source,
              3, MPI_COMM_WORLD, &mha_recv_req[2]);

    for (int m = 0; m < times; m++) {
      if (m == 0) {
        if (rank == 1) {
          MPI_Irecv(subResultPtr, subSize, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
                    &recv_req[0]);
          MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        } else {
          MPI_Irecv(subResultPtr, subSize, MPI_FLOAT, rank - 2, 0,
                    MPI_COMM_WORLD, &recv_req[0]);
          MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        }
      }

      _mlir_ciface_forward_prefill1(&sub3DContainer, &paramsContainersRMS[m],
                                    &subResultContainer);
      rmsPtr = sub3DContainer.getData();
      mhaPtr = tmp3DMemRef.getData();

      // ----- AllGather -----
      MPI_Isend(rmsPtr, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                &send_req[0]);
      MPI_Irecv(mhaPtr + offset0, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                &recv_req[0]);

      for (int idx = 0; idx < subSize; idx++) {
        mhaPtr[idx] = rmsPtr[idx];
      }

      MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);

      MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

      if (m == 0) {
        MPI_Waitall(3, mha_recv_req, MPI_STATUSES_IGNORE);
      }

      _mlir_ciface_forward_prefill2(kvContainerPtr0, &paramsContainersMHA[m],
                                    &mhaMemRef4D, &mhaMemRef3D1, &mhaMemRef3D2,
                                    &tmp3DMemRef);
      kv0[2 * m] = kvContainerPtr0->kcache;
      kv0[2 * m + 1] = kvContainerPtr0->vcache;

      tmp2DContainer = kvContainerPtr0->data;

      // 同步
      // ----- Reduce-Scatter  -----

      mhaOutputPtr = tmp2DContainer.getData();

      MPI_Isend(mhaOutputPtr + offset0, subSize, MPI_FLOAT, dest2, 0,
                MPI_COMM_WORLD, &send_req[0]);
      MPI_Irecv(sub2DPtr, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                &recv_req[0]);
      MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
      MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

      for (int idx = 0; idx < subSize; idx++) {
        sub2DPtr[idx] += mhaOutputPtr[idx];
      }

      // add
      _mlir_ciface_forward_prefill3(&subResultContainer, &subResultContainer,
                                    &sub2DContainer);
      _mlir_ciface_forward_prefill1(&sub3DContainer, &paramsContainersRMS0[m],
                                    &subResultContainer);
      rmsPtr = sub3DContainer.getData();
      mhaPtr = tmp3DMemRef.getData();

      // allgather
      //  ----- AllGather (2路: 仅与rank+1通信) -----
      MPI_Isend(rmsPtr, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                &send_req[0]);
      MPI_Irecv(mhaPtr + offset0, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                &recv_req[0]);
      for (int idx = 0; idx < subSize; idx++) {
        mhaPtr[idx] = rmsPtr[idx];
      }
      MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
      MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

      // mlp
      _mlir_ciface_forward_prefill5(&tmp2DContainer, &paramsContainersMLP[m],
                                    &tmp3DMemRef);
      mhaOutputPtr = tmp2DContainer.getData();

      MPI_Isend(mhaOutputPtr + offset0, subSize, MPI_FLOAT, dest2, 0,
                MPI_COMM_WORLD, &send_req[0]);
      MPI_Irecv(sub2DPtr, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                &recv_req[0]);
      MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
      MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

      for (int idx = 0; idx < subSize; idx++) {
        sub2DPtr[idx] += mhaOutputPtr[idx];
      }

      _mlir_ciface_forward_prefill3(&subResultContainer, &subResultContainer,
                                    &sub2DContainer);

      if (m == (times - 1)) {

        subResultPtr = subResultContainer.getData();
        MPI_Send(subResultPtr, subSize, MPI_FLOAT, nextRank, 0, MPI_COMM_WORLD);
      }
    }

    // decode
    //
    MemRef<long long, 1> cachePosition({1}, 0LL);
    MemRef<float, 3> subResultContainerDecode({1, 1, HiddenSize0});
    MemRef<float, 3> sub3DContainerDecode({1, 1, HiddenSize0});
    MemRef<int8_t, 4> mhaMemRef4DDecode({1, 1, 1, MaxTokenLength});
    MemRef<float, 3> mhaMemRef3D1Decode({1, 1, HiddenSize});
    MemRef<float, 3> mhaMemRef3D2Decode({1, 1, HiddenSize});
    // MemRef<float, 3> tmp3DMemRefDecode({1, MaxTokenLength, HiddenSize0});
    MemRef<float, 2> tmp2DContainerDecode({1, HiddenSize0});
    MemRef<float, 2> sub2DContainerDecode({1, HiddenSize0});
    float *subResultPtrDecode = subResultContainerDecode.getData();
    float *rmsPtrDecode = sub3DContainerDecode.getData();
    int8_t *mhaMemRef4DPtrDecode = mhaMemRef4DDecode.getData();
    float *mhaMemRef3D1PtrDecode = mhaMemRef3D1Decode.getData();
    float *mhaMemRef3D2PtrDecode = mhaMemRef3D2Decode.getData();
    float *mhaPtrDecode = tmp3DMemRef.getData();
    long long int *cachePositionPtr = cachePosition.getData();
    float *mhaOutputPtrDecode = tmp2DContainerDecode.getData();
    float *sub2DPtrDecode = sub2DContainerDecode.getData();

    MemRef<float, 2> mhaDataDecode0({1, HiddenSize0});
    MemRefContainer2 kvDecodeContainer0(kv0[0], kv0[1], tmp2DContainerDecode);
    MemRefContainer2 *kvDecodeContainerPtr0 = &kvDecodeContainer0;

    for (int i = 1; i <= generateLen; i++) {
      MPI_Irecv(mhaMemRef4DPtrDecode, MaxTokenLength, MPI_INT8_T, source, 1,
                MPI_COMM_WORLD, &mha_recv_decode[0]);
      MPI_Irecv(mhaMemRef3D1PtrDecode, HiddenSize, MPI_FLOAT, source, 2,
                MPI_COMM_WORLD, &mha_recv_decode[1]);
      MPI_Irecv(mhaMemRef3D2PtrDecode, HiddenSize, MPI_FLOAT, source, 3,
                MPI_COMM_WORLD, &mha_recv_decode[2]);
      MPI_Irecv(cachePositionPtr, 1, MPI_LONG_LONG, source, 4, MPI_COMM_WORLD,
                &mha_recv_decode[3]);
      for (int m = 0; m < times; m++) {
        if (m == 0) {
          if (rank == 1) {
            MPI_Irecv(subResultPtrDecode, HiddenSize0, MPI_FLOAT, source, 0,
                      MPI_COMM_WORLD, &recv_req[0]);

            MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

          } else {
            MPI_Irecv(subResultPtrDecode, HiddenSize0, MPI_FLOAT, rank - 2, 0,
                      MPI_COMM_WORLD, &recv_req[0]);
            MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
            // std::cout << "Rank " << rank
            //           << " received data for decode iteration " << i
            //           << ", MHA iteration " << m << std::endl;
            // std::cout << "subResultPtrDecode received: "
            //           << subResultPtrDecode[0] << "," <<
            //           subResultPtrDecode[1]
            //           << "," << subResultPtrDecode[2] << std::endl;
          }
        }
        std::cout << "input for decode1: " << subResultPtrDecode[0] << ","
                  << subResultPtrDecode[1] << "," << subResultPtrDecode[2]
                  << std::endl;
        _mlir_ciface_forward_decode1(&sub3DContainerDecode,
                                     &paramsContainersRMS[m],
                                     &subResultContainerDecode);
        // std::cout << "after decode1 "
        //           << "sub3DContainerDecode: "
        //           << sub3DContainerDecode.getData()[0] << ","
        //           << sub3DContainerDecode.getData()[1] << ","
        //           << sub3DContainerDecode.getData()[2] << std::endl;

        if (m == 0) {
          MPI_Waitall(4, mha_recv_decode, MPI_STATUSES_IGNORE);
        }

        // std::cout << "[DEBUG]:  INPUT FOR DECODE2, cachePosition: "
        //           << cachePosition.getData()[0] << std::endl;
        // std::cout << "[DEBUG]:  INPUT FOR DECODE2, mhaMemRef4D: "
        //           << (int)mhaMemRef4DDecode.getData()[0] << ","
        //           << (int)mhaMemRef4DDecode.getData()[1] << ","
        //           << (int)mhaMemRef4DDecode.getData()[2] << ","
        //           << (int)mhaMemRef4DDecode.getData()[3] << std::endl;
        // std::cout << "[DEBUG]:  INPUT FOR DECODE2, COS: "
        //           << mhaMemRef3D1Decode.getData()[0] << ","
        //           << mhaMemRef3D1Decode.getData()[1] << ","
        //           << mhaMemRef3D1Decode.getData()[2] << std::endl;
        // std::cout << "[DEBUG]:  INPUT FOR DECODE2, SIN: "
        //           << mhaMemRef3D2Decode.getData()[0] << ","
        //           << mhaMemRef3D2Decode.getData()[1] << ","
        //           << mhaMemRef3D2Decode.getData()[2] << std::endl;
        // std::cout << "[DEBUG]:  INPUT FOR DECODE2, kv0: "
        //           << kv0[2 * m].getData()[0] << ", " << kv0[2 *
        //           m].getData()[1]
        //           << ", " << kv0[2 * m].getData()[2] << std::endl;
        _mlir_ciface_forward_decode2(
            kvDecodeContainerPtr0, &paramsContainersMHA[m], &cachePosition,
            &kv0[2 * m], &kv0[2 * m + 1], &mhaMemRef4DDecode,
            &mhaMemRef3D1Decode, &mhaMemRef3D2Decode, &sub3DContainerDecode);
        kv0[2 * m] = kvDecodeContainerPtr0->kcache;
        kv0[2 * m + 1] = kvDecodeContainerPtr0->vcache;
        tmp2DContainerDecode = kvDecodeContainerPtr0->data;
        // std::cout << "after decode2 "
        //           << "tmp2DContainerDecode: "
        //           << tmp2DContainerDecode.getData()[0] << ","
        //           << tmp2DContainerDecode.getData()[1] << ","
        //           << tmp2DContainerDecode.getData()[2] << std::endl;
        // ----- Reduce-Scatter  -----
        mhaOutputPtrDecode = tmp2DContainerDecode.getData();

        MPI_Isend(mhaOutputPtrDecode, HiddenSize0, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtrDecode, HiddenSize0, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        for (int idx = 0; idx < HiddenSize0; idx++) {
          sub2DPtrDecode[idx] += mhaOutputPtrDecode[idx];
        }
        // std::cout << "after reduce-scatter "
        //           << "sub2DContainerDecode: " << sub2DPtrDecode[0] << ","
        //           << sub2DPtrDecode[1] << "," << sub2DPtrDecode[2] <<
        //           std::endl;
        _mlir_ciface_forward_decode3(&subResultContainerDecode,
                                     &subResultContainerDecode,
                                     &sub2DContainerDecode);
        // std::cout << "after decode3 "
        //           << "subResultContainerDecode: " << subResultPtrDecode[0]
        //           << "," << subResultPtrDecode[1] << ","
        //           << subResultPtrDecode[2] << std::endl;
        _mlir_ciface_forward_decode1(&sub3DContainerDecode,
                                     &paramsContainersRMS0[m],
                                     &subResultContainerDecode);

        _mlir_ciface_forward_decode5(&tmp2DContainerDecode,
                                     &paramsContainersMLP[m],
                                     &sub3DContainerDecode);
        mhaOutputPtrDecode = tmp2DContainerDecode.getData();
        // std::cout << "after decode5 "
        //           << "tmp2DContainerDecode: "
        //           << tmp2DContainerDecode.getData()[0] << ","
        //           << tmp2DContainerDecode.getData()[1] << ","
        //           << tmp2DContainerDecode.getData()[2] << std::endl;
        MPI_Isend(mhaOutputPtrDecode, HiddenSize0, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtrDecode, HiddenSize0, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        for (int idx = 0; idx < HiddenSize0; idx++) {
          sub2DPtrDecode[idx] += mhaOutputPtrDecode[idx];
        }
        // std::cout << "after second reduce-scatter "
        //           << "sub2DContainerDecode: " << sub2DPtrDecode[0] << ","
        //           << sub2DPtrDecode[1] << "," << sub2DPtrDecode[2] <<
        //           std::endl;
        _mlir_ciface_forward_decode3(&subResultContainerDecode,
                                     &subResultContainerDecode,
                                     &sub2DContainerDecode);
        // std::cout << "after second decode3 "
        //           << "subResultContainerDecode: " << subResultPtrDecode[0]
        //           << "," << subResultPtrDecode[1] << ","
        //           << subResultPtrDecode[2] << std::endl;
        if (m == (times - 1)) {
          subResultPtrDecode = subResultContainerDecode.getData();
          MPI_Send(subResultPtrDecode, HiddenSize0, MPI_FLOAT, nextRank, 0,
                   MPI_COMM_WORLD);
          // std::cout << "Rank " << rank << std::endl;
          // std::cout << "subResultPtrDecode sent: " << subResultPtrDecode[0]
          //           << "," << subResultPtrDecode[1] << ","
          //           << subResultPtrDecode[2] << std::endl;
        }
        //////////////////////////////
      }
    }
  } else if (rank == 2 || rank == 4) {
    // === RMSNorm ===
    // Rank 1 specific variables
    // MemRef<long long, 1> cachePosition({1}, 0LL);
    MemRef<float, 3> subResultContainer({1, SubMaxTokenLength, HiddenSize0});
    MemRef<float, 3> sub3DContainer({1, SubMaxTokenLength, HiddenSize0});
    MemRef<int8_t, 4> mhaMemRef4D({1, 1, MaxTokenLength, MaxTokenLength});
    MemRef<float, 3> mhaMemRef3D1({1, MaxTokenLength, HiddenSize});
    MemRef<float, 3> mhaMemRef3D2({1, MaxTokenLength, HiddenSize});
    MemRef<float, 3> tmp3DMemRef({1, MaxTokenLength, HiddenSize0});
    MemRef<float, 2> tmp2DContainer({MaxTokenLength, HiddenSize0});
    MemRef<float, 2> sub2DContainer({SubMaxTokenLength, HiddenSize0});
    std::vector<MemRef<float, 4>> kv0;
    kv0.reserve(56);
    for (int i = 0; i < 56; ++i) {
      kv0.emplace_back(std::vector<size_t>{1, 1, MaxTokenLength, HiddenSize});
    }
    MemRefContainer2 kvContainer0(kv0[0], kv0[1], tmp2DContainer);
    MemRefContainer2 *kvContainerPtr0 = &kvContainer0;

    //     // Get pointers once to avoid repeated calls
    float *subResultPtr = subResultContainer.getData();
    float *rmsPtr = sub3DContainer.getData();
    int8_t *mhaMemRef4DPtr = mhaMemRef4D.getData();
    float *mhaMemRef3D1Ptr = mhaMemRef3D1.getData();
    float *mhaMemRef3D2Ptr = mhaMemRef3D2.getData();
    float *mhaPtr = tmp3DMemRef.getData();
    float *mhaOutputPtr = tmp2DContainer.getData();
    float *sub2DPtr = sub2DContainer.getData();

    constexpr size_t paramSizeRMS = 1536;
    constexpr size_t paramSizeMHA = 2753536;
    constexpr size_t paramSizeMLP = 20643840;
    int id = rank / 2 - 1;
    int init = id * 84; // 42=6*7 6*14=84,
    int endVal = (id + 1) * 84;
    int times = 14;
    int source = 0;
    int dest2 = rank - 1;
    int nextRank = (rank == 2 ? 4 : 0);
    MPI_Request send_req[2], recv_req[2];
    MPI_Request mha_recv_req[3];
    MPI_Request mha_recv_decode[4];
    // bool mha_params_received = false;
    std::vector<std::string> paramsDirsRMS, paramsDirsRMS0;
    std::vector<std::string> paramsDirsMHA, paramsDirsMLP;
    std::vector<MemRef<float, 1>> paramsContainersRMS, paramsContainersRMS0;
    std::vector<MemRef<float, 1>> paramsContainersMHA, paramsContainersMLP;

    // Get Parameter file paths
    for (int i = init + 1; i < endVal; i += 6) {
      paramsDirsRMS.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i) + "_arg0" + ".data");
      paramsDirsRMS0.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                  std::to_string(i + 3) + "_arg0" + ".data");
      paramsDirsMHA.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i + 1) + "_arg1" + ".data");
      paramsDirsMLP.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i + 4) + "_arg1" + ".data");
    }

    // Load parameters after Bcast to avoid blocking rank 2 at MPI_Barrier
    for (int i = 0; i < times; i++) {
      // First RMS
      MemRef<float, 1> paramsContainerRMS({paramSizeRMS});
      loadParameters(paramsDirsRMS[i], paramsContainerRMS);
      paramsContainersRMS.push_back(paramsContainerRMS);
      // MHA
      MemRef<float, 1> paramsContainerMHA({paramSizeMHA});
      loadParameters(paramsDirsMHA[i], paramsContainerMHA);
      paramsContainersMHA.push_back(paramsContainerMHA);
      // Second RMS
      MemRef<float, 1> paramsContainerRMS0({paramSizeRMS});
      loadParameters(paramsDirsRMS0[i], paramsContainerRMS0);
      paramsContainersRMS0.push_back(paramsContainerRMS0);
      // MLP
      MemRef<float, 1> paramsContainerMLP({paramSizeMLP});
      loadParameters(paramsDirsMLP[i], paramsContainerMLP);
      paramsContainersMLP.push_back(paramsContainerMLP);
    }

    MPI_Irecv(mhaMemRef4DPtr, MaxTokenLength * MaxTokenLength, MPI_FLOAT,
              source, 1, MPI_COMM_WORLD, &mha_recv_req[0]);
    MPI_Irecv(mhaMemRef3D1Ptr, MaxTokenLength * HiddenSize0, MPI_FLOAT, source,
              2, MPI_COMM_WORLD, &mha_recv_req[1]);
    MPI_Irecv(mhaMemRef3D2Ptr, MaxTokenLength * HiddenSize0, MPI_FLOAT, source,
              3, MPI_COMM_WORLD, &mha_recv_req[2]);

    for (int m = 0; m < times; m++) {
      if (m == 0) {
        if (rank == 2) {
          MPI_Irecv(subResultPtr, subSize, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
                    &recv_req[0]);

          MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        } else {
          MPI_Irecv(subResultPtr, subSize, MPI_FLOAT, rank - 2, 0,
                    MPI_COMM_WORLD, &recv_req[0]);

          MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
        }
      }

      _mlir_ciface_forward_prefill1(&sub3DContainer, &paramsContainersRMS[m],
                                    &subResultContainer);
      rmsPtr = sub3DContainer.getData();
      mhaPtr = tmp3DMemRef.getData();

      // ----- AllGather -----
      MPI_Isend(rmsPtr, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                &send_req[0]);
      MPI_Irecv(mhaPtr, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                &recv_req[0]);

      for (int idx = 0; idx < subSize; idx++) {
        mhaPtr[idx + offset0] = rmsPtr[idx];
      }

      MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
      MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

      if (m == 0) {
        MPI_Waitall(3, mha_recv_req, MPI_STATUSES_IGNORE);
      }

      _mlir_ciface_forward_prefill2(kvContainerPtr0, &paramsContainersMHA[m],
                                    &mhaMemRef4D, &mhaMemRef3D1, &mhaMemRef3D2,
                                    &tmp3DMemRef);
      kv0[2 * m] = kvContainerPtr0->kcache;
      kv0[2 * m + 1] = kvContainerPtr0->vcache;

      tmp2DContainer = kvContainerPtr0->data;

      // ----- Reduce-Scatter  -----
      mhaOutputPtr = tmp2DContainer.getData();

      MPI_Isend(mhaOutputPtr, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                &send_req[0]);
      MPI_Irecv(sub2DPtr, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                &recv_req[0]);
      MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
      MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

      for (int idx = 0; idx < subSize; idx++) {
        sub2DPtr[idx] += mhaOutputPtr[idx + offset0];
      }
      // add
      _mlir_ciface_forward_prefill3(&subResultContainer, &subResultContainer,
                                    &sub2DContainer);
      _mlir_ciface_forward_prefill1(&sub3DContainer, &paramsContainersRMS0[m],
                                    &subResultContainer);
      rmsPtr = sub3DContainer.getData();
      mhaPtr = tmp3DMemRef.getData();
      // allgather
      //  ----- AllGather (2路: 仅与rank+1通信) -----
      MPI_Isend(rmsPtr, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                &send_req[0]);
      MPI_Irecv(mhaPtr, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                &recv_req[0]);
      for (int idx = 0; idx < subSize; idx++) {
        mhaPtr[idx + offset0] = rmsPtr[idx];
      }
      MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
      MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
      // mlp
      _mlir_ciface_forward_prefill5(&tmp2DContainer, &paramsContainersMLP[m],
                                    &tmp3DMemRef);
      mhaOutputPtr = tmp2DContainer.getData();

      MPI_Isend(mhaOutputPtr, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                &send_req[0]);
      MPI_Irecv(sub2DPtr, subSize, MPI_FLOAT, dest2, 0, MPI_COMM_WORLD,
                &recv_req[0]);
      MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
      MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

      for (int idx = 0; idx < subSize; idx++) {
        sub2DPtr[idx] += mhaOutputPtr[idx + offset0];
      }
      _mlir_ciface_forward_prefill3(&subResultContainer, &subResultContainer,
                                    &sub2DContainer);
      if (m == (times - 1)) {
        subResultPtr = subResultContainer.getData();
        MPI_Send(subResultPtr, subSize, MPI_FLOAT, nextRank, 0, MPI_COMM_WORLD);
      }
    }

    // decode
    //
    MemRef<long long, 1> cachePosition({1}, 0LL);
    MemRef<float, 3> subResultContainerDecode({1, 1, HiddenSize0});
    MemRef<float, 3> sub3DContainerDecode({1, 1, HiddenSize0});
    MemRef<int8_t, 4> mhaMemRef4DDecode({1, 1, 1, MaxTokenLength});
    MemRef<float, 3> mhaMemRef3D1Decode({1, 1, HiddenSize});
    MemRef<float, 3> mhaMemRef3D2Decode({1, 1, HiddenSize});
    // MemRef<float, 3> tmp3DMemRefDecode({1, MaxTokenLength, HiddenSize0});
    MemRef<float, 2> tmp2DContainerDecode({1, HiddenSize0});
    MemRef<float, 2> sub2DContainerDecode({1, HiddenSize0});
    float *subResultPtrDecode = subResultContainerDecode.getData();
    float *rmsPtrDecode = sub3DContainerDecode.getData();
    int8_t *mhaMemRef4DPtrDecode = mhaMemRef4DDecode.getData();
    float *mhaMemRef3D1PtrDecode = mhaMemRef3D1Decode.getData();
    float *mhaMemRef3D2PtrDecode = mhaMemRef3D2Decode.getData();
    float *mhaPtrDecode = tmp3DMemRef.getData();
    long long int *cachePositionPtr = cachePosition.getData();
    float *mhaOutputPtrDecode = tmp2DContainerDecode.getData();
    float *sub2DPtrDecode = sub2DContainerDecode.getData();

    MemRef<float, 2> mhaDataDecode0({1, HiddenSize0});
    MemRefContainer2 kvDecodeContainer0(kv0[0], kv0[1], tmp2DContainerDecode);
    MemRefContainer2 *kvDecodeContainerPtr0 = &kvDecodeContainer0;

    for (int i = 1; i <= generateLen; i++) {
      MPI_Irecv(mhaMemRef4DPtrDecode, MaxTokenLength, MPI_INT8_T, source, 1,
                MPI_COMM_WORLD, &mha_recv_decode[0]);
      MPI_Irecv(mhaMemRef3D1PtrDecode, HiddenSize, MPI_FLOAT, source, 2,
                MPI_COMM_WORLD, &mha_recv_decode[1]);
      MPI_Irecv(mhaMemRef3D2PtrDecode, HiddenSize, MPI_FLOAT, source, 3,
                MPI_COMM_WORLD, &mha_recv_decode[2]);
      MPI_Irecv(cachePositionPtr, 1, MPI_LONG_LONG, source, 4, MPI_COMM_WORLD,
                &mha_recv_decode[3]);
      for (int m = 0; m < times; m++) {
        if (m == 0) {
          if (rank == 2) {
            MPI_Irecv(subResultPtrDecode, HiddenSize0, MPI_FLOAT, source, 0,
                      MPI_COMM_WORLD, &recv_req[0]);
            MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
          } else {
            MPI_Irecv(subResultPtrDecode, HiddenSize0, MPI_FLOAT, rank - 2, 0,
                      MPI_COMM_WORLD, &recv_req[0]);
            MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
          }
        }
        _mlir_ciface_forward_decode1(&sub3DContainerDecode,
                                     &paramsContainersRMS[m],
                                     &subResultContainerDecode);

        if (m == 0) {

          MPI_Waitall(4, mha_recv_decode, MPI_STATUSES_IGNORE);
        }

        _mlir_ciface_forward_decode2(
            kvDecodeContainerPtr0, &paramsContainersMHA[m], &cachePosition,
            &kv0[2 * m], &kv0[2 * m + 1], &mhaMemRef4DDecode,
            &mhaMemRef3D1Decode, &mhaMemRef3D2Decode, &sub3DContainerDecode);
        kv0[2 * m] = kvDecodeContainerPtr0->kcache;
        kv0[2 * m + 1] = kvDecodeContainerPtr0->vcache;
        tmp2DContainerDecode = kvDecodeContainerPtr0->data;

        // ----- Reduce-Scatter  -----
        mhaOutputPtrDecode = tmp2DContainerDecode.getData();

        MPI_Isend(mhaOutputPtrDecode, HiddenSize0, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtrDecode, HiddenSize0, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        for (int idx = 0; idx < HiddenSize0; idx++) {
          sub2DPtrDecode[idx] += mhaOutputPtrDecode[idx];
        }
        _mlir_ciface_forward_decode3(&subResultContainerDecode,
                                     &subResultContainerDecode,
                                     &sub2DContainerDecode);
        _mlir_ciface_forward_decode1(&sub3DContainerDecode,
                                     &paramsContainersRMS0[m],
                                     &subResultContainerDecode);

        _mlir_ciface_forward_decode5(&tmp2DContainerDecode,
                                     &paramsContainersMLP[m],
                                     &sub3DContainerDecode);
        mhaOutputPtrDecode = tmp2DContainerDecode.getData();
        MPI_Isend(mhaOutputPtrDecode, HiddenSize0, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &send_req[0]);
        MPI_Irecv(sub2DPtrDecode, HiddenSize0, MPI_FLOAT, dest2, 0,
                  MPI_COMM_WORLD, &recv_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

        for (int idx = 0; idx < HiddenSize0; idx++) {
          sub2DPtrDecode[idx] += mhaOutputPtrDecode[idx];
        }
        _mlir_ciface_forward_decode3(&subResultContainerDecode,
                                     &subResultContainerDecode,
                                     &sub2DContainerDecode);
        if (m == (times - 1)) {
          subResultPtrDecode = subResultContainerDecode.getData();
          MPI_Send(subResultPtrDecode, HiddenSize0, MPI_FLOAT, nextRank, 0,
                   MPI_COMM_WORLD);
        }
      }
    }
  }
  return 0;
}
