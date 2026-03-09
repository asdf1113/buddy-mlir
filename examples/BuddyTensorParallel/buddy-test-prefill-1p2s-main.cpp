//===- buddy-deepseek-r1-main.cpp -----------------------------------------===//
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

#include <array>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/time.h>

using namespace buddy;
double total_time = 0;
constexpr size_t ParamsSize0 = 233373760;
constexpr size_t ParamsSize1 = 1536;
constexpr size_t ParamsSize2 = 5507072;
constexpr size_t ParamsSize4 = 1536;
constexpr size_t ParamsSize5 = 41287680;
constexpr size_t ParamsSize7 = 1496916480;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 1024;

constexpr size_t NUM_LAYERS = 56;
constexpr size_t HiddenSize = 128;
constexpr size_t HeadNum = 2;

extern "C" double _mlir_ciface_rtclock() {
#ifndef _WIN32
  struct timeval tp;
  int stat = gettimeofday(&tp, nullptr);
  if (stat != 0)
    fprintf(stderr, "Error returning time from gettimeofday: %d\n", stat);
  return (tp.tv_sec + tp.tv_usec * 1.0e-6);
#else
  fprintf(stderr, "Timing utility not implemented on Windows\n");
  return 0.0;
#endif // _WIN32
}

struct MemRefContainer {

  MemRef<float, 4> kv0;
  MemRef<float, 4> kv1;
  MemRef<float, 4> kv2;
  MemRef<float, 4> kv3;
  MemRef<float, 4> kv4;
  MemRef<float, 4> kv5;
  MemRef<float, 4> kv6;
  MemRef<float, 4> kv7;
  MemRef<float, 4> kv8;
  MemRef<float, 4> kv9;
  MemRef<float, 4> kv10;
  MemRef<float, 4> kv11;
  MemRef<float, 4> kv12;
  MemRef<float, 4> kv13;
  MemRef<float, 4> kv14;
  MemRef<float, 4> kv15;
  MemRef<float, 4> kv16;
  MemRef<float, 4> kv17;
  MemRef<float, 4> kv18;
  MemRef<float, 4> kv19;
  MemRef<float, 4> kv20;
  MemRef<float, 4> kv21;
  MemRef<float, 4> kv22;
  MemRef<float, 4> kv23;
  MemRef<float, 4> kv24;
  MemRef<float, 4> kv25;
  MemRef<float, 4> kv26;
  MemRef<float, 4> kv27;
  MemRef<float, 4> kv28;
  MemRef<float, 4> kv29;
  MemRef<float, 4> kv30;
  MemRef<float, 4> kv31;
  MemRef<float, 4> kv32;
  MemRef<float, 4> kv33;
  MemRef<float, 4> kv34;
  MemRef<float, 4> kv35;
  MemRef<float, 4> kv36;
  MemRef<float, 4> kv37;
  MemRef<float, 4> kv38;
  MemRef<float, 4> kv39;
  MemRef<float, 4> kv40;
  MemRef<float, 4> kv41;
  MemRef<float, 4> kv42;
  MemRef<float, 4> kv43;
  MemRef<float, 4> kv44;
  MemRef<float, 4> kv45;
  MemRef<float, 4> kv46;
  MemRef<float, 4> kv47;
  MemRef<float, 4> kv48;
  MemRef<float, 4> kv49;
  MemRef<float, 4> kv50;
  MemRef<float, 4> kv51;
  MemRef<float, 4> kv52;
  MemRef<float, 4> kv53;
  MemRef<float, 4> kv54;
  MemRef<float, 4> kv55;

  MemRef<float, 3> logits;

  std::array<MemRef<float, 4> *, 56> kv_ptrs;

  MemRefContainer(
      MemRef<float, 4> k0, MemRef<float, 4> k1, MemRef<float, 4> k2,
      MemRef<float, 4> k3, MemRef<float, 4> k4, MemRef<float, 4> k5,
      MemRef<float, 4> k6, MemRef<float, 4> k7, MemRef<float, 4> k8,
      MemRef<float, 4> k9, MemRef<float, 4> k10, MemRef<float, 4> k11,
      MemRef<float, 4> k12, MemRef<float, 4> k13, MemRef<float, 4> k14,
      MemRef<float, 4> k15, MemRef<float, 4> k16, MemRef<float, 4> k17,
      MemRef<float, 4> k18, MemRef<float, 4> k19, MemRef<float, 4> k20,
      MemRef<float, 4> k21, MemRef<float, 4> k22, MemRef<float, 4> k23,
      MemRef<float, 4> k24, MemRef<float, 4> k25, MemRef<float, 4> k26,
      MemRef<float, 4> k27, MemRef<float, 4> k28, MemRef<float, 4> k29,
      MemRef<float, 4> k30, MemRef<float, 4> k31, MemRef<float, 4> k32,
      MemRef<float, 4> k33, MemRef<float, 4> k34, MemRef<float, 4> k35,
      MemRef<float, 4> k36, MemRef<float, 4> k37, MemRef<float, 4> k38,
      MemRef<float, 4> k39, MemRef<float, 4> k40, MemRef<float, 4> k41,
      MemRef<float, 4> k42, MemRef<float, 4> k43, MemRef<float, 4> k44,
      MemRef<float, 4> k45, MemRef<float, 4> k46, MemRef<float, 4> k47,
      MemRef<float, 4> k48, MemRef<float, 4> k49, MemRef<float, 4> k50,
      MemRef<float, 4> k51, MemRef<float, 4> k52, MemRef<float, 4> k53,
      MemRef<float, 4> k54, MemRef<float, 4> k55, MemRef<float, 3> l)
      : kv0(k0), kv1(k1), kv2(k2), kv3(k3), kv4(k4), kv5(k5), kv6(k6), kv7(k7),
        kv8(k8), kv9(k9), kv10(k10), kv11(k11), kv12(k12), kv13(k13), kv14(k14),
        kv15(k15), kv16(k16), kv17(k17), kv18(k18), kv19(k19), kv20(k20),
        kv21(k21), kv22(k22), kv23(k23), kv24(k24), kv25(k25), kv26(k26),
        kv27(k27), kv28(k28), kv29(k29), kv30(k30), kv31(k31), kv32(k32),
        kv33(k33), kv34(k34), kv35(k35), kv36(k36), kv37(k37), kv38(k38),
        kv39(k39), kv40(k40), kv41(k41), kv42(k42), kv43(k43), kv44(k44),
        kv45(k45), kv46(k46), kv47(k47), kv48(k48), kv49(k49), kv50(k50),
        kv51(k51), kv52(k52), kv53(k53), kv54(k54), kv55(k55), logits(l),
        kv_ptrs{&kv0,  &kv1,  &kv2,  &kv3,  &kv4,  &kv5,  &kv6,  &kv7,

                &kv8,  &kv9,  &kv10, &kv11, &kv12, &kv13, &kv14, &kv15,

                &kv16, &kv17, &kv18, &kv19, &kv20, &kv21, &kv22, &kv23,

                &kv24, &kv25, &kv26, &kv27, &kv28, &kv29, &kv30, &kv31,

                &kv32, &kv33, &kv34, &kv35, &kv36, &kv37, &kv38, &kv39,

                &kv40, &kv41, &kv42, &kv43, &kv44, &kv45, &kv46, &kv47,

                &kv48, &kv49, &kv50, &kv51, &kv52, &kv53, &kv54, &kv55} {}
};

struct MemRefContainerPrefill0 {
  MemRef<float, 3> data;
  MemRef<int8_t, 4> mask;
  MemRef<float, 3> cos;
  MemRef<float, 3> sin;

  MemRefContainerPrefill0(MemRef<float, 3> m1, MemRef<int8_t, 4> m2,
                          MemRef<float, 3> m3, MemRef<float, 3> m4)
      : data(m1), mask(m2), cos(m3), sin(m4) {}
};

struct MemRefContainerPrefill2 {
  MemRef<float, 4> kcache;
  MemRef<float, 4> vcache;
  MemRef<float, 2> data;

  MemRefContainerPrefill2(MemRef<float, 4> m1, MemRef<float, 4> m2,
                          MemRef<float, 2> m3)
      : kcache(m1), vcache(m2), data(m3) {}
};

/// Declare DeepSeekR1 forward function.
extern "C" void _mlir_ciface_forward_prefill0(MemRefContainerPrefill0 *,
                                              MemRef<float, 1> *arg0,
                                              Text<size_t, 2> *arg1);
extern "C" void _mlir_ciface_forward_prefill1(MemRef<float, 3> *,
                                              MemRef<float, 1> *arg0,
                                              MemRef<float, 3> *arg1);
extern "C" void
_mlir_ciface_forward_prefill2(MemRefContainerPrefill2 *, MemRef<float, 1> *,
                              MemRef<int8_t, 4> *, MemRef<float, 3> *,
                              MemRef<float, 3> *, MemRef<float, 3> *);
extern "C" void _mlir_ciface_forward_prefill3(MemRef<float, 3> *,
                                              MemRef<float, 3> *,
                                              MemRef<float, 2> *);
extern "C" void _mlir_ciface_forward_prefill5(MemRef<float, 2> *,
                                              MemRef<float, 1> *,
                                              MemRef<float, 3> *);

extern "C" void
_mlir_ciface_forward_prefill7(MemRefContainer *result, MemRef<float, 1> *arg0,
                              MemRef<int8_t, 4> *, MemRef<float, 3> *,
                              MemRef<float, 3> *, MemRef<float, 3> *);

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

int main() {
  /// Print the title of this example.
  const std::string title = "DeepSeekR1 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Define directories of vacabulary and parameter file.
  std::string deepSeekR1Dir = DEEPSEEKR1_EXAMPLE_PATH;
  std::string deepSeekR1BuildDir = DEEPSEEKR1_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = deepSeekR1Dir + "vocab.txt";
  const std::string paramsDirPrefill0 =
      deepSeekR1BuildDir + "subgraph0_prefill0_arg0.data";
  const std::string paramsDirPrefill1 =
      deepSeekR1BuildDir + "subgraph0_prefill1_arg0.data";

  /// Get user message.
  std::string inputStr;
  getUserInput(inputStr);

  /// Initialize data containers
  //  - Input container.
  //  - Result container
  //  - Output container.
  //  - Parameters container.
  Text<size_t, 2> outputContainer;
  Text<size_t, 2> inputContainerPrefill(inputStr);
  MemRef<float, 1> ParamsContainerPrefill0({ParamsSize0});
  MemRef<float, 1> ParamsContainerPrefill1({ParamsSize1});

  MemRef<float, 3> logits_prefill({1, MaxTokenLength, MaxVocabSize});

  // Containers for prefill0 output
  MemRef<float, 3> prefill0_data({1, MaxTokenLength, 1536}); // hidden size
  MemRef<int8_t, 4> prefill0_mask({1, 1, MaxTokenLength, MaxTokenLength});
  MemRef<float, 3> prefill0_cos({1, MaxTokenLength, 128}); // head dim
  MemRef<float, 3> prefill0_sin({1, MaxTokenLength, 128}); // head dim

  // Intermediate result containers
  MemRef<float, 3> rmsmha({1, MaxTokenLength, 1536});
  MemRef<float, 4> mha_result({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRefContainerPrefill2 mha(mha_result, mha_result, 
                              MemRef<float, 2>({MaxTokenLength, 1536}));
  MemRef<float, 3> addmha({1, MaxTokenLength, 1536});
  MemRef<float, 3> rmsmlp({1, MaxTokenLength, 1536});
  MemRef<float, 3> mlp({1, MaxTokenLength, 1536});
  MemRef<float, 3> addmlp({1, MaxTokenLength, 1536});

  // Parameter containers for each stage
  MemRef<float, 1> paramsContainersPrefill2({ParamsSize2});
  MemRef<float, 1> paramsContainersPrefill4({ParamsSize4});
  MemRef<float, 1> paramsContainersPrefill5({ParamsSize5});
  MemRef<float, 1> ParamsContainerPrefill7({ParamsSize7});

  MemRefContainerPrefill0 prefill0ResultContainer(prefill0_data, prefill0_mask,
                                                  prefill0_cos, prefill0_sin);
  MemRefContainerPrefill0 *ptrPrefill0ResultContainer =
      &prefill0ResultContainer;

  MemRef<float, 4> kv0({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv1({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv2({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv3({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv4({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv5({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv6({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv7({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

  MemRef<float, 4> kv8({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv9({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv10({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv11({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv12({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv13({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv14({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv15({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

  MemRef<float, 4> kv16({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv17({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv18({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv19({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv20({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv21({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv22({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv23({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

  MemRef<float, 4> kv24({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv25({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv26({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv27({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv28({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv29({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv30({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv31({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

  MemRef<float, 4> kv32({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv33({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv34({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv35({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv36({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv37({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv38({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv39({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

  MemRef<float, 4> kv40({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv41({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv42({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv43({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv44({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv45({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv46({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv47({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

  MemRef<float, 4> kv48({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv49({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv50({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv51({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv52({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv53({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv54({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv55({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

  MemRefContainer prefillResultContainer(
      kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9, kv10, kv11, kv12, kv13,
      kv14, kv15, kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23, kv24, kv25,
      kv26, kv27, kv28, kv29, kv30, kv31, kv32, kv33, kv34, kv35, kv36, kv37,
      kv38, kv39, kv40, kv41, kv42, kv43, kv44, kv45, kv46, kv47, kv48, kv49,
      kv50, kv51, kv52, kv53, kv54, kv55, logits_prefill);
  MemRefContainer *ptrPrefillResultContainer = &prefillResultContainer;

  /// Fill data into containers
  //  - Input: register vocabulary and tokenize the input string.
  //  - Output: register vocabulary.
  //  - Parameters: load parameters from the `arg0` file into the container.
  tokenizeInput(vocabDir, inputContainerPrefill);
  outputContainer.loadVocab(vocabDir);
  loadParameters(paramsDirPrefill0, ParamsContainerPrefill0);
  loadParameters(paramsDirPrefill1, ParamsContainerPrefill1);

  /// Run DeepSeekR1 Inference
  //  - Perform the forward function.
  //  - Find and append the generated token.
  //  - Continue iterating until the terminal condition is met.

  double prefillTokensPerSec = 0.0;
  const auto inferenceStart = std::chrono::high_resolution_clock::now();

  // Stage 0: Forward prefill0
  _mlir_ciface_forward_prefill0(ptrPrefill0ResultContainer,
                                &ParamsContainerPrefill0,
                                &inputContainerPrefill);
  auto temp = ptrPrefill0ResultContainer->data;
  auto temp_mask = ptrPrefill0ResultContainer->mask;
  auto temp_cos = ptrPrefill0ResultContainer->cos;
  auto temp_sin = ptrPrefill0ResultContainer->sin;

  _mlir_ciface_forward_prefill1(&rmsmha, &paramsContainersPrefill1, &temp);

  _mlir_ciface_forward_prefill2(&mha, &paramsContainersPrefill2, &temp_mask,
                                &temp_cos, &temp_sin, &rmsmha);
  _mlir_ciface_forward_prefill3(&addmha, &temp, &mha);
  _mlir_ciface_forward_prefill1(&rmsmlp, &paramsContainersPrefill4, &addmha);
  _mlir_ciface_forward_prefill5(&mlp, &paramsContainersPrefill5, &rmsmlp);

  _mlir_ciface_forward_prefill3(&addmlp, &addmha, &mlp);

  // Stage 1: Forward prefill1 using prefill0 outputs
  _mlir_ciface_forward_prefill7(ptrPrefillResultContainer,
                                &ParamsContainerPrefill7, &temp_mask, &temp_cos,
                                &temp_sin, &addmlp);

  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
      inferenceEnd - inferenceStart;

  int tokenIndex = inputContainerPrefill.getTokenCnt() - 1;
  const float *startPtr =
      ptrPrefillResultContainer->logits.getData() + tokenIndex * MaxVocabSize;
  const float *endPtr = startPtr + MaxVocabSize;
  int maxIndex = findMaxIndex(startPtr, endPtr);
  std::string tok = inputContainerPrefill.getStr(maxIndex);
  printIterInfo(0, tok, inferenceTime.count() / 1000);
  const double prefillSeconds = inferenceTime.count() / 1000.0;
  if (prefillSeconds > 0.0) {
    prefillTokensPerSec = static_cast<double>(MaxTokenLength) / prefillSeconds;
  }
  outputContainer.appendTokenIdx(maxIndex);

  /// Print the final result
  std::cout << "\n\033[33;1m[Total time]\033[0m " << total_time << std::endl;
  std::cout << "\033[33;1m[Prefilling]\033[0m " << prefillTokensPerSec
            << " tokens/s" << std::endl;
  std::cout << "\033[33;1m[Input]\033[0m " << inputStr << std::endl;
  std::cout << "\033[33;1m[Output]\033[0m "
            << outputContainer.revertDeepSeekR1() << std::endl;

  // Print logits from prefill1 output
  std::cout << "\n\033[33;1m[Logits]\033[0m " << std::endl;
  float *logits_ptr = ptrPrefillResultContainer->logits.getData();
  int logits_size = ptrPrefillResultContainer->logits.getSize();
  std::cout << "Logits size: " << logits_size << std::endl;
  std::cout << "Logits (first 20 values): ";
  for (int i = 0; i < 20 && i < logits_size; ++i) {
    std::cout << logits_ptr[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
