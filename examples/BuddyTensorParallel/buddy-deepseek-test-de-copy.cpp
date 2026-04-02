//===-mha23.cpp--------------------------------------===//

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
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <mpi.h>
#include <sstream>
#include <string>
#include <vector>

using namespace buddy;
double total_time = 0;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 1024;
constexpr size_t SubMaxTokenLength = 512;

constexpr size_t NUM_LAYERS = 56;
constexpr size_t HiddenSize = 128;
constexpr size_t HiddenSize0 = 1536;
constexpr size_t HeadNum = 2;
constexpr bool EnableWorkerDebugLogs = false;
constexpr int FrontendRank = 0;
constexpr int PeerRank = 1;

namespace CommTag {
constexpr int PrefillHidden = 100;
constexpr int PrefillAuxMask = 101;
constexpr int PrefillAuxCos = 102;
constexpr int PrefillAuxSin = 103;
constexpr int PrefillReturnHidden = 104;
constexpr int DecodePacketA = 200;
constexpr int DecodePacketB = 201;
} // namespace CommTag

struct PrefillAuxPacket {
  std::array<int8_t, MaxTokenLength * MaxTokenLength> mask;
  std::array<float, MaxTokenLength * HiddenSize> cos;
  std::array<float, MaxTokenLength * HiddenSize> sin;
};

struct DecodePacketA {
  int ctrl = 0;
  long long cachePosition = 0LL;
  std::array<float, HiddenSize0> hidden;
};

struct DecodePacketB {
  std::array<int8_t, MaxTokenLength> mask;
  std::array<float, HiddenSize> cos;
  std::array<float, HiddenSize> sin;
};

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
struct MemRefContainer2temp {
  MemRef<float, 4> qcache;
  MemRef<float, 4> kcache;
  MemRef<float, 4> vcache;

  MemRefContainer2temp(MemRef<float, 4> m1, MemRef<float, 4> m2,
                   MemRef<float, 4> m3)
      : qcache(m1), kcache(m2), vcache(m3) {}
};

void packPrefillAuxPacket(PrefillAuxPacket &packet, MemRefContainer0 &src) {
  std::memcpy(packet.mask.data(), src.mask.getData(),
              sizeof(int8_t) * MaxTokenLength * MaxTokenLength);
  std::memcpy(packet.cos.data(), src.cos.getData(),
              sizeof(float) * MaxTokenLength * HiddenSize);
  std::memcpy(packet.sin.data(), src.sin.getData(),
              sizeof(float) * MaxTokenLength * HiddenSize);
}

void unpackPrefillAuxPacket(const PrefillAuxPacket &packet, MemRef<int8_t, 4> &mask,
                            MemRef<float, 3> &cos, MemRef<float, 3> &sin) {
  std::memcpy(mask.getData(), packet.mask.data(),
              sizeof(int8_t) * MaxTokenLength * MaxTokenLength);
  std::memcpy(cos.getData(), packet.cos.data(),
              sizeof(float) * MaxTokenLength * HiddenSize);
  std::memcpy(sin.getData(), packet.sin.data(),
              sizeof(float) * MaxTokenLength * HiddenSize);
}

void packDecodePacketA(DecodePacketA &packet, int ctrl, long long cachePosition,
                       const float *hidden) {
  packet.ctrl = ctrl;
  packet.cachePosition = cachePosition;
  std::memcpy(packet.hidden.data(), hidden, sizeof(float) * HiddenSize0);
}

void unpackDecodePacketA(const DecodePacketA &packet, MemRef<float, 3> &hidden,
                         MemRef<long long, 1> &cachePosition) {
  std::memcpy(hidden.getData(), packet.hidden.data(), sizeof(float) * HiddenSize0);
  cachePosition.getData()[0] = packet.cachePosition;
}

void packDecodePacketB(DecodePacketB &packet, MemRefContainer0 &src) {
  std::memcpy(packet.mask.data(), src.mask.getData(), sizeof(int8_t) * MaxTokenLength);
  std::memcpy(packet.cos.data(), src.cos.getData(), sizeof(float) * HiddenSize);
  std::memcpy(packet.sin.data(), src.sin.getData(), sizeof(float) * HiddenSize);
}

void unpackDecodePacketB(const DecodePacketB &packet, MemRef<int8_t, 4> &mask,
                         MemRef<float, 3> &cos, MemRef<float, 3> &sin) {
  std::memcpy(mask.getData(), packet.mask.data(), sizeof(int8_t) * MaxTokenLength);
  std::memcpy(cos.getData(), packet.cos.data(), sizeof(float) * HiddenSize);
  std::memcpy(sin.getData(), packet.sin.data(), sizeof(float) * HiddenSize);
}


/// Declare DeepSeekR1 forward function.
extern "C" {
void _mlir_ciface_forward_prefill0(MemRefContainer0 *, MemRef<float, 1> *,
                                   Text<size_t, 2> *);
void _mlir_ciface_forward_prefill1(MemRef<float, 3> *, MemRef<float, 1> *,
                                   MemRef<float, 3> *);
void _mlir_ciface_forward_prefill2(MemRefContainer2temp *, MemRef<float, 1> *,
                                   MemRef<float, 3> *);
void _mlir_ciface_forward_prefill3(MemRefContainer2 *, MemRef<float, 1> *,
                                   MemRef<int8_t, 4> *, MemRef<float, 3> *,
                                   MemRef<float, 3> *, MemRef<float, 4> *,
                                  MemRef<float, 4> *, MemRef<float, 4> *);

void _mlir_ciface_forward_prefill4(MemRef<float, 3> *, MemRef<float, 3> *,
                                   MemRef<float, 2> *);
void _mlir_ciface_forward_prefill6(MemRef<float, 2> *, MemRef<float, 1> *,
                                   MemRef<float, 3> *);
void _mlir_ciface_forward_prefill197(MemRef<float, 3> *, MemRef<float, 1> *,
                                     MemRef<float, 3> *);

void _mlir_ciface_forward_decode0(MemRefContainer0 *, MemRef<float, 1> *,
                                  MemRef<long long, 2> *,
                                  MemRef<long long, 1> *);
void _mlir_ciface_forward_decode1(MemRef<float, 3> *, MemRef<float, 1> *,
                                  MemRef<float, 3> *);
void _mlir_ciface_forward_decode2(MemRefContainer2temp *, MemRef<float, 1> *,
                                  MemRef<float, 3> *);
void _mlir_ciface_forward_decode3(MemRefContainer2 *, MemRef<float, 1> *,
                                  MemRef<long long, 1> *, MemRef<float, 4> *,
                                  MemRef<float, 4> *, MemRef<int8_t, 4> *,
                                  MemRef<float, 3> *, MemRef<float, 3> *,
                                MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *);
void _mlir_ciface_forward_decode4(MemRef<float, 3> *, MemRef<float, 3> *,
                                  MemRef<float, 2> *);
void _mlir_ciface_forward_decode6(MemRef<float, 2> *, MemRef<float, 1> *,
                                  MemRef<float, 3> *);
void _mlir_ciface_forward_decode197(MemRef<float, 3> *, MemRef<float, 1> *,
                                    MemRef<float, 3> *);
}

using SteadyClock = std::chrono::steady_clock;
using HighResClock = std::chrono::high_resolution_clock;

struct StageEntry {
  std::string phase;
  std::string category;
  std::string stage;
  double seconds = 0.0;
};

struct PhaseStats {
  double wall_s = 0.0;
  double compute_s = 0.0;
  double p2p_s = 0.0;
  double collective_s = 0.0;
  double wait_s = 0.0;
  double log_s = 0.0;
  std::vector<StageEntry> stages;
};

struct RankMetrics {
  int rank = -1;
  double param_load_s = 0.0;
  double param_load_log_s = 0.0;
  double tokenize_s = 0.0;
  double tokenize_log_s = 0.0;
  double input_log_s = 0.0;
  size_t input_token_count = 0;
  double prefill_latency_s = 0.0;
  double prefill_tokps_maxlen = 0.0;
  double prefill_tokps_input = 0.0;
  size_t decode_tokens = 0;
  double decode_total_s = 0.0;
  double decode_avg_latency_s = 0.0;
  double decode_avg_latency_ms = 0.0;
  double decode_tokps = 0.0;
  double final_log_s = 0.0;
  PhaseStats prefill;
  PhaseStats decode;
};

double elapsedSeconds(const SteadyClock::time_point &start) {
  return std::chrono::duration<double>(SteadyClock::now() - start).count();
}

double elapsedSeconds(const HighResClock::time_point &start,
                      const HighResClock::time_point &end) {
  return std::chrono::duration<double>(end - start).count();
}

void addStage(PhaseStats &stats, const std::string &phase,
              const std::string &category, const std::string &stage,
              double seconds) {
  if (seconds < 0.0)
    seconds = 0.0;
  if (category == "compute") {
    stats.compute_s += seconds;
  } else if (category == "p2p") {
    stats.p2p_s += seconds;
  } else if (category == "collective") {
    stats.collective_s += seconds;
  } else if (category == "wait") {
    stats.wait_s += seconds;
  } else if (category == "log") {
    stats.log_s += seconds;
  }
  stats.stages.push_back(StageEntry{phase, category, stage, seconds});
}

std::string csvEscape(const std::string &s) {
  bool needsQuotes = s.find(',') != std::string::npos ||
                     s.find('"') != std::string::npos ||
                     s.find('\n') != std::string::npos;
  if (!needsQuotes)
    return s;
  std::string out = "\"";
  for (char c : s) {
    if (c == '"')
      out += "\"\"";
    else
      out += c;
  }
  out += '"';
  return out;
}

double trackedPhaseSeconds(const PhaseStats &stats) {
  return stats.compute_s + stats.p2p_s + stats.collective_s + stats.wait_s +
         stats.log_s;
}

double safePercent(double part, double total) {
  return total > 0.0 ? (100.0 * part / total) : 0.0;
}

void printPhaseBreakdown(int rank, const std::string &phase,
                         const PhaseStats &stats) {
  const double tracked = trackedPhaseSeconds(stats);
  const double wall = stats.wall_s;
  const double untracked = std::max(0.0, wall - tracked);
  std::cout << "\n\033[36;1m[Rank " << rank << " " << phase
            << " Breakdown]\033[0m" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "  wall_s        = " << wall << std::endl;
  std::cout << "  compute_s     = " << stats.compute_s << " ("
            << safePercent(stats.compute_s, wall) << "%)" << std::endl;
  std::cout << "  p2p_s         = " << stats.p2p_s << " ("
            << safePercent(stats.p2p_s, wall) << "%)" << std::endl;
  std::cout << "  collective_s  = " << stats.collective_s << " ("
            << safePercent(stats.collective_s, wall) << "%)" << std::endl;
  std::cout << "  wait_s        = " << stats.wait_s << " ("
            << safePercent(stats.wait_s, wall) << "%)" << std::endl;
  std::cout << "  log_s         = " << stats.log_s << " ("
            << safePercent(stats.log_s, wall) << "%)" << std::endl;
  std::cout << "  tracked_s     = " << tracked << " ("
            << safePercent(tracked, wall) << "%)" << std::endl;
  std::cout << "  untracked_s   = " << untracked << " ("
            << safePercent(untracked, wall) << "%)" << std::endl;
}

void writeSummaryCsv(const RankMetrics &metrics) {
  std::ofstream ofs("profile_rank" + std::to_string(metrics.rank) +
                    "_summary.csv");
  ofs << "kind,name,value\n";
  ofs << "metric,rank," << metrics.rank << "\n";
  ofs << "metric,param_load_s," << std::setprecision(12)
      << metrics.param_load_s << "\n";
  ofs << "metric,param_load_log_s," << metrics.param_load_log_s << "\n";
  ofs << "metric,tokenize_s," << metrics.tokenize_s << "\n";
  ofs << "metric,tokenize_log_s," << metrics.tokenize_log_s << "\n";
  ofs << "metric,input_log_s," << metrics.input_log_s << "\n";
  ofs << "metric,input_token_count," << metrics.input_token_count << "\n";
  ofs << "metric,prefill_latency_s," << metrics.prefill_latency_s << "\n";
  ofs << "metric,prefill_tokps_maxlen," << metrics.prefill_tokps_maxlen
      << "\n";
  ofs << "metric,prefill_tokps_input," << metrics.prefill_tokps_input
      << "\n";
  ofs << "metric,decode_tokens," << metrics.decode_tokens << "\n";
  ofs << "metric,decode_total_s," << metrics.decode_total_s << "\n";
  ofs << "metric,decode_avg_latency_s," << metrics.decode_avg_latency_s
      << "\n";
  ofs << "metric,decode_avg_latency_ms," << metrics.decode_avg_latency_ms
      << "\n";
  ofs << "metric,decode_tokps," << metrics.decode_tokps << "\n";
  ofs << "metric,final_log_s," << metrics.final_log_s << "\n";

  auto writePhase = [&ofs](const std::string &phase, const PhaseStats &stats) {
    const double tracked = trackedPhaseSeconds(stats);
    const double untracked = std::max(0.0, stats.wall_s - tracked);
    ofs << "phase," << phase << "_wall_s," << stats.wall_s << "\n";
    ofs << "phase," << phase << "_compute_s," << stats.compute_s << "\n";
    ofs << "phase," << phase << "_p2p_s," << stats.p2p_s << "\n";
    ofs << "phase," << phase << "_collective_s," << stats.collective_s
        << "\n";
    ofs << "phase," << phase << "_wait_s," << stats.wait_s << "\n";
    ofs << "phase," << phase << "_log_s," << stats.log_s << "\n";
    ofs << "phase," << phase << "_tracked_s," << tracked << "\n";
    ofs << "phase," << phase << "_untracked_s," << untracked << "\n";
    ofs << "phase_pct," << phase << "_compute_pct,"
        << safePercent(stats.compute_s, stats.wall_s) << "\n";
    ofs << "phase_pct," << phase << "_p2p_pct,"
        << safePercent(stats.p2p_s, stats.wall_s) << "\n";
    ofs << "phase_pct," << phase << "_collective_pct,"
        << safePercent(stats.collective_s, stats.wall_s) << "\n";
    ofs << "phase_pct," << phase << "_wait_pct,"
        << safePercent(stats.wait_s, stats.wall_s) << "\n";
    ofs << "phase_pct," << phase << "_log_pct,"
        << safePercent(stats.log_s, stats.wall_s) << "\n";
    ofs << "phase_pct," << phase << "_tracked_pct,"
        << safePercent(tracked, stats.wall_s) << "\n";
    ofs << "phase_pct," << phase << "_untracked_pct,"
        << safePercent(untracked, stats.wall_s) << "\n";
  };

  writePhase("prefill", metrics.prefill);
  writePhase("decode", metrics.decode);
}

void writeStagesCsv(const RankMetrics &metrics) {
  std::ofstream ofs("profile_rank" + std::to_string(metrics.rank) +
                    "_stages.csv");
  ofs << "phase,category,stage,seconds\n";
  auto emit = [&ofs](const PhaseStats &stats) {
    for (const auto &entry : stats.stages) {
      ofs << csvEscape(entry.phase) << ',' << csvEscape(entry.category) << ','
          << csvEscape(entry.stage) << ',' << std::setprecision(12)
          << entry.seconds << "\n";
    }
  };
  emit(metrics.prefill);
  emit(metrics.decode);
}

void printRankSummary(const RankMetrics &metrics) {
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "\n\033[35;1m[Rank " << metrics.rank
            << " Summary]\033[0m" << std::endl;
  std::cout << "Param load time: " << metrics.param_load_s << "s" << std::endl;
  if (metrics.rank == FrontendRank) {
    std::cout << "Tokenize time: " << metrics.tokenize_s << "s" << std::endl;
    std::cout << "Input token count: " << metrics.input_token_count
              << std::endl;
    std::cout << "Prefill latency: " << metrics.prefill_latency_s << "s"
              << std::endl;
    std::cout << "Prefill throughput (MaxTokenLength/prefillSeconds): "
              << metrics.prefill_tokps_maxlen << " tokens/s" << std::endl;
    std::cout << "Prefill throughput (input_token_count/prefillSeconds): "
              << metrics.prefill_tokps_input << " tokens/s" << std::endl;
    std::cout << "Decode tokens: " << metrics.decode_tokens << std::endl;
    std::cout << "Decode total time: " << metrics.decode_total_s << "s"
              << std::endl;
    std::cout << "Decode avg latency: " << metrics.decode_avg_latency_s
              << "s/token (" << metrics.decode_avg_latency_ms << " ms/token)"
              << std::endl;
    std::cout << "Decode throughput: " << metrics.decode_tokps << " tokens/s"
              << std::endl;
  }
  printPhaseBreakdown(metrics.rank, "prefill", metrics.prefill);
  printPhaseBreakdown(metrics.rank, "decode", metrics.decode);
  std::cout << "CSV written: profile_rank" << metrics.rank
            << "_summary.csv, profile_rank" << metrics.rank
            << "_stages.csv" << std::endl;
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
void printIterInfo(size_t iterIdx, const std::string &str, double time) {
  total_time += time;
  std::cout << "\033[32;1m[Iteration " << iterIdx << "] \033[0m";
  std::cout << "Token: " << str << " | "
            << "Time: " << time << "s" << std::endl;
}

/// Tokenize input data in the container.
void tokenizeInput(const std::string &vocabFile, Text<size_t, 2> &inputContainer,
                   double *tokenizeSeconds = nullptr,
                   double *tokenizeLogSeconds = nullptr) {
  double localLog = 0.0;

  auto tlog = SteadyClock::now();
  printLogLabel();
  std::cout << "Vocab file: " << std::filesystem::canonical(vocabFile)
            << std::endl;
  localLog += elapsedSeconds(tlog);

  const auto buddyTokenizeStart = HighResClock::now();
  inputContainer.tokenizeDeepSeekR1(vocabFile, MaxTokenLength);
  const auto buddyTokenizeEnd = HighResClock::now();
  const double tokenizeTime = elapsedSeconds(buddyTokenizeStart, buddyTokenizeEnd);

  tlog = SteadyClock::now();
  printLogLabel();
  std::cout << "Tokenize time: " << tokenizeTime * 1000.0 << "ms"
            << std::endl;
  localLog += elapsedSeconds(tlog);

  if (tokenizeSeconds)
    *tokenizeSeconds += tokenizeTime;
  if (tokenizeLogSeconds)
    *tokenizeLogSeconds += localLog;
}

/// Load parameters into data container.
void loadParameters(const std::string &paramFilePath, MemRef<float, 1> &params,
                    double *loadSeconds = nullptr,
                    double *loadLogSeconds = nullptr) {
  double localLog = 0.0;
  const auto loadStart = HighResClock::now();
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    std::cout << paramFilePath << std::endl;
    throw std::runtime_error("[Error] Failed to open params file!");
  }

  auto tlog = SteadyClock::now();
  printLogLabel();
  std::cout << "Loading params..." << std::endl;
  printLogLabel();
  std::cout << "Params file: " << std::filesystem::canonical(paramFilePath)
            << std::endl;
  localLog += elapsedSeconds(tlog);

  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * (params.getSize()));
  if (paramFile.fail()) {
    throw std::runtime_error("Error occurred while reading params file!");
  }
  paramFile.close();
  const auto loadEnd = HighResClock::now();
  const double loadTime = elapsedSeconds(loadStart, loadEnd);

  tlog = SteadyClock::now();
  printLogLabel();
  std::cout << "Params load time: " << loadTime << "s\n" << std::endl;
  localLog += elapsedSeconds(tlog);

  if (loadSeconds)
    *loadSeconds += loadTime;
  if (loadLogSeconds)
    *loadLogSeconds += localLog;
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
  std::string deepSeekR1Dir = DEEPSEEKR1_EXAMPLE_PATH;
  std::string deepSeekR1BuildDir = DEEPSEEKR1_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = deepSeekR1Dir + "/vocab.txt";

  // Common variables needed by all ranks
  int subSize = SubMaxTokenLength * HiddenSize0;
  int offset0 = subSize;
  int offset1 = subSize * 2;
  (void)offset1;

  int rank, size;
  int generateLen = MaxTokenLength;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  RankMetrics metrics;
  metrics.rank = rank;

  if (rank == FrontendRank) {
    /// Print the title of this example.
    const std::string title = "DeepSeekR1  Inference Powered by Buddy Compiler";
    std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

    // -----------------------------------------------------------------------
    // Frontend resources originally on rank0.
    // -----------------------------------------------------------------------
    Text<size_t, 2> outputContainer;
    outputContainer.loadVocab(vocabDir);

    MemRef<float, 3> myMemRef1({1, MaxTokenLength, HiddenSize0});
    MemRef<int8_t, 4> myMemRef2({1, 1, MaxTokenLength, MaxTokenLength});
    MemRef<float, 3> myMemRef3({1, MaxTokenLength, HiddenSize});
    MemRef<float, 3> myMemRef4({1, MaxTokenLength, HiddenSize});
    MemRefContainer0 resultContainer(myMemRef1, myMemRef2, myMemRef3,
                                     myMemRef4);
    MemRefContainer0 *resultContainerPtr = &resultContainer;

    MemRef<float, 3> tmp3DMemRef({1, MaxTokenLength, HiddenSize0});
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

    // Load embedding / unembedding parameters on rank1.
    constexpr size_t param_size0 = 233373760;
    const std::string paramsDir0 =
        deepSeekR1BuildDir + "/subgraph0_prefill0_arg0.data";
    constexpr size_t param_size1 = 233375232;
    const std::string paramsDir1 =
        deepSeekR1BuildDir + "/subgraph0_prefill197_arg0.data";
    MemRef<float, 1> paramsContainer0({param_size0});
    loadParameters(paramsDir0, paramsContainer0, &metrics.param_load_s,
                   &metrics.param_load_log_s);
    MemRef<float, 1> paramsContainer1({param_size1});
    loadParameters(paramsDir1, paramsContainer1, &metrics.param_load_s,
                   &metrics.param_load_log_s);

    // -----------------------------------------------------------------------
    // Worker resources originally on rank1.
    // -----------------------------------------------------------------------
    MemRef<float, 3> subResultContainer({1, SubMaxTokenLength, HiddenSize0});
    MemRef<float, 3> sub3DContainer({1, SubMaxTokenLength, HiddenSize0});
    MemRef<float, 2> tmp2DContainer({MaxTokenLength, HiddenSize0});
    MemRef<float, 2> sub2DContainer({SubMaxTokenLength, HiddenSize0});
    std::vector<MemRef<float, 4>> kv0;
    kv0.reserve(56);
    for (int i = 0; i < 56; ++i) {
      kv0.emplace_back(std::vector<size_t>{1, 1, MaxTokenLength, HiddenSize});
    }
    MemRefContainer2 kvContainer0(kv0[0], kv0[1], tmp2DContainer);
    MemRefContainer2 *kvContainerPtr0 = &kvContainer0;

    MemRef<float, 4> tempQ4D({1, 6, MaxTokenLength, HiddenSize});
    MemRef<float, 4> tempK4D({1, 1, MaxTokenLength, HiddenSize});
    MemRef<float, 4> tempV4D({1, 1, MaxTokenLength, HiddenSize});
    MemRefContainer2temp kvContainerTemp(tempQ4D, tempK4D, tempV4D);
    MemRefContainer2temp *kvContainerTempPtr = &kvContainerTemp;

    float *subResultPtr = subResultContainer.getData();
    float *rmsPtr = sub3DContainer.getData();
    float *mhaOutputPtr = tmp2DContainer.getData();
    float *sub2DPtr = sub2DContainer.getData();

    constexpr size_t paramSizeRMS = 1536;
    constexpr size_t paramSizeMHA0 = 1573888;
    constexpr size_t paramSizeMHA1 = 1179648;
    constexpr size_t paramSizeMLP = 20643840;
    int times = 28;
    int peerRank = PeerRank;

    std::vector<std::string> paramsDirsRMS, paramsDirsRMS0;
    std::vector<std::string> paramsDirsMHA0, paramsDirsMHA1, paramsDirsMLP;
    std::vector<MemRef<float, 1>> paramsContainersRMS, paramsContainersRMS0;
    std::vector<MemRef<float, 1>> paramsContainersMHA0, paramsContainersMHA1,
        paramsContainersMLP;

    for (int i = 1; i < 197; i += 7) {
      paramsDirsRMS.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i) + "_arg0.data");
      paramsDirsRMS0.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                  std::to_string(i + 4) + "_arg0.data");
    }
    for (int i = 2; i < 197; i += 7) {
      paramsDirsMHA0.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                  std::to_string(i) + "_arg0.data");
      paramsDirsMHA1.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                  std::to_string(i + 1) + "_arg0.data");
    }
    for (int i = 6; i < 197; i += 7) {
      paramsDirsMLP.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i) + "_arg0.data");
    }

    for (int i = 0; i < times; i++) {
      MemRef<float, 1> paramsContainerRMS({paramSizeRMS});
      loadParameters(paramsDirsRMS[i], paramsContainerRMS, &metrics.param_load_s,
                     &metrics.param_load_log_s);
      paramsContainersRMS.push_back(paramsContainerRMS);

      MemRef<float, 1> paramsContainerMHA0({paramSizeMHA0});
      loadParameters(paramsDirsMHA0[i], paramsContainerMHA0, &metrics.param_load_s,
                     &metrics.param_load_log_s);
      paramsContainersMHA0.push_back(paramsContainerMHA0);

      MemRef<float, 1> paramsContainerMHA1({paramSizeMHA1});
      loadParameters(paramsDirsMHA1[i], paramsContainerMHA1, &metrics.param_load_s,
                     &metrics.param_load_log_s);
      paramsContainersMHA1.push_back(paramsContainerMHA1);

      MemRef<float, 1> paramsContainerRMS0({paramSizeRMS});
      loadParameters(paramsDirsRMS0[i], paramsContainerRMS0,
                     &metrics.param_load_s, &metrics.param_load_log_s);
      paramsContainersRMS0.push_back(paramsContainerRMS0);

      MemRef<float, 1> paramsContainerMLP({paramSizeMLP});
      loadParameters(paramsDirsMLP[i], paramsContainerMLP, &metrics.param_load_s,
                     &metrics.param_load_log_s);
      paramsContainersMLP.push_back(paramsContainerMLP);
    }

    MPI_Comm comm_sub = MPI_COMM_NULL;
    MPI_Group world_group = MPI_GROUP_NULL;
    MPI_Group sub_group = MPI_GROUP_NULL;
    int ranks_in_sub[2] = {FrontendRank, PeerRank};
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, 2, ranks_in_sub, &sub_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &comm_sub);

    std::string inputStr;
    auto tInputLog = SteadyClock::now();
    getUserInput(inputStr);
    metrics.input_log_s += elapsedSeconds(tInputLog);
    Text<size_t, 2> inputContainerPrefill(inputStr);
    tokenizeInput(vocabDir, inputContainerPrefill, &metrics.tokenize_s,
                  &metrics.tokenize_log_s);
    metrics.input_token_count = inputContainerPrefill.getTokenCnt();
    tInputLog = SteadyClock::now();
    std::cout << "Input token count: " << inputContainerPrefill.getTokenCnt()
              << std::endl;
    metrics.input_log_s += elapsedSeconds(tInputLog);

    MPI_Request send_req[2], recv_req[2];
    MPI_Request send_req_mha[3];
    DecodePacketA decodePacketASend;
    DecodePacketB decodePacketBSend;

    float *inputPtr = nullptr;
    float *outputPtr = tmp3DMemRef.getData();
    float *outputPtrDecode = myMemRef_decode1.getData();
    (void)outputPtrDecode;

    double prefillTokensPerSec = 0.0;
    double prefillTokensPerSecInput = 0.0;
    const auto inferenceStart = HighResClock::now();

    auto tlog = SteadyClock::now();
    std::cout << "\n\033[33;1m[Inference Start]\033[0m" << std::endl;
    addStage(metrics.prefill, "prefill", "log", "rank1.prefill.print_start",
             elapsedSeconds(tlog));

    auto t0 = SteadyClock::now();
    _mlir_ciface_forward_prefill0(resultContainerPtr, &paramsContainer0,
                                  &inputContainerPrefill);
    addStage(metrics.prefill, "prefill", "compute",
             "rank1.prefill.forward_prefill0", elapsedSeconds(t0));
    inputPtr = resultContainerPtr->data.getData();

    t0 = SteadyClock::now();
    std::memcpy(subResultPtr, inputPtr, sizeof(float) * subSize);
    addStage(metrics.prefill, "prefill", "compute",
             "rank1.prefill.copy_local_half", elapsedSeconds(t0));

    t0 = SteadyClock::now();
    MPI_Isend(inputPtr + offset0, subSize, MPI_FLOAT, peerRank,
              CommTag::PrefillHidden, MPI_COMM_WORLD, &send_req[0]);
    addStage(metrics.prefill, "prefill", "p2p", "rank1.prefill.isend_hidden_half",
             elapsedSeconds(t0));

    t0 = SteadyClock::now();
    MPI_Isend(resultContainerPtr->mask.getData(),
              MaxTokenLength * MaxTokenLength, MPI_BYTE, peerRank,
              CommTag::PrefillAuxMask, MPI_COMM_WORLD, &send_req_mha[0]);
    MPI_Isend(resultContainerPtr->cos.getData(),
              MaxTokenLength * HiddenSize, MPI_FLOAT, peerRank,
              CommTag::PrefillAuxCos, MPI_COMM_WORLD, &send_req_mha[1]);
    MPI_Isend(resultContainerPtr->sin.getData(),
              MaxTokenLength * HiddenSize, MPI_FLOAT, peerRank,
              CommTag::PrefillAuxSin, MPI_COMM_WORLD, &send_req_mha[2]);
    addStage(metrics.prefill, "prefill", "p2p",
             "rank1.prefill.isend_aux_triplet", elapsedSeconds(t0));

    t0 = SteadyClock::now();
    MPI_Irecv(outputPtr + offset0, subSize, MPI_FLOAT, peerRank,
              CommTag::PrefillReturnHidden, MPI_COMM_WORLD, &recv_req[0]);
    addStage(metrics.prefill, "prefill", "p2p",
             "rank1.prefill.irecv_peer_half", elapsedSeconds(t0));

    for (int m = 0; m < times; m++) {
      t0 = SteadyClock::now();
      _mlir_ciface_forward_prefill1(&sub3DContainer, &paramsContainersRMS[m],
                                    &subResultContainer);
      addStage(metrics.prefill, "prefill", "compute",
               "rank1.prefill.forward_prefill1.rms1", elapsedSeconds(t0));
      rmsPtr = sub3DContainer.getData();

      if (comm_sub != MPI_COMM_NULL) {
        t0 = SteadyClock::now();
        MPI_Allgather(rmsPtr, subSize, MPI_FLOAT, tmp3DMemRef.getData(),
                      subSize, MPI_FLOAT, comm_sub);
        addStage(metrics.prefill, "prefill", "collective",
                 "rank1.prefill.allgather_before_mha", elapsedSeconds(t0));
      }

      t0 = SteadyClock::now();
      _mlir_ciface_forward_prefill2(kvContainerTempPtr, &paramsContainersMHA0[m],
                                    &tmp3DMemRef);
      addStage(metrics.prefill, "prefill", "compute",
               "rank1.prefill.forward_prefill2.part2", elapsedSeconds(t0));

      t0 = SteadyClock::now();
      _mlir_ciface_forward_prefill3(kvContainerPtr0, &paramsContainersMHA1[m],
                                    &resultContainerPtr->mask,
                                    &resultContainerPtr->cos,
                                    &resultContainerPtr->sin,
                                    &kvContainerTempPtr->qcache,
                                    &kvContainerTempPtr->kcache,
                                    &kvContainerTempPtr->vcache);
      addStage(metrics.prefill, "prefill", "compute",
               "rank1.prefill.forward_prefill3.part3", elapsedSeconds(t0));

      kv0[2 * m] = kvContainerPtr0->kcache;
      kv0[2 * m + 1] = kvContainerPtr0->vcache;
      tmp2DContainer = kvContainerPtr0->data;
      mhaOutputPtr = tmp2DContainer.getData();

      if (comm_sub != MPI_COMM_NULL) {
        t0 = SteadyClock::now();
        MPI_Reduce_scatter_block(mhaOutputPtr, sub2DPtr, subSize, MPI_FLOAT,
                                 MPI_SUM, comm_sub);
        addStage(metrics.prefill, "prefill", "collective",
                 "rank1.prefill.reduce_scatter_after_mha", elapsedSeconds(t0));
      }

      t0 = SteadyClock::now();
      _mlir_ciface_forward_prefill4(&subResultContainer, &subResultContainer,
                                    &sub2DContainer);
      addStage(metrics.prefill, "prefill", "compute",
               "rank1.prefill.forward_prefill4.add1", elapsedSeconds(t0));

      t0 = SteadyClock::now();
      _mlir_ciface_forward_prefill1(&sub3DContainer, &paramsContainersRMS0[m],
                                    &subResultContainer);
      addStage(metrics.prefill, "prefill", "compute",
               "rank1.prefill.forward_prefill1.rms2", elapsedSeconds(t0));
      rmsPtr = sub3DContainer.getData();

      if (comm_sub != MPI_COMM_NULL) {
        t0 = SteadyClock::now();
        MPI_Allgather(rmsPtr, subSize, MPI_FLOAT, tmp3DMemRef.getData(),
                      subSize, MPI_FLOAT, comm_sub);
        addStage(metrics.prefill, "prefill", "collective",
                 "rank1.prefill.allgather_before_mlp", elapsedSeconds(t0));
      }

      t0 = SteadyClock::now();
      _mlir_ciface_forward_prefill6(&tmp2DContainer, &paramsContainersMLP[m],
                                    &tmp3DMemRef);
      addStage(metrics.prefill, "prefill", "compute",
               "rank1.prefill.forward_prefill6.mlp", elapsedSeconds(t0));
      mhaOutputPtr = tmp2DContainer.getData();

      if (comm_sub != MPI_COMM_NULL) {
        t0 = SteadyClock::now();
        MPI_Reduce_scatter_block(mhaOutputPtr, sub2DPtr, subSize, MPI_FLOAT,
                                 MPI_SUM, comm_sub);
        addStage(metrics.prefill, "prefill", "collective",
                 "rank1.prefill.reduce_scatter_after_mlp", elapsedSeconds(t0));
      }

      t0 = SteadyClock::now();
      _mlir_ciface_forward_prefill4(&subResultContainer, &subResultContainer,
                                    &sub2DContainer);
      addStage(metrics.prefill, "prefill", "compute",
               "rank1.prefill.forward_prefill4.add2", elapsedSeconds(t0));
    }

    t0 = SteadyClock::now();
    std::memcpy(outputPtr, subResultContainer.getData(), sizeof(float) * subSize);
    addStage(metrics.prefill, "prefill", "compute",
             "rank1.prefill.copy_local_half_back", elapsedSeconds(t0));

    t0 = SteadyClock::now();
    MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
    addStage(metrics.prefill, "prefill", "wait",
             "rank1.prefill.wait_peer_half", elapsedSeconds(t0));

    MPI_Request prefillSends[4] = {send_req[0], send_req_mha[0],
                                   send_req_mha[1], send_req_mha[2]};
    t0 = SteadyClock::now();
    MPI_Waitall(4, prefillSends, MPI_STATUSES_IGNORE);
    addStage(metrics.prefill, "prefill", "wait",
             "rank1.prefill.wait_send_complete", elapsedSeconds(t0));

    t0 = SteadyClock::now();
    _mlir_ciface_forward_prefill197(&resultPrefill, &paramsContainer1,
                                    &tmp3DMemRef);
    addStage(metrics.prefill, "prefill", "compute",
             "rank1.prefill.forward_prefill197", elapsedSeconds(t0));

    const auto inferenceEnd = HighResClock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;
    metrics.prefill_latency_s = inferenceTime.count() / 1000.0;
    metrics.prefill.wall_s = metrics.prefill_latency_s;

    int tokenIndex = inputContainerPrefill.getTokenCnt() - 1;
    const float *startPtr = resultPrefill.getData() + tokenIndex * MaxVocabSize;
    const float *endPtr = startPtr + MaxVocabSize;
    int maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = inputContainerPrefill.getStr(maxIndex);

    tlog = SteadyClock::now();
    printIterInfo(0, tok, inferenceTime.count() / 1000.0);
    addStage(metrics.prefill, "prefill", "log", "rank1.prefill.print_iter0",
             elapsedSeconds(tlog));

    const double prefillSeconds = inferenceTime.count() / 1000.0;
    if (prefillSeconds > 0.0) {
      prefillTokensPerSec = static_cast<double>(MaxTokenLength) / prefillSeconds;
      prefillTokensPerSecInput =
          static_cast<double>(inputContainerPrefill.getTokenCnt()) /
          prefillSeconds;
    }
    metrics.prefill_tokps_maxlen = prefillTokensPerSec;
    metrics.prefill_tokps_input = prefillTokensPerSecInput;

    inputContainerDecode.getData()[0] = (long long)maxIndex;
    outputContainer.appendTokenIdx(maxIndex);
    cachePosition.getData()[0] = inputContainerPrefill.getTokenCnt();
    generateLen = MaxTokenLength - inputContainerPrefill.getTokenCnt();
    double decodeTimeAccumMs = 0.0;
    size_t decodeTokens = 0;

    MPI_Request send_req_decode[2];
    bool sentStop = false;

    // -----------------------------------------------------------------------
    // Local decode worker resources (rank1 shard).
    // Keep a worker-local snapshot of decode inputs so the merged frontend path
    // still follows the original worker dataflow.
    // -----------------------------------------------------------------------
    MemRef<float, 3> subResultContainerDecode({1, 1, HiddenSize0});
    MemRef<float, 3> sub3DContainerDecode({1, 1, HiddenSize0});
    MemRef<float, 2> tmp2DContainerDecode({1, HiddenSize0});
    MemRef<float, 2> sub2DContainerDecode({1, HiddenSize0});
    // Do not cache subResultContainerDecode.getData() across iterations.
    // MLIR iface calls may update the MemRef descriptor, so a cached raw pointer
    // can become stale and point to the previous iteration's buffer.
    float *mhaOutputPtrDecode = tmp2DContainerDecode.getData();
    float *sub2DPtrDecode = sub2DContainerDecode.getData();
    (void)sub2DPtrDecode;

    MemRefContainer2 kvDecodeContainer0(kv0[0], kv0[1], tmp2DContainerDecode);
    MemRefContainer2 *kvDecodeContainerPtr0 = &kvDecodeContainer0;

    MemRef<float, 4> tempDecodeQ4D({1, 6, 1, HiddenSize});
    MemRef<float, 4> tempDecodeK4D({1, 1, 1, HiddenSize});
    MemRef<float, 4> tempDecodeV4D({1, 1, 1, HiddenSize});
    MemRefContainer2temp kvDecodeContainerTemp(tempDecodeQ4D, tempDecodeK4D,
                                               tempDecodeV4D);
    MemRefContainer2temp *kvDecodeContainerTempPtr = &kvDecodeContainerTemp;

    for (int i = 1; i <= generateLen; i++) {
      const auto decodeIterStart = HighResClock::now();
      t0 = SteadyClock::now();
      _mlir_ciface_forward_decode0(resultContainerDecodePtr, &paramsContainer0,
                                   &inputContainerDecode, &cachePosition);
      addStage(metrics.decode, "decode", "compute",
               "rank1.decode.forward_decode0", elapsedSeconds(t0));
      inputPtr = resultContainerDecodePtr->data.getData();

      t0 = SteadyClock::now();
      std::memcpy(subResultContainerDecode.getData(), inputPtr,
                  sizeof(float) * HiddenSize0);
      packDecodePacketA(decodePacketASend, 0, cachePosition.getData()[0],
                        inputPtr);
      packDecodePacketB(decodePacketBSend, *resultContainerDecodePtr);
      addStage(metrics.decode, "decode", "compute",
               "rank1.decode.copy_hidden_and_pack_packets", elapsedSeconds(t0));

      t0 = SteadyClock::now();
      MPI_Isend(reinterpret_cast<const void *>(&decodePacketASend),
                static_cast<int>(sizeof(DecodePacketA)), MPI_BYTE, peerRank,
                CommTag::DecodePacketA, MPI_COMM_WORLD, &send_req_decode[0]);
      MPI_Isend(reinterpret_cast<const void *>(&decodePacketBSend),
                static_cast<int>(sizeof(DecodePacketB)), MPI_BYTE, peerRank,
                CommTag::DecodePacketB, MPI_COMM_WORLD, &send_req_decode[1]);
      addStage(metrics.decode, "decode", "p2p",
               "rank1.decode.isend_packet_a_b", elapsedSeconds(t0));

      for (int m = 0; m < times; m++) {
        t0 = SteadyClock::now();
        _mlir_ciface_forward_decode1(&sub3DContainerDecode,
                                     &paramsContainersRMS[m],
                                     &subResultContainerDecode);
        addStage(metrics.decode, "decode", "compute",
                 "rank1.decode.forward_decode1.rms1", elapsedSeconds(t0));

        t0 = SteadyClock::now();
        _mlir_ciface_forward_decode2(kvDecodeContainerTempPtr,
                                     &paramsContainersMHA0[m],
                                     &sub3DContainerDecode);
        addStage(metrics.decode, "decode", "compute",
                 "rank1.decode.forward_decode2.part2", elapsedSeconds(t0));

        t0 = SteadyClock::now();
        _mlir_ciface_forward_decode3(
            kvDecodeContainerPtr0, &paramsContainersMHA1[m],
            &cachePosition, &kv0[2 * m], &kv0[2 * m + 1],
            &resultContainerDecodePtr->mask, &resultContainerDecodePtr->cos,
            &resultContainerDecodePtr->sin, &kvDecodeContainerTempPtr->qcache,
            &kvDecodeContainerTempPtr->kcache,
            &kvDecodeContainerTempPtr->vcache);
        addStage(metrics.decode, "decode", "compute",
                 "rank1.decode.forward_decode3.part3", elapsedSeconds(t0));

        kv0[2 * m] = kvDecodeContainerPtr0->kcache;
        kv0[2 * m + 1] = kvDecodeContainerPtr0->vcache;
        tmp2DContainerDecode = kvDecodeContainerPtr0->data;
        mhaOutputPtrDecode = tmp2DContainerDecode.getData();

        if (comm_sub != MPI_COMM_NULL) {
          t0 = SteadyClock::now();
          MPI_Allreduce(mhaOutputPtrDecode, sub2DContainerDecode.getData(),
                        HiddenSize0, MPI_FLOAT, MPI_SUM, comm_sub);
          addStage(metrics.decode, "decode", "collective",
                   "rank1.decode.allreduce_after_mha", elapsedSeconds(t0));
        }

        t0 = SteadyClock::now();
        _mlir_ciface_forward_decode4(&subResultContainerDecode,
                                     &subResultContainerDecode,
                                     &sub2DContainerDecode);
        addStage(metrics.decode, "decode", "compute",
                 "rank1.decode.forward_decode4.add1", elapsedSeconds(t0));

        t0 = SteadyClock::now();
        _mlir_ciface_forward_decode1(&sub3DContainerDecode,
                                     &paramsContainersRMS0[m],
                                     &subResultContainerDecode);
        addStage(metrics.decode, "decode", "compute",
                 "rank1.decode.forward_decode1.rms2", elapsedSeconds(t0));

        t0 = SteadyClock::now();
        _mlir_ciface_forward_decode6(&tmp2DContainerDecode,
                                     &paramsContainersMLP[m],
                                     &sub3DContainerDecode);
        addStage(metrics.decode, "decode", "compute",
                 "rank1.decode.forward_decode6.mlp", elapsedSeconds(t0));
        mhaOutputPtrDecode = tmp2DContainerDecode.getData();

        if (comm_sub != MPI_COMM_NULL) {
          t0 = SteadyClock::now();
          MPI_Allreduce(mhaOutputPtrDecode, sub2DContainerDecode.getData(),
                        HiddenSize0, MPI_FLOAT, MPI_SUM, comm_sub);
          addStage(metrics.decode, "decode", "collective",
                   "rank1.decode.allreduce_after_mlp", elapsedSeconds(t0));
        }

        t0 = SteadyClock::now();
        _mlir_ciface_forward_decode4(&subResultContainerDecode,
                                     &subResultContainerDecode,
                                     &sub2DContainerDecode);
        addStage(metrics.decode, "decode", "compute",
                 "rank1.decode.forward_decode4.add2", elapsedSeconds(t0));

      }

      t0 = SteadyClock::now();
      MPI_Waitall(2, send_req_decode, MPI_STATUSES_IGNORE);
      addStage(metrics.decode, "decode", "wait",
               "rank1.decode.wait_send_complete", elapsedSeconds(t0));

      t0 = SteadyClock::now();
      _mlir_ciface_forward_decode197(&resultDecode, &paramsContainer1,
                                     &subResultContainerDecode);
      addStage(metrics.decode, "decode", "compute",
               "rank1.decode.forward_decode197", elapsedSeconds(t0));

      const auto decodeIterEnd = HighResClock::now();
      const std::chrono::duration<double, std::milli> decodeIterTime =
          decodeIterEnd - decodeIterStart;
      decodeTimeAccumMs += decodeIterTime.count();
      decodeTokens += 1;

      const float *decodeStartPtr = resultDecode.getData();
      const float *decodeEndPtr = decodeStartPtr + MaxVocabSize;
      int maxIndex = findMaxIndex(decodeStartPtr, decodeEndPtr);
      std::string tok = inputContainerPrefill.getStr(maxIndex);

      tlog = SteadyClock::now();
      printIterInfo(i, tok, decodeIterTime.count() / 1000.0);
      addStage(metrics.decode, "decode", "log", "rank1.decode.print_iter",
               elapsedSeconds(tlog));

      if (maxIndex == 151643) {
        packDecodePacketA(decodePacketASend, 1, cachePosition.getData()[0],
                          subResultContainerDecode.getData());
        t0 = SteadyClock::now();
        MPI_Send(reinterpret_cast<const void *>(&decodePacketASend),
                 static_cast<int>(sizeof(DecodePacketA)), MPI_BYTE, peerRank,
                 CommTag::DecodePacketA, MPI_COMM_WORLD);
        addStage(metrics.decode, "decode", "p2p",
                 "rank1.decode.send_stop_packet_a", elapsedSeconds(t0));
        sentStop = true;
        break;
      }

      inputContainerDecode.getData()[0] = maxIndex;
      outputContainer.appendTokenIdx(maxIndex);
      cachePosition.getData()[0] += 1;
    }

    if (!sentStop) {
      packDecodePacketA(decodePacketASend, 1, cachePosition.getData()[0],
                        subResultContainerDecode.getData());
      t0 = SteadyClock::now();
      MPI_Send(reinterpret_cast<const void *>(&decodePacketASend),
               static_cast<int>(sizeof(DecodePacketA)), MPI_BYTE, peerRank,
               CommTag::DecodePacketA, MPI_COMM_WORLD);
      addStage(metrics.decode, "decode", "p2p",
               "rank1.decode.send_final_stop_packet_a", elapsedSeconds(t0));
    }

    double decodeSeconds = decodeTimeAccumMs / 1000.0;
    const double decodeTokensPerSec =
        decodeSeconds > 0.0 ? static_cast<double>(decodeTokens) / decodeSeconds
                            : 0.0;

    metrics.decode_tokens = decodeTokens;
    metrics.decode_total_s = decodeSeconds;
    metrics.decode_avg_latency_s =
        decodeTokens > 0 ? decodeSeconds / static_cast<double>(decodeTokens)
                         : 0.0;
    metrics.decode_avg_latency_ms = metrics.decode_avg_latency_s * 1000.0;
    metrics.decode_tokps = decodeTokensPerSec;
    metrics.decode.wall_s = decodeSeconds;

    tlog = SteadyClock::now();
    std::cout << "\n\033[33;1m[Total time]\033[0m " << total_time
              << std::endl;
    std::cout << "\033[33;1m[Prefilling]\033[0m " << prefillTokensPerSec
              << " tokens/s" << std::endl;
    std::cout << "\033[33;1m[Prefilling-real]\033[0m "
              << prefillTokensPerSecInput << " tokens/s" << std::endl;
    std::cout << "\033[33;1m[Prefill latency]\033[0m " << prefillSeconds
              << " s" << std::endl;
    std::cout << "\033[33;1m[Decoding]\033[0m " << decodeTokensPerSec
              << " tokens/s" << std::endl;
    std::cout << "\033[33;1m[Decode avg latency]\033[0m "
              << metrics.decode_avg_latency_ms << " ms/token" << std::endl;
    std::cout << "\033[33;1m[Input]\033[0m " << inputStr << std::endl;
    std::cout << "\033[33;1m[Output]\033[0m "
              << outputContainer.revertDeepSeekR1() << std::endl;
    metrics.final_log_s += elapsedSeconds(tlog);

    if (comm_sub != MPI_COMM_NULL) {
      MPI_Comm_free(&comm_sub);
    }
    if (sub_group != MPI_GROUP_NULL) {
      MPI_Group_free(&sub_group);
    }
    if (world_group != MPI_GROUP_NULL) {
      MPI_Group_free(&world_group);
    }

  } else if (rank == PeerRank) {

    // === Worker shard on rank2 ===
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

    MemRef<float, 4> tempQ4D({1, 6, MaxTokenLength, HiddenSize});
    MemRef<float, 4> tempK4D({1, 1, MaxTokenLength, HiddenSize});
    MemRef<float, 4> tempV4D({1, 1, MaxTokenLength, HiddenSize});
    MemRefContainer2temp kvContainerTemp(tempQ4D, tempK4D, tempV4D);
    MemRefContainer2temp *kvContainerTempPtr = &kvContainerTemp;

    float *subResultPtr = subResultContainer.getData();
    float *rmsPtr = sub3DContainer.getData();
    float *mhaOutputPtr = tmp2DContainer.getData();
    float *sub2DPtr = sub2DContainer.getData();

    constexpr size_t paramSizeRMS = 1536;
    constexpr size_t paramSizeMHA0 = 1573888;
    constexpr size_t paramSizeMHA1 = 1179648;
    constexpr size_t paramSizeMLP = 20643840;
    int times = 28;
    int source = FrontendRank;
    int nextRank = FrontendRank;
    MPI_Request recv_req[2];
    MPI_Request mha_recv_req[3];
    MPI_Request decode_recv_packet_b = MPI_REQUEST_NULL;
    DecodePacketA decodePacketARecv;
    DecodePacketB decodePacketBRecv;
    std::vector<std::string> paramsDirsRMS, paramsDirsRMS0;
    std::vector<std::string> paramsDirsMHA0, paramsDirsMHA1, paramsDirsMLP;
    std::vector<MemRef<float, 1>> paramsContainersRMS, paramsContainersRMS0;
    std::vector<MemRef<float, 1>> paramsContainersMHA0, paramsContainersMHA1,
        paramsContainersMLP;

    for (int i = 1; i < 197; i += 7) {
      paramsDirsRMS.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i) + "_arg0.data");
      paramsDirsRMS0.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                  std::to_string(i + 4) + "_arg0.data");
    }
    for (int i = 2; i < 197; i += 7) {
      paramsDirsMHA0.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                  std::to_string(i) + "_arg1.data");
      paramsDirsMHA1.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                  std::to_string(i + 1) + "_arg1.data");
    }
    for (int i = 6; i < 197; i += 7) {
      paramsDirsMLP.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i) + "_arg1.data");
    }

    for (int i = 0; i < times; i++) {
      MemRef<float, 1> paramsContainerRMS({paramSizeRMS});
      loadParameters(paramsDirsRMS[i], paramsContainerRMS, &metrics.param_load_s,
                     &metrics.param_load_log_s);
      paramsContainersRMS.push_back(paramsContainerRMS);

      MemRef<float, 1> paramsContainerMHA0({paramSizeMHA0});
      loadParameters(paramsDirsMHA0[i], paramsContainerMHA0, &metrics.param_load_s,
                     &metrics.param_load_log_s);
      paramsContainersMHA0.push_back(paramsContainerMHA0);

      MemRef<float, 1> paramsContainerMHA1({paramSizeMHA1});
      loadParameters(paramsDirsMHA1[i], paramsContainerMHA1, &metrics.param_load_s,
                     &metrics.param_load_log_s);
      paramsContainersMHA1.push_back(paramsContainerMHA1);

      MemRef<float, 1> paramsContainerRMS0({paramSizeRMS});
      loadParameters(paramsDirsRMS0[i], paramsContainerRMS0,
                     &metrics.param_load_s, &metrics.param_load_log_s);
      paramsContainersRMS0.push_back(paramsContainerRMS0);

      MemRef<float, 1> paramsContainerMLP({paramSizeMLP});
      loadParameters(paramsDirsMLP[i], paramsContainerMLP, &metrics.param_load_s,
                     &metrics.param_load_log_s);
      paramsContainersMLP.push_back(paramsContainerMLP);
    }

    MPI_Comm comm_sub = MPI_COMM_NULL;
    MPI_Group world_group = MPI_GROUP_NULL;
    MPI_Group sub_group = MPI_GROUP_NULL;
    int ranks_in_sub[2] = {FrontendRank, PeerRank};
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, 2, ranks_in_sub, &sub_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &comm_sub);

    auto workerPrefillStart = SteadyClock::now();

    auto t0 = SteadyClock::now();
    MPI_Irecv(mhaMemRef4D.getData(), MaxTokenLength * MaxTokenLength, MPI_BYTE,
              source, CommTag::PrefillAuxMask, MPI_COMM_WORLD,
              &mha_recv_req[0]);
    MPI_Irecv(mhaMemRef3D1.getData(), MaxTokenLength * HiddenSize, MPI_FLOAT,
              source, CommTag::PrefillAuxCos, MPI_COMM_WORLD,
              &mha_recv_req[1]);
    MPI_Irecv(mhaMemRef3D2.getData(), MaxTokenLength * HiddenSize, MPI_FLOAT,
              source, CommTag::PrefillAuxSin, MPI_COMM_WORLD,
              &mha_recv_req[2]);
    addStage(metrics.prefill, "prefill", "p2p",
             "rank2.prefill.irecv_aux_triplet", elapsedSeconds(t0));

    t0 = SteadyClock::now();
    MPI_Irecv(subResultPtr, subSize, MPI_FLOAT, source, CommTag::PrefillHidden,
              MPI_COMM_WORLD, &recv_req[0]);
    addStage(metrics.prefill, "prefill", "p2p", "rank2.prefill.irecv_hidden",
             elapsedSeconds(t0));

    t0 = SteadyClock::now();
    MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
    addStage(metrics.prefill, "prefill", "wait", "rank2.prefill.wait_hidden",
             elapsedSeconds(t0));

    for (int m = 0; m < times; m++) {
      t0 = SteadyClock::now();
      _mlir_ciface_forward_prefill1(&sub3DContainer, &paramsContainersRMS[m],
                                    &subResultContainer);
      addStage(metrics.prefill, "prefill", "compute",
               "rank2.prefill.forward_prefill1.rms1", elapsedSeconds(t0));
      rmsPtr = sub3DContainer.getData();

      if (comm_sub != MPI_COMM_NULL) {
        t0 = SteadyClock::now();
        MPI_Allgather(rmsPtr, subSize, MPI_FLOAT, tmp3DMemRef.getData(),
                      subSize, MPI_FLOAT, comm_sub);
        addStage(metrics.prefill, "prefill", "collective",
                 "rank2.prefill.allgather_before_mha", elapsedSeconds(t0));
      }

      t0 = SteadyClock::now();
      _mlir_ciface_forward_prefill2(kvContainerTempPtr, &paramsContainersMHA0[m],
                                    &tmp3DMemRef);
      addStage(metrics.prefill, "prefill", "compute",
               "rank2.prefill.forward_prefill2.part2", elapsedSeconds(t0));

      if (m == 0) {
        t0 = SteadyClock::now();
        MPI_Waitall(3, mha_recv_req, MPI_STATUSES_IGNORE);
        addStage(metrics.prefill, "prefill", "wait",
                 "rank2.prefill.wait_aux_triplet", elapsedSeconds(t0));
      }

      t0 = SteadyClock::now();
      _mlir_ciface_forward_prefill3(kvContainerPtr0, &paramsContainersMHA1[m],
                                    &mhaMemRef4D, &mhaMemRef3D1, &mhaMemRef3D2,
                                    &kvContainerTempPtr->qcache,
                                    &kvContainerTempPtr->kcache,
                                    &kvContainerTempPtr->vcache);
      addStage(metrics.prefill, "prefill", "compute",
               "rank2.prefill.forward_prefill3.part3", elapsedSeconds(t0));

      kv0[2 * m] = kvContainerPtr0->kcache;
      kv0[2 * m + 1] = kvContainerPtr0->vcache;
      tmp2DContainer = kvContainerPtr0->data;
      mhaOutputPtr = tmp2DContainer.getData();

      if (comm_sub != MPI_COMM_NULL) {
        t0 = SteadyClock::now();
        MPI_Reduce_scatter_block(mhaOutputPtr, sub2DPtr, subSize, MPI_FLOAT,
                                 MPI_SUM, comm_sub);
        addStage(metrics.prefill, "prefill", "collective",
                 "rank2.prefill.reduce_scatter_after_mha", elapsedSeconds(t0));
      }

      t0 = SteadyClock::now();
      _mlir_ciface_forward_prefill4(&subResultContainer, &subResultContainer,
                                    &sub2DContainer);
      addStage(metrics.prefill, "prefill", "compute",
               "rank2.prefill.forward_prefill4.add1", elapsedSeconds(t0));

      t0 = SteadyClock::now();
      _mlir_ciface_forward_prefill1(&sub3DContainer, &paramsContainersRMS0[m],
                                    &subResultContainer);
      addStage(metrics.prefill, "prefill", "compute",
               "rank2.prefill.forward_prefill1.rms2", elapsedSeconds(t0));
      rmsPtr = sub3DContainer.getData();

      if (comm_sub != MPI_COMM_NULL) {
        t0 = SteadyClock::now();
        MPI_Allgather(rmsPtr, subSize, MPI_FLOAT, tmp3DMemRef.getData(),
                      subSize, MPI_FLOAT, comm_sub);
        addStage(metrics.prefill, "prefill", "collective",
                 "rank2.prefill.allgather_before_mlp", elapsedSeconds(t0));
      }

      t0 = SteadyClock::now();
      _mlir_ciface_forward_prefill6(&tmp2DContainer, &paramsContainersMLP[m],
                                    &tmp3DMemRef);
      addStage(metrics.prefill, "prefill", "compute",
               "rank2.prefill.forward_prefill6.mlp", elapsedSeconds(t0));
      mhaOutputPtr = tmp2DContainer.getData();

      if (comm_sub != MPI_COMM_NULL) {
        t0 = SteadyClock::now();
        MPI_Reduce_scatter_block(mhaOutputPtr, sub2DPtr, subSize, MPI_FLOAT,
                                 MPI_SUM, comm_sub);
        addStage(metrics.prefill, "prefill", "collective",
                 "rank2.prefill.reduce_scatter_after_mlp", elapsedSeconds(t0));
      }

      t0 = SteadyClock::now();
      _mlir_ciface_forward_prefill4(&subResultContainer, &subResultContainer,
                                    &sub2DContainer);
      addStage(metrics.prefill, "prefill", "compute",
               "rank2.prefill.forward_prefill4.add2", elapsedSeconds(t0));

      if (m == (times - 1)) {
        subResultPtr = subResultContainer.getData();
        t0 = SteadyClock::now();
        MPI_Send(subResultPtr, subSize, MPI_FLOAT, nextRank,
                 CommTag::PrefillReturnHidden, MPI_COMM_WORLD);
        addStage(metrics.prefill, "prefill", "p2p", "rank2.prefill.send_back",
                 elapsedSeconds(t0));
      }
    }
    metrics.prefill.wall_s = elapsedSeconds(workerPrefillStart);

    // decode
    MemRef<long long, 1> cachePosition({1}, 0LL);
    MemRef<float, 3> subResultContainerDecode({1, 1, HiddenSize0});
    MemRef<float, 3> sub3DContainerDecode({1, 1, HiddenSize0});
    MemRef<int8_t, 4> mhaMemRef4DDecode({1, 1, 1, MaxTokenLength});
    MemRef<float, 3> mhaMemRef3D1Decode({1, 1, HiddenSize});
    MemRef<float, 3> mhaMemRef3D2Decode({1, 1, HiddenSize});
    MemRef<float, 2> tmp2DContainerDecode({1, HiddenSize0});
    MemRef<float, 2> sub2DContainerDecode({1, HiddenSize0});
    // Do not cache subResultContainerDecode.getData() across iterations.
    // The active MemRef descriptor may change after MLIR iface calls.
    float *mhaOutputPtrDecode = tmp2DContainerDecode.getData();

    MemRefContainer2 kvDecodeContainer0(kv0[0], kv0[1], tmp2DContainerDecode);
    MemRefContainer2 *kvDecodeContainerPtr0 = &kvDecodeContainer0;

    MemRef<float, 4> tempDecodeQ4D({1, 6, 1, HiddenSize});
    MemRef<float, 4> tempDecodeK4D({1, 1, 1, HiddenSize});
    MemRef<float, 4> tempDecodeV4D({1, 1, 1, HiddenSize});
    MemRefContainer2temp kvDecodeContainerTemp(tempDecodeQ4D, tempDecodeK4D,
                                               tempDecodeV4D);
    MemRefContainer2temp *kvDecodeContainerTempPtr = &kvDecodeContainerTemp;

    auto workerDecodeStart = SteadyClock::now();
    for (int i = 1; i <= generateLen; i++) {
      t0 = SteadyClock::now();
      MPI_Recv(reinterpret_cast<void *>(&decodePacketARecv),
               static_cast<int>(sizeof(DecodePacketA)), MPI_BYTE, source,
               CommTag::DecodePacketA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      addStage(metrics.decode, "decode", "p2p",
               "rank2.decode.recv_packet_a", elapsedSeconds(t0));
      if (decodePacketARecv.ctrl == 1) {
        break;
      }

      t0 = SteadyClock::now();
      unpackDecodePacketA(decodePacketARecv, subResultContainerDecode,
                          cachePosition);
      addStage(metrics.decode, "decode", "compute",
               "rank2.decode.unpack_packet_a", elapsedSeconds(t0));

      t0 = SteadyClock::now();
      MPI_Irecv(reinterpret_cast<void *>(&decodePacketBRecv),
                static_cast<int>(sizeof(DecodePacketB)), MPI_BYTE, source,
                CommTag::DecodePacketB, MPI_COMM_WORLD, &decode_recv_packet_b);
      addStage(metrics.decode, "decode", "p2p",
               "rank2.decode.irecv_packet_b", elapsedSeconds(t0));

      for (int m = 0; m < times; m++) {
        t0 = SteadyClock::now();
        _mlir_ciface_forward_decode1(&sub3DContainerDecode,
                                     &paramsContainersRMS[m],
                                     &subResultContainerDecode);
        addStage(metrics.decode, "decode", "compute",
                 "rank2.decode.forward_decode1.rms1", elapsedSeconds(t0));

        t0 = SteadyClock::now();
        _mlir_ciface_forward_decode2(kvDecodeContainerTempPtr,
                                     &paramsContainersMHA0[m],
                                     &sub3DContainerDecode);
        addStage(metrics.decode, "decode", "compute",
                 "rank2.decode.forward_decode2.part2", elapsedSeconds(t0));

        if (m == 0) {
          t0 = SteadyClock::now();
          MPI_Wait(&decode_recv_packet_b, MPI_STATUS_IGNORE);
          addStage(metrics.decode, "decode", "wait",
                   "rank2.decode.wait_packet_b", elapsedSeconds(t0));

          t0 = SteadyClock::now();
          unpackDecodePacketB(decodePacketBRecv, mhaMemRef4DDecode,
                              mhaMemRef3D1Decode, mhaMemRef3D2Decode);
          addStage(metrics.decode, "decode", "compute",
                   "rank2.decode.unpack_packet_b", elapsedSeconds(t0));
        }

        t0 = SteadyClock::now();
        _mlir_ciface_forward_decode3(
            kvDecodeContainerPtr0, &paramsContainersMHA1[m], &cachePosition,
            &kv0[2 * m], &kv0[2 * m + 1], &mhaMemRef4DDecode,
            &mhaMemRef3D1Decode, &mhaMemRef3D2Decode,
            &kvDecodeContainerTempPtr->qcache,
            &kvDecodeContainerTempPtr->kcache,
            &kvDecodeContainerTempPtr->vcache);
        addStage(metrics.decode, "decode", "compute",
                 "rank2.decode.forward_decode3.part3", elapsedSeconds(t0));

        kv0[2 * m] = kvDecodeContainerPtr0->kcache;
        kv0[2 * m + 1] = kvDecodeContainerPtr0->vcache;
        tmp2DContainerDecode = kvDecodeContainerPtr0->data;
        mhaOutputPtrDecode = tmp2DContainerDecode.getData();
        if (comm_sub != MPI_COMM_NULL) {
          t0 = SteadyClock::now();
          MPI_Allreduce(mhaOutputPtrDecode, sub2DContainerDecode.getData(),
                        HiddenSize0, MPI_FLOAT, MPI_SUM, comm_sub);
          addStage(metrics.decode, "decode", "collective",
                   "rank2.decode.allreduce_after_mha", elapsedSeconds(t0));
        }

        t0 = SteadyClock::now();
        _mlir_ciface_forward_decode4(&subResultContainerDecode,
                                     &subResultContainerDecode,
                                     &sub2DContainerDecode);
        addStage(metrics.decode, "decode", "compute",
                 "rank2.decode.forward_decode4.add1", elapsedSeconds(t0));

        t0 = SteadyClock::now();
        _mlir_ciface_forward_decode1(&sub3DContainerDecode,
                                     &paramsContainersRMS0[m],
                                     &subResultContainerDecode);
        addStage(metrics.decode, "decode", "compute",
                 "rank2.decode.forward_decode1.rms2", elapsedSeconds(t0));

        t0 = SteadyClock::now();
        _mlir_ciface_forward_decode6(&tmp2DContainerDecode,
                                     &paramsContainersMLP[m],
                                     &sub3DContainerDecode);
        addStage(metrics.decode, "decode", "compute",
                 "rank2.decode.forward_decode6.mlp", elapsedSeconds(t0));
        mhaOutputPtrDecode = tmp2DContainerDecode.getData();
        if (comm_sub != MPI_COMM_NULL) {
          t0 = SteadyClock::now();
          MPI_Allreduce(mhaOutputPtrDecode, sub2DContainerDecode.getData(),
                        HiddenSize0, MPI_FLOAT, MPI_SUM, comm_sub);
          addStage(metrics.decode, "decode", "collective",
                   "rank2.decode.allreduce_after_mlp", elapsedSeconds(t0));
        }

        t0 = SteadyClock::now();
        _mlir_ciface_forward_decode4(&subResultContainerDecode,
                                     &subResultContainerDecode,
                                     &sub2DContainerDecode);
        addStage(metrics.decode, "decode", "compute",
                 "rank2.decode.forward_decode4.add2", elapsedSeconds(t0));

        if (EnableWorkerDebugLogs) {
          auto tdbg = SteadyClock::now();
          std::cout << "completed " << m << std::endl;
          addStage(metrics.decode, "decode", "log",
                   "rank2.decode.debug_completed", elapsedSeconds(tdbg));
        }
      }
    }
    metrics.decode.wall_s = elapsedSeconds(workerDecodeStart);

    if (comm_sub != MPI_COMM_NULL) {
      MPI_Comm_free(&comm_sub);
    }
    if (sub_group != MPI_GROUP_NULL) {
      MPI_Group_free(&sub_group);
    }
    if (world_group != MPI_GROUP_NULL) {
      MPI_Group_free(&world_group);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  printRankSummary(metrics);
  writeSummaryCsv(metrics);
  writeStagesCsv(metrics);

  MPI_Finalize();

  return 0;
}
