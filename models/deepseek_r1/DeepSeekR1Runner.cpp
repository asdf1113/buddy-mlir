//===- DeepSeekR1Runner.cpp - DeepSeekR1 full inference loop --------------===//
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
//
// Implements DeepSeekR1Runner::run().  This file owns session setup and
// dispatch:
//   - weight file loading
//   - chat template loading
//   - sampler creation
//   - dispatch to single-shot generation or interactive REPL
//
// buddy-cli calls this through the generic InferenceRunner interface.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/DeepSeekR1Runner.h"
#include "buddy/LLM/ChatTemplate.h"
#include "buddy/LLM/ConversationManager.h"
#include "buddy/LLM/TextContainer.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/llm/InteractiveSession.h"
#include "buddy/runtime/llm/TextGeneration.h"
#include "buddy/runtime/models/ModelSession.h"

#ifdef BUDDY_RUNTIME_ENABLE_MPI
#include "buddy/runtime/models/DeepSeekR1RaxRunner.h"
#include "buddy/runtime/models/DeepSeekR1RaxSession.h"
#endif

#include "buddy/Core/Container.h"

using buddy::Text;

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace buddy {
namespace runtime {

namespace {
static constexpr int kEosToken = 151643; // <|end▁of▁sentence|>
static constexpr int kEotToken = 151647; // <|EOT|>
} // namespace

//===----------------------------------------------------------------------===//
// DeepSeekR1Runner::run
//===----------------------------------------------------------------------===//

void DeepSeekR1Runner::run(const RunConfig &cfgIn) {
  RunConfig cfg = cfgIn;
  const bool tensorParallel = cfg.tensorParallelSize > 1;

  if (tensorParallel) {
    if (cfg.raxPath.empty())
      throw std::runtime_error(
          "DeepSeek tensor-parallel execution requires --model <rank0.rax>");
#ifndef BUDDY_RUNTIME_ENABLE_MPI
    throw std::runtime_error(
        "DeepSeek tensor-parallel execution requires a build configured with "
        "BUDDY_RUNTIME_ENABLE_MPI=ON");
#endif
    const auto &sampler = cfg.samplerConfig;
    if (sampler.temperature != 0.0f || sampler.topK != 0 ||
        sampler.topP != 1.0f || sampler.minP != 0.0f ||
        sampler.repeatPenalty != 1.0f)
      throw std::runtime_error(
          "DeepSeek tensor-parallel execution currently requires deterministic "
          "greedy sampling");
    if (cfg.interactive)
      throw std::runtime_error(
          "DeepSeek tensor-parallel interactive mode is not supported");
  }
  const bool singleProcessSuppress = cfg.suppressStats || cfg.streamJsonl;
  if (!tensorParallel && !singleProcessSuppress)
    std::cerr
        << "\033[33;1mDeepSeekR1 Inference (buddy-cli / BuddyRuntime)\033[0m\n";

  // ── Chat template: load if provided ─────────────────────────────────────
  std::vector<long long> stopTokenIds = {kEosToken, kEotToken};
  std::unique_ptr<buddy::ChatTemplate> chatTmpl;

  if (!cfg.chatTemplatePath.empty()) {
    chatTmpl = std::make_unique<buddy::ChatTemplate>(
        buddy::ChatTemplate::fromFile(cfg.chatTemplatePath));
    for (int id : chatTmpl->stopTokenIds()) {
      if (std::find(stopTokenIds.begin(), stopTokenIds.end(), id) ==
          stopTokenIds.end()) {
        stopTokenIds.push_back(static_cast<long long>(id));
      }
    }
    if (!tensorParallel)
      printLog("Chat template loaded: " + cfg.chatTemplatePath,
               singleProcessSuppress);
  }

  // ── Create session ───────────────────────────────────────────────────────
  std::unique_ptr<LLMSession> session;
  std::vector<std::string> weightPaths;
  std::string vocabPath;
  ModelManifest manifest;

  if (!tensorParallel && !cfg.raxPath.empty()) {
    printLog("Manifest: " + cfg.raxPath, singleProcessSuppress);
    session = ModelSession::createFromRax(cfg.raxPath, manifest);

    weightPaths = manifest.weightPaths;
    vocabPath = manifest.vocabPath.empty()
                    ? (std::filesystem::path(manifest.soPath).parent_path() /
                       "vocab.txt")
                          .string()
                    : manifest.vocabPath;

    printLog("  .so     = " + manifest.soPath, singleProcessSuppress);
    for (const auto &path : weightPaths)
      printLog("  weights = " + path, singleProcessSuppress);
    printLog("  vocab   = " + vocabPath, singleProcessSuppress);
  } else if (!tensorParallel) {
    if (cfg.modelSoPath.empty())
      throw std::runtime_error("Mode B requires modelSoPath (--model-so).");
    if (cfg.weightsPath.empty())
      throw std::runtime_error("Mode B requires weightsPath (--weights).");

    weightPaths.push_back(cfg.weightsPath);
    vocabPath = cfg.vocabPath.empty()
                    ? (std::filesystem::path(cfg.modelSoPath).parent_path() /
                       "vocab.txt")
                          .string()
                    : cfg.vocabPath;

    ModelSession::Config mcfg;
    mcfg.modelSoPath = cfg.modelSoPath;
    printLog("Loading model: " + cfg.modelSoPath, singleProcessSuppress);
    session = ModelSession::create(mcfg);
  }

  // ── Model-specific text codec ───────────────────────────────────────────
  TextCodec codec;
  codec.tokenize = [](Text<size_t, 2> &t, const std::string &vocab) {
    t.tokenizeDeepSeekR1(vocab, BUDDY_DSR1_MAX_TOKEN_LEN);
  };
  codec.detokenize = [](Text<size_t, 2> &t) { return t.revertDeepSeekR1(); };
  codec.maxTokenLen = BUDDY_DSR1_MAX_TOKEN_LEN;

  // ── Create Sampler ───────────────────────────────────────────────────────
  buddy::Sampler sampler(cfg.samplerConfig);

  auto generate = [&](LLMSession &activeSession,
                      const std::vector<std::string> &activeWeightPaths,
                      const std::string &activeVocabPath,
                      const ModelManifest *activeManifest, bool emitOutput) {
    const bool suppress = cfg.suppressStats || cfg.streamJsonl || !emitOutput;
    if (tensorParallel && !suppress)
      std::cerr << "\033[33;1mDeepSeekR1 Inference (buddy-cli / "
                   "BuddyRuntime)\033[0m\n";
    if (tensorParallel && !cfg.chatTemplatePath.empty())
      printLog("Chat template loaded: " + cfg.chatTemplatePath, suppress);
    if (tensorParallel && activeManifest) {
      printLog("Manifest: " + cfg.raxPath, suppress);
      printLog("  .so     = " + activeManifest->soPath, suppress);
      for (const auto &path : activeWeightPaths)
        printLog("  weights = " + path, suppress);
      printLog("  vocab   = " + activeVocabPath, suppress);
    }

    activeSession.loadWeights(activeWeightPaths);
    printLog("Weights loaded.", suppress);
    printLog("Vocab: " + activeVocabPath, suppress);
    printLog("KV cache: " + std::to_string(BUDDY_DSR1_KV_LAYERS) + " x {1," +
                 std::to_string(BUDDY_DSR1_HEAD_NUM) + "," +
                 std::to_string(BUDDY_DSR1_MAX_TOKEN_LEN) + "," +
                 std::to_string(BUDDY_DSR1_HIDDEN_SIZE) + "} f32",
             suppress);

    if (cfg.interactive) {
      if (!chatTmpl)
        throw std::runtime_error(
            "--interactive requires --chat-template <path.json>");
      buddy::ConversationManager conv(
          std::move(*chatTmpl),
          [&activeVocabPath](const std::string &text) -> size_t {
            Text<size_t, 2> tmp(text);
            tmp.tokenizeDeepSeekR1(activeVocabPath, BUDDY_DSR1_MAX_TOKEN_LEN);
            return tmp.getTokenCnt();
          });
      if (!cfg.prompt.empty())
        conv.setSystemPrompt(cfg.prompt);
      runInteractiveSession(activeSession, activeVocabPath, cfg, stopTokenIds,
                            conv, codec, sampler);
      return;
    }

    std::string finalPrompt = cfg.prompt;
    if (chatTmpl) {
      std::vector<buddy::Message> msgs = {{"user", cfg.prompt}};
      finalPrompt = chatTmpl->apply(msgs);
    }
    GenerationResult result = runGeneration(
        finalPrompt, activeSession, activeVocabPath, cfg.maxNewTokens,
        stopTokenIds, sampler, codec, suppress, cfg.streamJsonl, emitOutput);
    if (!suppress)
      printStats(result, /*verbose=*/true);
  };

#ifdef BUDDY_RUNTIME_ENABLE_MPI
  if (tensorParallel) {
    runDeepSeekR1Rax(
        cfg.raxPath, cfg.tensorParallelSize,
        [&](DeepSeekR1RaxSession &raxSession, int rank) {
          const ModelManifest &raxManifest = raxSession.manifest();
          const std::string rankVocabPath =
              raxManifest.vocabPath.empty()
                  ? (std::filesystem::path(raxManifest.soPath).parent_path() /
                     "vocab.txt")
                        .string()
                  : raxManifest.vocabPath;
          generate(raxSession, raxManifest.weightPaths, rankVocabPath,
                   &raxManifest, rank == 0);
        });
    return;
  }
#endif

  generate(*session, weightPaths, vocabPath,
           cfg.raxPath.empty() ? nullptr : &manifest, true);
}

} // namespace runtime
} // namespace buddy
