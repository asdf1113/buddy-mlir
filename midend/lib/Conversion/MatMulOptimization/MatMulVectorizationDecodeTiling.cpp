//===- MatMulVectorizationDecodeTiling.cpp --------------------------------===//
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
// Decode-specialized matmul optimization for row-vector x matrix:
//   [1 x K] * [K x N] -> [1 x N]
//
// Key properties:
//  1. only matches M == 1, otherwise fail and let other passes handle.
//  2. keep AffineParallelOp on column tiles.
//  3. internal loops rewritten to SCF nesting.
//  4. keep K blocking.
//  5. keep load-C -> accumulate in registers -> single store.
//  6. load A[0, k] once, broadcast once, feed multiple output vectors.
//  7. preserve B access pattern: B[k, j:j+VL] contiguous vector loads.
//  8. prefetch next B[k + delta, j:j+...].
//  9. tail handled once at the end, not in the hot path.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

class MatMulVectorizationDecodeTilingPattern : public ConversionPattern {
public:
  explicit MatMulVectorizationDecodeTilingPattern(
      MLIRContext *context, int64_t vectorSizeParam, int64_t kBlockSizeParam,
      int64_t colUnrollFactorParam, int64_t prefetchDistanceParam)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context),
        vectorSize(vectorSizeParam), kBlockSize(kBlockSizeParam),
        colUnrollFactor(colUnrollFactorParam),
        prefetchDistance(prefetchDistanceParam) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto matmul = dyn_cast<linalg::MatmulOp>(op);
    if (!matmul)
      return failure();

    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    auto aType = dyn_cast<MemRefType>(A.getType());
    auto bType = dyn_cast<MemRefType>(B.getType());
    auto cType = dyn_cast<MemRefType>(C.getType());
    if (!aType || !bType || !cType)
      return failure();

    // Only match row-vector x matrix:
    // A: [1 x K], B: [K x N], C: [1 x N]
    if (aType.getRank() != 2 || bType.getRank() != 2 || cType.getRank() != 2)
      return failure();
    if (aType.getDimSize(0) != 1 || cType.getDimSize(0) != 1)
      return failure();

    Type elementType = aType.getElementType();
    if (elementType != bType.getElementType() ||
        elementType != cType.getElementType())
      return failure();

    const int64_t colTile = colUnrollFactor * vectorSize;

    // ------------------------------------------------------------------
    // constants
    // ------------------------------------------------------------------
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cVec = rewriter.create<arith::ConstantIndexOp>(loc, vectorSize);
    Value cColTile = rewriter.create<arith::ConstantIndexOp>(loc, colTile);
    Value cKBlock = rewriter.create<arith::ConstantIndexOp>(loc, kBlockSize);
    Value cPrefetchDist =
        rewriter.create<arith::ConstantIndexOp>(loc, prefetchDistance);

    Value zeroScalar =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementType));
    VectorType vecTy = VectorType::get({vectorSize}, elementType);
    Value zeroVec = rewriter.create<vector::SplatOp>(loc, vecTy, zeroScalar);

    // ------------------------------------------------------------------
    // dims
    // ------------------------------------------------------------------
    Value K = rewriter.create<memref::DimOp>(loc, B, c0);
    Value N = rewriter.create<memref::DimOp>(loc, B, c1);

    // fullTileUpper = N - colTile + 1
    Value fullTileUpper = rewriter.create<affine::AffineApplyOp>(
        loc,
        AffineMap::get(1, 0,
                       {rewriter.getAffineDimExpr(0) - colTile + 1}),
        ValueRange{N});

    // ------------------------------------------------------------------
    // parallel over full multi-vector column tiles
    // ------------------------------------------------------------------
    affine::AffineParallelOp parallelLoop =
        rewriter.create<affine::AffineParallelOp>(
            loc, TypeRange{}, ValueRange{fullTileUpper},
            ArrayRef<NamedAttribute>{
                rewriter.getNamedAttr(
                    "lowerBoundsGroups", rewriter.getI32TensorAttr({1})),
                rewriter.getNamedAttr(
                    "upperBoundsGroups", rewriter.getI32TensorAttr({1})),
                rewriter.getNamedAttr(
                    "lowerBoundsMap",
                    AffineMapAttr::get(
                        AffineMap::get(0, 0, {rewriter.getAffineConstantExpr(0)},
                                       rewriter.getContext()))),
                rewriter.getNamedAttr(
                    "upperBoundsMap",
                    AffineMapAttr::get(
                        AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)},
                                       rewriter.getContext()))),
                rewriter.getNamedAttr("reductions", rewriter.getArrayAttr({})),
                rewriter.getNamedAttr("steps",
                                      rewriter.getI64ArrayAttr({colTile}))});

    Block *parallelBody = new Block();
    parallelBody->addArgument(rewriter.getIndexType(), loc);
    rewriter.setInsertionPointToStart(parallelBody);

    Value jBase = parallelBody->getArgument(0);

    // Conservative version: load C first, then accumulate in registers.
    SmallVector<Value, 4> initAccs;
    initAccs.reserve(colUnrollFactor);
    for (int64_t u = 0; u < colUnrollFactor; ++u) {
      Value uIdx = rewriter.create<arith::ConstantIndexOp>(loc, u * vectorSize);
      Value jU = rewriter.create<arith::AddIOp>(loc, jBase, uIdx);
      Value acc =
          rewriter.create<vector::LoadOp>(loc, vecTy, C, ValueRange{c0, jU});
      initAccs.push_back(acc);
    }

    auto kbLoop = rewriter.create<scf::ForOp>(
        loc, c0, K, cKBlock, ValueRange(initAccs),
        [&](OpBuilder &builder, Location loc, Value kb,
            ValueRange kbIterArgs) {
          Value kHigh = builder.create<arith::AddIOp>(loc, kb, cKBlock);
          Value kHighMin = builder.create<arith::MinUIOp>(loc, kHigh, K);

          auto kLoop = builder.create<scf::ForOp>(
              loc, kb, kHighMin, c1, kbIterArgs,
              [&](OpBuilder &builder2, Location loc, Value k,
                  ValueRange kIterArgs) {
                // Load A[0, k] once.
                Value aScalar =
                    builder2.create<memref::LoadOp>(loc, A, ValueRange{c0, k});
                Value aVec =
                    builder2.create<vector::BroadcastOp>(loc, vecTy, aScalar);

                // Prefetch B[k + delta, jBase + u*VL]
                Value kPf = builder2.create<arith::AddIOp>(loc, k, cPrefetchDist);
                Value inRange = builder2.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::ult, kPf, K);

                auto ifPrefetch = builder2.create<scf::IfOp>(
                    loc, TypeRange{}, inRange, /*withElseRegion=*/false);
                {
                  OpBuilder thenBuilder = ifPrefetch.getThenBodyBuilder();
                  for (int64_t u = 0; u < colUnrollFactor; ++u) {
                    Value uIdx = thenBuilder.create<arith::ConstantIndexOp>(
                        loc, u * vectorSize);
                    Value jPf =
                        thenBuilder.create<arith::AddIOp>(loc, jBase, uIdx);

                    thenBuilder.create<affine::AffinePrefetchOp>(
                        loc, B, ValueRange{kPf, jPf},
                        /*isWrite=*/false,
                        /*localityHint=*/3,
                        /*isDataCache=*/true,
                        AffineMap::get(2, 0,
                                       {rewriter.getAffineDimExpr(0),
                                        rewriter.getAffineDimExpr(1)},
                                       rewriter.getContext()));
                  }
                  thenBuilder.create<scf::YieldOp>(loc);
                }

                SmallVector<Value, 4> nextAccs;
                nextAccs.reserve(colUnrollFactor);

                for (int64_t u = 0; u < colUnrollFactor; ++u) {
                  Value uIdx = builder2.create<arith::ConstantIndexOp>(
                      loc, u * vectorSize);
                  Value jU = builder2.create<arith::AddIOp>(loc, jBase, uIdx);

                  // Keep B access pattern unchanged.
                  Value bVec = builder2.create<vector::LoadOp>(
                      loc, vecTy, B, ValueRange{k, jU});

                  Value updated;
                  if (isa<IntegerType>(elementType)) {
                    Value mul =
                        builder2.create<arith::MulIOp>(loc, aVec, bVec);
                    updated = builder2.create<arith::AddIOp>(
                        loc, mul, kIterArgs[u]);
                  } else {
                    updated = builder2.create<vector::FMAOp>(
                        loc, aVec, bVec, kIterArgs[u]);
                  }
                  nextAccs.push_back(updated);
                }

                builder2.create<scf::YieldOp>(loc, nextAccs);
              });

          builder.create<scf::YieldOp>(loc, kLoop.getResults());
        });

    for (int64_t u = 0; u < colUnrollFactor; ++u) {
      Value uIdx = rewriter.create<arith::ConstantIndexOp>(loc, u * vectorSize);
      Value jU = rewriter.create<arith::AddIOp>(loc, jBase, uIdx);
      rewriter.create<vector::StoreOp>(loc, kbLoop.getResult(u), C,
                                       ValueRange{c0, jU});
    }

    rewriter.create<affine::AffineYieldOp>(loc);
    parallelLoop.getRegion().push_back(parallelBody);

    // ------------------------------------------------------------------
    // remaining full single-vector tiles
    // ------------------------------------------------------------------
    rewriter.setInsertionPointAfter(parallelLoop);

    Value nDivColTile = rewriter.create<arith::DivUIOp>(loc, N, cColTile);
    Value remStart = rewriter.create<arith::MulIOp>(loc, nDivColTile, cColTile);

    Value nModVec = rewriter.create<arith::RemUIOp>(loc, N, cVec);
    Value fullVecEnd = rewriter.create<arith::SubIOp>(loc, N, nModVec);

    auto remVecLoop = rewriter.create<scf::ForOp>(
        loc, remStart, fullVecEnd, cVec, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value j,
            ValueRange /*iterArgs*/) {
          Value acc =
              builder.create<vector::LoadOp>(loc, vecTy, C, ValueRange{c0, j});

          auto kbLoop2 = builder.create<scf::ForOp>(
              loc, c0, K, cKBlock, ValueRange{acc},
              [&](OpBuilder &builder2, Location loc, Value kb,
                  ValueRange kbIterArgs) {
                Value kHigh = builder2.create<arith::AddIOp>(loc, kb, cKBlock);
                Value kHighMin = builder2.create<arith::MinUIOp>(loc, kHigh, K);

                auto kLoop2 = builder2.create<scf::ForOp>(
                    loc, kb, kHighMin, c1, kbIterArgs,
                    [&](OpBuilder &builder3, Location loc, Value k,
                        ValueRange kIterArgs) {
                      Value aScalar = builder3.create<memref::LoadOp>(
                          loc, A, ValueRange{c0, k});
                      Value aVec =
                          builder3.create<vector::BroadcastOp>(loc, vecTy, aScalar);

                      Value kPf =
                          builder3.create<arith::AddIOp>(loc, k, cPrefetchDist);
                      Value inRange = builder3.create<arith::CmpIOp>(
                          loc, arith::CmpIPredicate::ult, kPf, K);

                      auto ifPrefetch = builder3.create<scf::IfOp>(
                          loc, TypeRange{}, inRange, false);
                      {
                        OpBuilder thenBuilder = ifPrefetch.getThenBodyBuilder();
                        thenBuilder.create<affine::AffinePrefetchOp>(
                            loc, B, ValueRange{kPf, j},
                            /*isWrite=*/false,
                            /*localityHint=*/3,
                            /*isDataCache=*/true,
                            AffineMap::get(2, 0,
                                           {rewriter.getAffineDimExpr(0),
                                            rewriter.getAffineDimExpr(1)},
                                           rewriter.getContext()));
                        thenBuilder.create<scf::YieldOp>(loc);
                      }

                      Value bVec = builder3.create<vector::LoadOp>(
                          loc, vecTy, B, ValueRange{k, j});

                      Value updated;
                      if (isa<IntegerType>(elementType)) {
                        Value mul =
                            builder3.create<arith::MulIOp>(loc, aVec, bVec);
                        updated = builder3.create<arith::AddIOp>(
                            loc, mul, kIterArgs[0]);
                      } else {
                        updated = builder3.create<vector::FMAOp>(
                            loc, aVec, bVec, kIterArgs[0]);
                      }
                      builder3.create<scf::YieldOp>(loc, updated);
                    });

                builder2.create<scf::YieldOp>(loc, kLoop2.getResult(0));
              });

          builder.create<vector::StoreOp>(loc, kbLoop2.getResult(0), C,
                                          ValueRange{c0, j});
          builder.create<scf::YieldOp>(loc);
        });

    (void)remVecLoop;

    // ------------------------------------------------------------------
    // final masked tail
    // ------------------------------------------------------------------
    Value hasTail = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, nModVec, c0);

    auto tailIf =
        rewriter.create<scf::IfOp>(loc, TypeRange{}, hasTail, false);
    {
      OpBuilder tailBuilder = tailIf.getThenBodyBuilder();

      Value tailStart = fullVecEnd;
      Value tailSize = nModVec;
      VectorType maskTy = VectorType::get({vectorSize}, rewriter.getI1Type());
      Value mask = tailBuilder.create<vector::CreateMaskOp>(
          loc, maskTy, ValueRange{tailSize});

      Value acc = tailBuilder.create<vector::MaskedLoadOp>(
          loc, vecTy, C, ValueRange{c0, tailStart}, mask, zeroVec);

      auto kbLoop3 = tailBuilder.create<scf::ForOp>(
          loc, c0, K, cKBlock, ValueRange{acc},
          [&](OpBuilder &builder2, Location loc, Value kb,
              ValueRange kbIterArgs) {
            Value kHigh = builder2.create<arith::AddIOp>(loc, kb, cKBlock);
            Value kHighMin = builder2.create<arith::MinUIOp>(loc, kHigh, K);

            auto kLoop3 = builder2.create<scf::ForOp>(
                loc, kb, kHighMin, c1, kbIterArgs,
                [&](OpBuilder &builder3, Location loc, Value k,
                    ValueRange kIterArgs) {
                  Value aScalar =
                      builder3.create<memref::LoadOp>(loc, A, ValueRange{c0, k});
                  Value aVec =
                      builder3.create<vector::BroadcastOp>(loc, vecTy, aScalar);

                  Value kPf =
                      builder3.create<arith::AddIOp>(loc, k, cPrefetchDist);
                  Value inRange = builder3.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::ult, kPf, K);

                  auto ifPrefetch = builder3.create<scf::IfOp>(
                      loc, TypeRange{}, inRange, false);
                  {
                    OpBuilder thenBuilder = ifPrefetch.getThenBodyBuilder();
                    thenBuilder.create<affine::AffinePrefetchOp>(
                        loc, B, ValueRange{kPf, tailStart},
                        /*isWrite=*/false,
                        /*localityHint=*/3,
                        /*isDataCache=*/true,
                        AffineMap::get(2, 0,
                                       {rewriter.getAffineDimExpr(0),
                                        rewriter.getAffineDimExpr(1)},
                                       rewriter.getContext()));
                    thenBuilder.create<scf::YieldOp>(loc);
                  }

                  Value bVec = builder3.create<vector::MaskedLoadOp>(
                      loc, vecTy, B, ValueRange{k, tailStart}, mask, zeroVec);

                  Value updated;
                  if (isa<IntegerType>(elementType)) {
                    Value mul =
                        builder3.create<arith::MulIOp>(loc, aVec, bVec);
                    updated = builder3.create<arith::AddIOp>(
                        loc, mul, kIterArgs[0]);
                  } else {
                    updated = builder3.create<vector::FMAOp>(
                        loc, aVec, bVec, kIterArgs[0]);
                  }

                  builder3.create<scf::YieldOp>(loc, updated);
                });

            builder2.create<scf::YieldOp>(loc, kLoop3.getResult(0));
          });

      tailBuilder.create<vector::MaskedStoreOp>(
          loc, C, ValueRange{c0, tailStart}, mask, kbLoop3.getResult(0));
      tailBuilder.create<scf::YieldOp>(loc);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vectorSize;
  int64_t kBlockSize;
  int64_t colUnrollFactor;
  int64_t prefetchDistance;
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {

class MatMulVectorizationDecodeTilingPass
    : public PassWrapper<MatMulVectorizationDecodeTilingPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MatMulVectorizationDecodeTilingPass)

  StringRef getArgument() const final {
    return "matmul-vectorization-decode-tiling";
  }

  StringRef getDescription() const final {
    return "Decode-specialized row-vector x matrix tiling/vectorization optimization.";
  }

  MatMulVectorizationDecodeTilingPass() = default;

  // 必须手写 copy ctor，不能让编译器默认拷 Option<> 成员
  MatMulVectorizationDecodeTilingPass(
      const MatMulVectorizationDecodeTilingPass &other)
      : PassWrapper(other) {
    vectorSize = other.vectorSize;
    kBlockSize = other.kBlockSize;
    colUnrollFactor = other.colUnrollFactor;
    prefetchDistance = other.prefetchDistance;
  }

  explicit MatMulVectorizationDecodeTilingPass(
      int64_t vectorSizeParam, int64_t kBlockSizeParam,
      int64_t colUnrollFactorParam, int64_t prefetchDistanceParam) {
    vectorSize = vectorSizeParam;
    kBlockSize = kBlockSizeParam;
    colUnrollFactor = colUnrollFactorParam;
    prefetchDistance = prefetchDistanceParam;
  }
  void runOnOperation() override {
    llvm::errs() << "[DecodeTilingPass] runOnOperation entered\n";
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                           scf::SCFDialect, memref::MemRefDialect,
                           vector::VectorDialect, func::FuncDialect,
                           linalg::LinalgDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
    target.addLegalOp<linalg::FillOp>();

    RewritePatternSet patterns(context);
    patterns.add<MatMulVectorizationDecodeTilingPattern>(
        context, vectorSize, kBlockSize, colUnrollFactor, prefetchDistance);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, memref::MemRefDialect,
                    vector::VectorDialect, func::FuncDialect,
                    arith::ArithDialect>();
  }

  Option<int64_t> vectorSize{
      *this, "vector-size",
      llvm::cl::desc("Vector size for decode-specialized matmul."),
      llvm::cl::init(64)};

  Option<int64_t> kBlockSize{
      *this, "k-block-size",
      llvm::cl::desc("K blocking size for decode-specialized matmul."),
      llvm::cl::init(32)};

  Option<int64_t> colUnrollFactor{
      *this, "col-unroll-factor",
      llvm::cl::desc("Number of output vectors updated by one A[0,k] load."),
      llvm::cl::init(2)};

  Option<int64_t> prefetchDistance{
      *this, "prefetch-distance",
      llvm::cl::desc("Prefetch distance on K dimension for B."),
      llvm::cl::init(1)};
};

} // namespace

namespace mlir {
namespace buddy {

void registerMatMulVectorizationDecodeTilingPass() {
  PassRegistration<MatMulVectorizationDecodeTilingPass>();
}

} // namespace buddy
} // namespace mlir