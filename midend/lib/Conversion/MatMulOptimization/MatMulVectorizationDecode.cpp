//===- MatMulVectorizationDecode.cpp --------------------------------------===//
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
// This file implements the matmul-vectorization-decode optimization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class MatMulVectorizationDecodePattern : public ConversionPattern {
public:
  explicit MatMulVectorizationDecodePattern(MLIRContext *context,
                                            int64_t vectorSizeParam,
                                            int64_t unrollFactorParam)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context) {
    vectorSize = vectorSizeParam;
    unrollFactor = unrollFactorParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    auto aType = dyn_cast<MemRefType>(A.getType());
    auto bType = dyn_cast<MemRefType>(B.getType());
    auto cType = dyn_cast<MemRefType>(C.getType());
    if (!aType || !bType || !cType)
      return failure();

    if (aType.getRank() != 2 || bType.getRank() != 2 || cType.getRank() != 2)
      return failure();

    Type elementType = aType.getElementType();
    if (elementType != bType.getElementType() ||
        elementType != cType.getElementType())
      return failure();

    // Decode-specialized path only handles static M = 1.
    if (aType.isDynamicDim(0) || aType.getDimSize(0) != 1)
      return failure();

    const Value zeroIndex =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value oneIndex =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    const Value vecSizeValue =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(vectorSize));
    const Value tileSizeValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(vectorSize * unrollFactor));

    const auto vecType = VectorType::get({vectorSize}, elementType);
    const Value zeroElement =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementType));
    const Value zeroVector =
        rewriter.create<vector::SplatOp>(loc, vecType, zeroElement);

    // For M = 1:
    // A: [1, K], B: [K, N], C: [1, N]
    Value kDim = rewriter.create<memref::DimOp>(loc, B, zeroIndex);
    Value nDim = rewriter.create<memref::DimOp>(loc, B, oneIndex);

    emitDecodeKernel(rewriter, loc, A, B, C, elementType, zeroIndex, oneIndex,
                     vecSizeValue, tileSizeValue, zeroVector, kDim, nDim);

    rewriter.eraseOp(op);
    return success();
  }

private:
  Value createMulAdd(OpBuilder &builder, Location loc, Type elementType,
                     Value lhs, Value rhs, Value acc) const {
    if (isa<IntegerType>(elementType)) {
      Value mul = builder.create<arith::MulIOp>(loc, lhs, rhs);
      return builder.create<arith::AddIOp>(loc, mul, acc);
    }
    return builder.create<vector::FMAOp>(loc, lhs, rhs, acc);
  }

  void emitDecodeKernel(OpBuilder &builder, Location loc, Value A, Value B,
                        Value C, Type elementType, Value zeroIndex,
                        Value oneIndex, Value vecSizeValue, Value tileSizeValue,
                        Value zeroVector, Value kDim, Value nDim) const {
    const auto vecType = VectorType::get({vectorSize}, elementType);

    // Nmain = floor(N / (vectorSize * unrollFactor)) * (vectorSize * unrollFactor)
    Value mainBlockCount =
        builder.create<arith::DivUIOp>(loc, nDim, tileSizeValue);
    Value nMain =
        builder.create<arith::MulIOp>(loc, mainBlockCount, tileSizeValue);

    // Main loop: process `unrollFactor` x vectorSize columns each time.
    builder.create<scf::ForOp>(
        loc, zeroIndex, nMain, tileSizeValue, ValueRange{},
        [&](OpBuilder &b, Location loc, Value nBase, ValueRange /*args*/) {
          SmallVector<Value> colBases;
          SmallVector<Value> initAccs;
          colBases.reserve(unrollFactor);
          initAccs.reserve(unrollFactor);

          for (int64_t i = 0; i < unrollFactor; ++i) {
            Value offset =
                b.create<arith::ConstantOp>(loc, b.getIndexAttr(i * vectorSize));
            Value colBase =
                (i == 0) ? nBase : b.create<arith::AddIOp>(loc, nBase, offset);
            colBases.push_back(colBase);

            Value accInit = b.create<vector::LoadOp>(
                loc, vecType, C, ValueRange{zeroIndex, colBase});
            initAccs.push_back(accInit);
          }

          auto kLoop = b.create<scf::ForOp>(
              loc, zeroIndex, kDim, oneIndex, initAccs,
              [&](OpBuilder &kb, Location loc, Value k, ValueRange iterArgs) {
                Value a = kb.create<memref::LoadOp>(loc, A,
                                                    ValueRange{zeroIndex, k});
                Value va = kb.create<vector::BroadcastOp>(loc, vecType, a);

                SmallVector<Value> nextAccs;
                nextAccs.reserve(unrollFactor);

                for (int64_t i = 0; i < unrollFactor; ++i) {
                  Value bVec = kb.create<vector::LoadOp>(
                      loc, vecType, B, ValueRange{k, colBases[i]});
                  Value out = createMulAdd(kb, loc, elementType, va, bVec,
                                           iterArgs[i]);
                  nextAccs.push_back(out);
                }

                kb.create<scf::YieldOp>(loc, nextAccs);
              });

          for (int64_t i = 0; i < unrollFactor; ++i) {
            b.create<vector::StoreOp>(loc, kLoop.getResult(i), C,
                                      ValueRange{zeroIndex, colBases[i]});
          }

          b.create<scf::YieldOp>(loc);
        });

    // Nvec = floor(N / vectorSize) * vectorSize
    Value vecBlockCount =
        builder.create<arith::DivUIOp>(loc, nDim, vecSizeValue);
    Value nVec = builder.create<arith::MulIOp>(loc, vecBlockCount, vecSizeValue);

    // Handle remaining full vector blocks in [nMain, nVec).
    builder.create<scf::ForOp>(
        loc, nMain, nVec, vecSizeValue, ValueRange{},
        [&](OpBuilder &b, Location loc, Value nBase, ValueRange /*args*/) {
          Value initAcc = b.create<vector::LoadOp>(
              loc, vecType, C, ValueRange{zeroIndex, nBase});

          auto kLoop = b.create<scf::ForOp>(
              loc, zeroIndex, kDim, oneIndex, ValueRange{initAcc},
              [&](OpBuilder &kb, Location loc, Value k, ValueRange iterArgs) {
                Value a = kb.create<memref::LoadOp>(loc, A,
                                                    ValueRange{zeroIndex, k});
                Value va = kb.create<vector::BroadcastOp>(loc, vecType, a);
                Value bVec = kb.create<vector::LoadOp>(
                    loc, vecType, B, ValueRange{k, nBase});

                Value out =
                    createMulAdd(kb, loc, elementType, va, bVec, iterArgs[0]);
                kb.create<scf::YieldOp>(loc, out);
              });

          b.create<vector::StoreOp>(loc, kLoop.getResult(0), C,
                                    ValueRange{zeroIndex, nBase});
          b.create<scf::YieldOp>(loc);
        });

    // Handle tail < vectorSize by masked load/store.
    Value hasTail = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, nVec, nDim);

    auto tailIf =
        builder.create<scf::IfOp>(loc, TypeRange{}, hasTail, /*withElseRegion=*/false);

    {
      OpBuilder tb = tailIf.getThenBodyBuilder();

      Value tail = tb.create<arith::SubIOp>(loc, nDim, nVec);
      Value mask = tb.create<vector::CreateMaskOp>(
          loc, VectorType::get({vectorSize}, tb.getI1Type()), ValueRange{tail});

      Value initAcc = tb.create<vector::MaskedLoadOp>(
          loc, vecType, C, ValueRange{zeroIndex, nVec}, mask, zeroVector);

      auto kLoop = tb.create<scf::ForOp>(
          loc, zeroIndex, kDim, oneIndex, ValueRange{initAcc},
          [&](OpBuilder &kb, Location loc, Value k, ValueRange iterArgs) {
            Value a = kb.create<memref::LoadOp>(loc, A,
                                                ValueRange{zeroIndex, k});
            Value va = kb.create<vector::BroadcastOp>(loc, vecType, a);
            Value bVec = kb.create<vector::MaskedLoadOp>(
                loc, vecType, B, ValueRange{k, nVec}, mask, zeroVector);

            Value out =
                createMulAdd(kb, loc, elementType, va, bVec, iterArgs[0]);
            kb.create<scf::YieldOp>(loc, out);
          });

      tb.create<vector::MaskedStoreOp>(loc, C, ValueRange{zeroIndex, nVec}, mask,
                                       kLoop.getResult(0));
    }
  }

private:
  int64_t vectorSize;
  int64_t unrollFactor;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// MatMulVectorizationDecodePass
//===----------------------------------------------------------------------===//

namespace {

class MatMulVectorizationDecodePass
    : public PassWrapper<MatMulVectorizationDecodePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulVectorizationDecodePass)

  StringRef getArgument() const final {
    return "matmul-vectorization-decode";
  }

  StringRef getDescription() const final {
    return "MatMul decode-only vectorization optimization for static M=1.";
  }

  MatMulVectorizationDecodePass() = default;
  MatMulVectorizationDecodePass(const MatMulVectorizationDecodePass &) {}

  explicit MatMulVectorizationDecodePass(int64_t vectorSizeParam,
                                         int64_t unrollFactorParam) {
    vectorSize = vectorSizeParam;
    unrollFactor = unrollFactorParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }

  Option<int64_t> vectorSize{
      *this, "vector-size",
      llvm::cl::desc("Vector size used by decode matmul vectorization."),
      llvm::cl::init(64)};

  Option<int64_t> unrollFactor{
      *this, "unroll-factor",
      llvm::cl::desc("Column unroll factor for decode matmul."),
      llvm::cl::init(2)};
};

} // end anonymous namespace

void MatMulVectorizationDecodePass::runOnOperation() {
  if (vectorSize <= 0 || unrollFactor <= 0) {
    signalPassFailure();
    return;
  }

  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                         memref::MemRefDialect, vector::VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  // Only static M=1 matmul is illegal and should be rewritten by this pass.
  // All other matmul ops remain legal and can be handled by other passes.
  target.addDynamicallyLegalOp<linalg::MatmulOp>([](linalg::MatmulOp op) {
    auto aType = dyn_cast<MemRefType>(op.getOperand(0).getType());
    auto bType = dyn_cast<MemRefType>(op.getOperand(1).getType());
    auto cType = dyn_cast<MemRefType>(op.getOperand(2).getType());

    if (!aType || !bType || !cType)
      return true;
    if (aType.getRank() != 2 || bType.getRank() != 2 || cType.getRank() != 2)
      return true;
    if (aType.isDynamicDim(0))
      return true;

    return aType.getDimSize(0) != 1;
  });

  RewritePatternSet patterns(context);
  patterns.add<MatMulVectorizationDecodePattern>(context, vectorSize,
                                                 unrollFactor);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulVectorizationDecodePass() {
  PassRegistration<MatMulVectorizationDecodePass>();
}
} // namespace buddy
} // namespace mlir