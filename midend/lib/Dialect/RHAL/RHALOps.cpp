//===- RHALOps.cpp - RHAL Dialect Operations ------------------------------===//
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

#include "RHAL/RHALOps.h"
#include "RHAL/RHALDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "RHAL/RHALOps.cpp.inc"

mlir::LogicalResult buddy::rhal::FuncOp::verify() {
  bool hasDispatch = (*this)->hasAttr("dispatch");
  bool hasArgs = (*this)->hasAttr("args");

  if (getBody().empty()) {
    if (!hasDispatch || !hasArgs)
      return emitOpError(
          "legacy form requires 'dispatch' and 'args' attributes");
  } else if (hasDispatch || hasArgs) {
    return emitOpError(
        "body form must not have legacy 'dispatch' or 'args' attributes");
  }

  return mlir::success();
}

mlir::LogicalResult buddy::rhal::CollectiveOp::verify() {
  if (getBuffers().empty())
    return emitOpError("requires at least one buffer");

  mlir::StringAttr kind = getKindAttr();
  mlir::StringAttr reduction = getReductionAttr();
  mlir::IntegerAttr root = getRootAttr();

  if (kind.getValue() == "all_reduce") {
    if (!reduction || reduction.getValue() != "sum")
      return emitOpError("with kind 'all_reduce' requires reduction = \"sum\"");
    return mlir::success();
  }

  if (kind.getValue() == "broadcast") {
    if (!root)
      return emitOpError("with kind 'broadcast' requires a 'root' attribute");
    if (root.getInt() < 0)
      return emitOpError("with kind 'broadcast' requires a non-negative root");
    return mlir::success();
  }

  return emitOpError("has unsupported kind '") << kind.getValue() << "'";
}
