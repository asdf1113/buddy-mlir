// RUN: split-file %s %t
// RUN: not rax-pack %t/missing-reduction.mlir -o %t/missing-reduction.rax 2>&1 | FileCheck %s --check-prefix=MISSING-REDUCTION
// RUN: not rax-pack %t/invalid-reduction.mlir -o %t/invalid-reduction.rax 2>&1 | FileCheck %s --check-prefix=INVALID-REDUCTION
// RUN: not rax-pack %t/missing-root.mlir -o %t/missing-root.rax 2>&1 | FileCheck %s --check-prefix=MISSING-ROOT
// RUN: not rax-pack %t/unsupported-kind.mlir -o %t/unsupported-kind.rax 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-KIND
// RUN: not rax-pack %t/empty-buffers.mlir -o %t/empty-buffers.rax 2>&1 | FileCheck %s --check-prefix=EMPTY-BUFFERS
// RUN: not rax-pack %t/negative-root.mlir -o %t/negative-root.rax 2>&1 | FileCheck %s --check-prefix=NEGATIVE-ROOT

//--- missing-reduction.mlir
rhal.module @missing_reduction {
  rhal.collective [@a, @b] {
    kind = "all_reduce"
  }
}

// MISSING-REDUCTION: 'rhal.collective' op with kind 'all_reduce' requires reduction = "sum"

//--- invalid-reduction.mlir
rhal.module @invalid_reduction {
  rhal.collective [@a, @b] {
    kind = "all_reduce",
    reduction = "max"
  }
}

// INVALID-REDUCTION: 'rhal.collective' op with kind 'all_reduce' requires reduction = "sum"

//--- missing-root.mlir
rhal.module @missing_root {
  rhal.collective [@mask, @cos, @sin] {
    kind = "broadcast"
  }
}

// MISSING-ROOT: 'rhal.collective' op with kind 'broadcast' requires a 'root' attribute

//--- unsupported-kind.mlir
rhal.module @unsupported_kind {
  rhal.collective [@a] {
    kind = "reduce"
  }
}

// UNSUPPORTED-KIND: 'rhal.collective' op has unsupported kind 'reduce'

//--- empty-buffers.mlir
rhal.module @empty_buffers {
  rhal.collective [] {
    kind = "all_reduce",
    reduction = "sum"
  }
}

// EMPTY-BUFFERS: 'rhal.collective' op requires at least one buffer

//--- negative-root.mlir
rhal.module @negative_root {
  rhal.collective [@a] {
    kind = "broadcast",
    root = -1 : i32
  }
}

// NEGATIVE-ROOT: 'rhal.collective' op with kind 'broadcast' requires a non-negative root
