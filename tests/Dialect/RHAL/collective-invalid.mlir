// RUN: split-file %s %t
// RUN: not rax-pack %t/missing-reduction.mlir -o %t/missing-reduction.rax 2>&1 | FileCheck %s --check-prefix=MISSING-REDUCTION
// RUN: not rax-pack %t/invalid-reduction.mlir -o %t/invalid-reduction.rax 2>&1 | FileCheck %s --check-prefix=INVALID-REDUCTION
// RUN: not rax-pack %t/missing-root.mlir -o %t/missing-root.rax 2>&1 | FileCheck %s --check-prefix=MISSING-ROOT
// RUN: not rax-pack %t/unsupported-kind.mlir -o %t/unsupported-kind.rax 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-KIND
// RUN: not rax-pack %t/empty-buffers.mlir -o %t/empty-buffers.rax 2>&1 | FileCheck %s --check-prefix=EMPTY-BUFFERS
// RUN: not rax-pack %t/negative-root.mlir -o %t/negative-root.rax 2>&1 | FileCheck %s --check-prefix=NEGATIVE-ROOT
// RUN: not rax-pack %t/all-reduce-output-buffers.mlir -o %t/all-reduce-output-buffers.rax 2>&1 | FileCheck %s --check-prefix=ALL-REDUCE-OUTPUT-BUFFERS
// RUN: not rax-pack %t/all-reduce-recv-counts.mlir -o %t/all-reduce-recv-counts.rax 2>&1 | FileCheck %s --check-prefix=ALL-REDUCE-RECV-COUNTS
// RUN: not rax-pack %t/all-reduce-displacements.mlir -o %t/all-reduce-displacements.rax 2>&1 | FileCheck %s --check-prefix=ALL-REDUCE-DISPLACEMENTS
// RUN: not rax-pack %t/broadcast-output-buffers.mlir -o %t/broadcast-output-buffers.rax 2>&1 | FileCheck %s --check-prefix=BROADCAST-OUTPUT-BUFFERS
// RUN: not rax-pack %t/broadcast-reduction.mlir -o %t/broadcast-reduction.rax 2>&1 | FileCheck %s --check-prefix=BROADCAST-REDUCTION
// RUN: not rax-pack %t/broadcast-recv-counts.mlir -o %t/broadcast-recv-counts.rax 2>&1 | FileCheck %s --check-prefix=BROADCAST-RECV-COUNTS
// RUN: not rax-pack %t/broadcast-displacements.mlir -o %t/broadcast-displacements.rax 2>&1 | FileCheck %s --check-prefix=BROADCAST-DISPLACEMENTS
// RUN: not rax-pack %t/all-gatherv-missing-output.mlir -o %t/all-gatherv-missing-output.rax 2>&1 | FileCheck %s --check-prefix=ALL-GATHERV-MISSING-OUTPUT
// RUN: not rax-pack %t/all-gatherv-missing-counts.mlir -o %t/all-gatherv-missing-counts.rax 2>&1 | FileCheck %s --check-prefix=ALL-GATHERV-MISSING-COUNTS
// RUN: not rax-pack %t/all-gatherv-missing-displacements.mlir -o %t/all-gatherv-missing-displacements.rax 2>&1 | FileCheck %s --check-prefix=ALL-GATHERV-MISSING-DISPLACEMENTS
// RUN: not rax-pack %t/all-gatherv-size-mismatch.mlir -o %t/all-gatherv-size-mismatch.rax 2>&1 | FileCheck %s --check-prefix=ALL-GATHERV-SIZE-MISMATCH
// RUN: not rax-pack %t/all-gatherv-negative-count.mlir -o %t/all-gatherv-negative-count.rax 2>&1 | FileCheck %s --check-prefix=ALL-GATHERV-NEGATIVE-COUNT
// RUN: not rax-pack %t/all-gatherv-negative-displacement.mlir -o %t/all-gatherv-negative-displacement.rax 2>&1 | FileCheck %s --check-prefix=ALL-GATHERV-NEGATIVE-DISPLACEMENT
// RUN: not rax-pack %t/all-gatherv-reduction.mlir -o %t/all-gatherv-reduction.rax 2>&1 | FileCheck %s --check-prefix=ALL-GATHERV-REDUCTION
// RUN: not rax-pack %t/all-gatherv-root.mlir -o %t/all-gatherv-root.rax 2>&1 | FileCheck %s --check-prefix=ALL-GATHERV-ROOT
// RUN: not rax-pack %t/reduce-scatter-missing-output.mlir -o %t/reduce-scatter-missing-output.rax 2>&1 | FileCheck %s --check-prefix=REDUCE-SCATTER-MISSING-OUTPUT
// RUN: not rax-pack %t/reduce-scatter-missing-counts.mlir -o %t/reduce-scatter-missing-counts.rax 2>&1 | FileCheck %s --check-prefix=REDUCE-SCATTER-MISSING-COUNTS
// RUN: not rax-pack %t/reduce-scatter-missing-reduction.mlir -o %t/reduce-scatter-missing-reduction.rax 2>&1 | FileCheck %s --check-prefix=REDUCE-SCATTER-MISSING-REDUCTION
// RUN: not rax-pack %t/reduce-scatter-displacements.mlir -o %t/reduce-scatter-displacements.rax 2>&1 | FileCheck %s --check-prefix=REDUCE-SCATTER-DISPLACEMENTS
// RUN: not rax-pack %t/reduce-scatter-root.mlir -o %t/reduce-scatter-root.rax 2>&1 | FileCheck %s --check-prefix=REDUCE-SCATTER-ROOT

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

//--- all-reduce-output-buffers.mlir
rhal.module @all_reduce_output_buffers {
  rhal.collective [@input] {
    kind = "all_reduce",
    output_buffers = [@output],
    reduction = "sum"
  }
}

// ALL-REDUCE-OUTPUT-BUFFERS: 'rhal.collective' op with kind 'all_reduce' does not accept an 'output_buffers' attribute

//--- all-reduce-recv-counts.mlir
rhal.module @all_reduce_recv_counts {
  rhal.collective [@input] {
    kind = "all_reduce",
    recv_counts = array<i64: 2, 3>,
    reduction = "sum"
  }
}

// ALL-REDUCE-RECV-COUNTS: 'rhal.collective' op with kind 'all_reduce' does not accept a 'recv_counts' attribute

//--- all-reduce-displacements.mlir
rhal.module @all_reduce_displacements {
  rhal.collective [@input] {
    kind = "all_reduce",
    displacements = array<i64: 0, 2>,
    reduction = "sum"
  }
}

// ALL-REDUCE-DISPLACEMENTS: 'rhal.collective' op with kind 'all_reduce' does not accept a 'displacements' attribute

//--- broadcast-output-buffers.mlir
rhal.module @broadcast_output_buffers {
  rhal.collective [@input] {
    kind = "broadcast",
    output_buffers = [@output],
    root = 0 : i32
  }
}

// BROADCAST-OUTPUT-BUFFERS: 'rhal.collective' op with kind 'broadcast' does not accept an 'output_buffers' attribute

//--- broadcast-reduction.mlir
rhal.module @broadcast_reduction {
  rhal.collective [@input] {
    kind = "broadcast",
    reduction = "sum",
    root = 0 : i32
  }
}

// BROADCAST-REDUCTION: 'rhal.collective' op with kind 'broadcast' does not accept a 'reduction' attribute

//--- broadcast-recv-counts.mlir
rhal.module @broadcast_recv_counts {
  rhal.collective [@input] {
    kind = "broadcast",
    recv_counts = array<i64: 2, 3>,
    root = 0 : i32
  }
}

// BROADCAST-RECV-COUNTS: 'rhal.collective' op with kind 'broadcast' does not accept a 'recv_counts' attribute

//--- broadcast-displacements.mlir
rhal.module @broadcast_displacements {
  rhal.collective [@input] {
    kind = "broadcast",
    displacements = array<i64: 0, 2>,
    root = 0 : i32
  }
}

// BROADCAST-DISPLACEMENTS: 'rhal.collective' op with kind 'broadcast' does not accept a 'displacements' attribute

//--- all-gatherv-missing-output.mlir
rhal.module @all_gatherv_missing_output {
  rhal.collective [@input] {
    kind = "all_gatherv",
    recv_counts = array<i64: 2, 3>,
    displacements = array<i64: 0, 2>
  }
}

// ALL-GATHERV-MISSING-OUTPUT: 'rhal.collective' op with kind 'all_gatherv' requires exactly one explicit output buffer

//--- all-gatherv-missing-counts.mlir
rhal.module @all_gatherv_missing_counts {
  rhal.collective [@input] {
    kind = "all_gatherv",
    output_buffers = [@output],
    displacements = array<i64: 0, 2>
  }
}

// ALL-GATHERV-MISSING-COUNTS: 'rhal.collective' op with kind 'all_gatherv' requires non-empty 'recv_counts'

//--- all-gatherv-missing-displacements.mlir
rhal.module @all_gatherv_missing_displacements {
  rhal.collective [@input] {
    kind = "all_gatherv",
    output_buffers = [@output],
    recv_counts = array<i64: 2, 3>
  }
}

// ALL-GATHERV-MISSING-DISPLACEMENTS: 'rhal.collective' op with kind 'all_gatherv' requires 'displacements'

//--- all-gatherv-size-mismatch.mlir
rhal.module @all_gatherv_size_mismatch {
  rhal.collective [@input] {
    kind = "all_gatherv",
    output_buffers = [@output],
    recv_counts = array<i64: 2, 3>,
    displacements = array<i64: 0>
  }
}

// ALL-GATHERV-SIZE-MISMATCH: 'rhal.collective' op with kind 'all_gatherv' requires 'recv_counts' and 'displacements' to have equal sizes

//--- all-gatherv-negative-count.mlir
rhal.module @all_gatherv_negative_count {
  rhal.collective [@input] {
    kind = "all_gatherv",
    output_buffers = [@output],
    recv_counts = array<i64: 2, -3>,
    displacements = array<i64: 0, 2>
  }
}

// ALL-GATHERV-NEGATIVE-COUNT: 'rhal.collective' op with kind 'all_gatherv' requires non-negative recv_counts

//--- all-gatherv-negative-displacement.mlir
rhal.module @all_gatherv_negative_displacement {
  rhal.collective [@input] {
    kind = "all_gatherv",
    output_buffers = [@output],
    recv_counts = array<i64: 2, 3>,
    displacements = array<i64: 0, -2>
  }
}

// ALL-GATHERV-NEGATIVE-DISPLACEMENT: 'rhal.collective' op with kind 'all_gatherv' requires non-negative displacements

//--- all-gatherv-reduction.mlir
rhal.module @all_gatherv_reduction {
  rhal.collective [@input] {
    kind = "all_gatherv",
    output_buffers = [@output],
    recv_counts = array<i64: 2, 3>,
    displacements = array<i64: 0, 2>,
    reduction = "sum"
  }
}

// ALL-GATHERV-REDUCTION: 'rhal.collective' op with kind 'all_gatherv' does not accept a 'reduction' attribute

//--- all-gatherv-root.mlir
rhal.module @all_gatherv_root {
  rhal.collective [@input] {
    kind = "all_gatherv",
    output_buffers = [@output],
    recv_counts = array<i64: 2, 3>,
    displacements = array<i64: 0, 2>,
    root = 0 : i32
  }
}

// ALL-GATHERV-ROOT: 'rhal.collective' op with kind 'all_gatherv' does not accept a 'root' attribute

//--- reduce-scatter-missing-output.mlir
rhal.module @reduce_scatter_missing_output {
  rhal.collective [@input] {
    kind = "reduce_scatter",
    recv_counts = array<i64: 2, 2>,
    reduction = "sum"
  }
}

// REDUCE-SCATTER-MISSING-OUTPUT: 'rhal.collective' op with kind 'reduce_scatter' requires exactly one explicit output buffer

//--- reduce-scatter-missing-counts.mlir
rhal.module @reduce_scatter_missing_counts {
  rhal.collective [@input] {
    kind = "reduce_scatter",
    output_buffers = [@output],
    reduction = "sum"
  }
}

// REDUCE-SCATTER-MISSING-COUNTS: 'rhal.collective' op with kind 'reduce_scatter' requires non-empty 'recv_counts'

//--- reduce-scatter-missing-reduction.mlir
rhal.module @reduce_scatter_missing_reduction {
  rhal.collective [@input] {
    kind = "reduce_scatter",
    output_buffers = [@output],
    recv_counts = array<i64: 2, 2>
  }
}

// REDUCE-SCATTER-MISSING-REDUCTION: 'rhal.collective' op with kind 'reduce_scatter' requires reduction = "sum"

//--- reduce-scatter-displacements.mlir
rhal.module @reduce_scatter_displacements {
  rhal.collective [@input] {
    kind = "reduce_scatter",
    output_buffers = [@output],
    recv_counts = array<i64: 2, 2>,
    displacements = array<i64: 0, 2>,
    reduction = "sum"
  }
}

// REDUCE-SCATTER-DISPLACEMENTS: 'rhal.collective' op with kind 'reduce_scatter' does not accept a 'displacements' attribute

//--- reduce-scatter-root.mlir
rhal.module @reduce_scatter_root {
  rhal.collective [@input] {
    kind = "reduce_scatter",
    output_buffers = [@output],
    recv_counts = array<i64: 2, 2>,
    reduction = "sum",
    root = 0 : i32
  }
}

// REDUCE-SCATTER-ROOT: 'rhal.collective' op with kind 'reduce_scatter' does not accept a 'root' attribute
