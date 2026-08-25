// RUN: rax-pack %s -o %t.rax

rhal.module @collectives {
  rhal.buffer @a {space = "dram", type = tensor<4xf32>}
  rhal.buffer @b {space = "dram", type = tensor<4xf32>}
  rhal.buffer @mask {space = "dram", type = tensor<4xf32>}
  rhal.buffer @cos {space = "dram", type = tensor<4xf32>}
  rhal.buffer @sin {space = "dram", type = tensor<4xf32>}

  rhal.collective [@a, @b] {
    kind = "all_reduce",
    reduction = "sum"
  }
  rhal.collective [@mask, @cos, @sin] {
    kind = "broadcast",
    root = 0 : i32
  }
  rhal.collective [@a] {
    kind = "all_gatherv",
    output_buffers = [@b],
    recv_counts = array<i64: 2, 3>,
    displacements = array<i64: 0, 2>
  }
  rhal.collective [@mask] {
    kind = "reduce_scatter",
    output_buffers = [@cos],
    recv_counts = array<i64: 2, 2>,
    reduction = "sum"
  }
}
