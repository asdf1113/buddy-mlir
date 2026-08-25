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
}
