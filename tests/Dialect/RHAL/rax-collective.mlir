// RUN: rax-pack %s -o %t.rax
// RUN: rax-inspect %t.rax | FileCheck %s

rhal.module @stage2b {
  rhal.codeobj @first {id = 10 : i32, kind = "host_shared_lib",
                       backend = "cpu", uri = "file:kernels.so"}
  rhal.codeobj @second {id = 11 : i32, kind = "host_shared_lib",
                        backend = "cpu", uri = "file:kernels.so"}
  rhal.codeobj @third {id = 12 : i32, kind = "host_shared_lib",
                       backend = "cpu", uri = "file:kernels.so"}

  rhal.buffer @a {space = "dram", type = tensor<4xf32>}
  rhal.buffer @b {space = "dram", type = tensor<4xf32>}
  rhal.buffer @mask {space = "dram", type = tensor<4xf32>}
  rhal.buffer @cos {space = "dram", type = tensor<4xf32>}
  rhal.buffer @sin {space = "dram", type = tensor<4xf32>}

  rhal.func @ordered {inputs = ["a"], outputs = ["b"]} body {
    rhal.dispatch @first [@a]
    rhal.dispatch @second [@b]
    rhal.collective [@a, @b] {
      kind = "all_reduce",
      reduction = "sum"
    }
    rhal.dispatch @third [@b]
  }

  rhal.func @broadcast {inputs = ["mask", "cos"], outputs = ["sin"]} body {
    rhal.collective [@mask, @cos, @sin] {
      kind = "broadcast",
      root = 2 : i32
    }
  }

  rhal.func @all_gatherv {inputs = ["a"], outputs = ["b"]} body {
    rhal.collective [@a] {
      kind = "all_gatherv",
      output_buffers = [@b],
      recv_counts = array<i64: 2, 3>,
      displacements = array<i64: 0, 2>
    }
  }

  rhal.func @reduce_scatter {inputs = ["mask"], outputs = ["cos"]} body {
    rhal.collective [@mask] {
      kind = "reduce_scatter",
      output_buffers = [@cos],
      recv_counts = array<i64: 2, 2>,
      reduction = "sum"
    }
  }
}

// CHECK-LABEL: functions: 4
// CHECK-NEXT: @ordered
// CHECK-NEXT: [0] Dispatch code_object_id=10 args=[buffer:1]
// CHECK-NEXT: [1] Dispatch code_object_id=11 args=[buffer:2]
// CHECK-NEXT: [2] Collective kind=AllReduce reduction=Sum operands=[1->1, 2->2]
// CHECK-NEXT: [3] Dispatch code_object_id=12 args=[buffer:2]
// CHECK-NEXT: @broadcast
// CHECK-NEXT: [0] Collective kind=Broadcast root=2 operands=[3->3, 4->4, 5->5]
// CHECK-NEXT: @all_gatherv
// CHECK-NEXT: [0] Collective kind=AllGatherV operands=[1->2 recv_counts=[2,3] displacements=[0,2]]
// CHECK-NEXT: @reduce_scatter
// CHECK-NEXT: [0] Collective kind=ReduceScatter reduction=Sum operands=[3->4 recv_counts=[2,2]]
