// RUN: rax-pack %s -o %t.rax
// RUN: rax-inspect %t.rax | FileCheck %s

rhal.module @stage1a attributes {version = "0.1.0"} {
  rhal.constant @weights {id = 7 : i32, storage = "external",
                          type = tensor<4xf32>, uri = "file:weights.bin"}

  rhal.codeobj @stage0 {id = 10 : i32, kind = "host_shared_lib",
                        backend = "cpu", uri = "file:kernels.so",
                        entry_symbol = "kernel_a"}
  rhal.codeobj @stage1 {id = 11 : i32, kind = "host_shared_lib",
                        backend = "cpu", uri = "file:kernels.so",
                        entry_symbol = "kernel_b"}

  rhal.buffer @input {space = "host", type = tensor<4xf32>}
  rhal.buffer @scratch {space = "dram", type = tensor<4xf32>}
  rhal.buffer @output {space = "host", type = tensor<4xf32>}

  rhal.func @legacy {inputs = ["input"], outputs = ["output"],
                     dispatch = "stage0", args = ["input", "output"]}

  rhal.func @pipeline {inputs = ["input"], outputs = ["output"]} body {
    rhal.dispatch @stage0 [@weights, @input, @scratch]
    rhal.dispatch @stage1 [@scratch, @weights, @output]
  }
}

// CHECK: code_objects: 2
// CHECK-NEXT: [10] @stage0  kind=HostSharedLib  uri=file:kernels.so  entry_symbol=kernel_a
// CHECK-NEXT: [11] @stage1  kind=HostSharedLib  uri=file:kernels.so  entry_symbol=kernel_b
// CHECK: functions: 2
// CHECK-NEXT: @legacy
// CHECK-NEXT: [0] Dispatch code_object_id=10 args=[buffer:1, buffer:3]
// CHECK-NEXT: [1] Barrier
// CHECK-NEXT: @pipeline
// CHECK-NEXT: [0] Dispatch code_object_id=10 args=[constant:7, buffer:1, buffer:2]
// CHECK-NEXT: [1] Dispatch code_object_id=11 args=[buffer:2, constant:7, buffer:3]
