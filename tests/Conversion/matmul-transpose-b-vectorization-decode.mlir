// RUN: buddy-opt %s \
// RUN:     -matmul-transpose-b-vectorization-decode="vector-size=8 unroll=2 n-tile=3" \
// RUN:     -convert-linalg-to-loops -lower-affine \
// RUN:     -convert-scf-to-cf -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#map_a = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_bt = affine_map<(d0, d1, d2) -> (d1, d2)>
#map_c = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @matmul_transpose_b_decode(
      %a: memref<1x20xf32>,
      %b: memref<5x20xf32>,
      %c: memref<1x5xf32>) {
    linalg.matmul
        indexing_maps = [#map_a, #map_bt, #map_c]
        ins(%a, %b : memref<1x20xf32>, memref<5x20xf32>)
        outs(%c : memref<1x5xf32>)
    return
  }

  func.func @main() {
    %one = arith.constant 1.0 : f32

    %a = memref.alloc() : memref<1x20xf32>
    %b = memref.alloc() : memref<5x20xf32>
    %c = memref.alloc() : memref<1x5xf32>

    linalg.fill ins(%one : f32) outs(%a : memref<1x20xf32>)
    linalg.fill ins(%one : f32) outs(%b : memref<5x20xf32>)
    linalg.fill ins(%one : f32) outs(%c : memref<1x5xf32>)

    call @matmul_transpose_b_decode(%a, %b, %c)
        : (memref<1x20xf32>,
           memref<5x20xf32>,
           memref<1x5xf32>) -> ()

    // Each output element is initialized to 1 and accumulates twenty
    // products whose operands are both 1.
    //
    // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [1, 5] strides = [5, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME:  [21, 21, 21, 21, 21]
    // CHECK-SAME: ]

    %result =
        memref.cast %c : memref<1x5xf32> to memref<*xf32>
    call @printMemrefF32(%result) : (memref<*xf32>) -> ()

    memref.dealloc %c : memref<1x5xf32>
    memref.dealloc %b : memref<5x20xf32>
    memref.dealloc %a : memref<1x20xf32>
    return
  }
}