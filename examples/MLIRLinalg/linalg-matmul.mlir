module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @rtclock() -> f64

  // A: [1, K], B: [K, N], C: [1, N]
  func.func @matmul(%a : memref<1x?xf32>, %b : memref<?x?xf32>, %c : memref<1x?xf32>) {
    %t_start = call @rtclock() : () -> f64

    linalg.matmul
      ins(%a, %b : memref<1x?xf32>, memref<?x?xf32>)
      outs(%c : memref<1x?xf32>)

    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64

    %print_c = memref.cast %c : memref<1x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_c) : (memref<*xf32>) -> ()

    vector.print %time : f64
    return
  }

  func.func @main() {
    // Set up dims: [1,1536] x [1536,4480] -> [1,4480]
    // %cN = arith.constant 4480 : index
    // %cK = arith.constant 1536 : index

    %cN = arith.constant 1536 : index
    %cK = arith.constant 4480 : index

    // Init values
    %cf0 = arith.constant 2.0 : f32
    %cf1 = arith.constant 3.0 : f32

    // memref<1x?xf32> only has 1 dynamic dim, so alloc takes one dynamic size.
    %A = memref.alloc(%cK) : memref<1x?xf32>
    %B = memref.alloc(%cK, %cN) : memref<?x?xf32>
    %C = memref.alloc(%cN) : memref<1x?xf32>

    linalg.fill
      ins(%cf1 : f32)
      outs(%A : memref<1x?xf32>)

    linalg.fill
      ins(%cf1 : f32)
      outs(%B : memref<?x?xf32>)

    linalg.fill
      ins(%cf0 : f32)
      outs(%C : memref<1x?xf32>)

    call @matmul(%A, %B, %C) : (memref<1x?xf32>, memref<?x?xf32>, memref<1x?xf32>) -> ()

    memref.dealloc %C : memref<1x?xf32>
    memref.dealloc %B : memref<?x?xf32>
    memref.dealloc %A : memref<1x?xf32>
    return
  }
}