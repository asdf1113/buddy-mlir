module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg2, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %c32 = arith.constant 32 : index
    %0 = arith.remui %dim, %c8 : index
    %1 = arith.subi %dim, %0 : index
    %c8_2 = arith.constant 8 : index
    scf.parallel (%arg3) = (%c0) to (%1) step (%c8_2) {
      %c0_4 = arith.constant 0 : index
      %2 = arith.addi %arg3, %c0_4 : index
      %c1_5 = arith.constant 1 : index
      %3 = arith.addi %arg3, %c1_5 : index
      %c2 = arith.constant 2 : index
      %4 = arith.addi %arg3, %c2 : index
      %c3 = arith.constant 3 : index
      %5 = arith.addi %arg3, %c3 : index
      %c4 = arith.constant 4 : index
      %6 = arith.addi %arg3, %c4 : index
      %c5 = arith.constant 5 : index
      %7 = arith.addi %arg3, %c5 : index
      %c6 = arith.constant 6 : index
      %8 = arith.addi %arg3, %c6 : index
      %c7 = arith.constant 7 : index
      %9 = arith.addi %arg3, %c7 : index
      %10 = arith.subi %dim_0, %c32 : index
      %11 = arith.addi %10, %c1 : index
      %12 = scf.for %arg4 = %c0 to %11 step %c32 iter_args(%arg5 = %c0) -> (index) {
        %13 = vector.load %arg2[%2, %arg4] : memref<?x?xf32>, vector<32xf32>
        %14 = vector.load %arg2[%3, %arg4] : memref<?x?xf32>, vector<32xf32>
        %15 = vector.load %arg2[%4, %arg4] : memref<?x?xf32>, vector<32xf32>
        %16 = vector.load %arg2[%5, %arg4] : memref<?x?xf32>, vector<32xf32>
        %17 = vector.load %arg2[%6, %arg4] : memref<?x?xf32>, vector<32xf32>
        %18 = vector.load %arg2[%7, %arg4] : memref<?x?xf32>, vector<32xf32>
        %19 = vector.load %arg2[%8, %arg4] : memref<?x?xf32>, vector<32xf32>
        %20 = vector.load %arg2[%9, %arg4] : memref<?x?xf32>, vector<32xf32>
        %21:8 = scf.for %arg6 = %c0 to %dim_1 step %c1 iter_args(%arg7 = %13, %arg8 = %14, %arg9 = %15, %arg10 = %16, %arg11 = %17, %arg12 = %18, %arg13 = %19, %arg14 = %20) -> (vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>) {
          %23 = vector.load %arg1[%arg6, %arg4] : memref<?x?xf32>, vector<32xf32>
          %24 = memref.load %arg0[%2, %arg6] : memref<?x?xf32>
          %25 = vector.broadcast %24 : f32 to vector<32xf32>
          %26 = vector.fma %25, %23, %arg7 : vector<32xf32>
          %27 = memref.load %arg0[%3, %arg6] : memref<?x?xf32>
          %28 = vector.broadcast %27 : f32 to vector<32xf32>
          %29 = vector.fma %28, %23, %arg8 : vector<32xf32>
          %30 = memref.load %arg0[%4, %arg6] : memref<?x?xf32>
          %31 = vector.broadcast %30 : f32 to vector<32xf32>
          %32 = vector.fma %31, %23, %arg9 : vector<32xf32>
          %33 = memref.load %arg0[%5, %arg6] : memref<?x?xf32>
          %34 = vector.broadcast %33 : f32 to vector<32xf32>
          %35 = vector.fma %34, %23, %arg10 : vector<32xf32>
          %36 = memref.load %arg0[%6, %arg6] : memref<?x?xf32>
          %37 = vector.broadcast %36 : f32 to vector<32xf32>
          %38 = vector.fma %37, %23, %arg11 : vector<32xf32>
          %39 = memref.load %arg0[%7, %arg6] : memref<?x?xf32>
          %40 = vector.broadcast %39 : f32 to vector<32xf32>
          %41 = vector.fma %40, %23, %arg12 : vector<32xf32>
          %42 = memref.load %arg0[%8, %arg6] : memref<?x?xf32>
          %43 = vector.broadcast %42 : f32 to vector<32xf32>
          %44 = vector.fma %43, %23, %arg13 : vector<32xf32>
          %45 = memref.load %arg0[%9, %arg6] : memref<?x?xf32>
          %46 = vector.broadcast %45 : f32 to vector<32xf32>
          %47 = vector.fma %46, %23, %arg14 : vector<32xf32>
          scf.yield %26, %29, %32, %35, %38, %41, %44, %47 : vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>
        }
        vector.store %21#0, %arg2[%2, %arg4] : memref<?x?xf32>, vector<32xf32>
        vector.store %21#1, %arg2[%3, %arg4] : memref<?x?xf32>, vector<32xf32>
        vector.store %21#2, %arg2[%4, %arg4] : memref<?x?xf32>, vector<32xf32>
        vector.store %21#3, %arg2[%5, %arg4] : memref<?x?xf32>, vector<32xf32>
        vector.store %21#4, %arg2[%6, %arg4] : memref<?x?xf32>, vector<32xf32>
        vector.store %21#5, %arg2[%7, %arg4] : memref<?x?xf32>, vector<32xf32>
        vector.store %21#6, %arg2[%8, %arg4] : memref<?x?xf32>, vector<32xf32>
        vector.store %21#7, %arg2[%9, %arg4] : memref<?x?xf32>, vector<32xf32>
        %22 = arith.addi %arg4, %c32 : index
        scf.yield %22 : index
      }
      scf.for %arg4 = %12 to %dim_0 step %c1 {
        %13 = memref.load %arg2[%2, %arg4] : memref<?x?xf32>
        %14 = memref.load %arg2[%3, %arg4] : memref<?x?xf32>
        %15 = memref.load %arg2[%4, %arg4] : memref<?x?xf32>
        %16 = memref.load %arg2[%5, %arg4] : memref<?x?xf32>
        %17 = memref.load %arg2[%6, %arg4] : memref<?x?xf32>
        %18 = memref.load %arg2[%7, %arg4] : memref<?x?xf32>
        %19 = memref.load %arg2[%8, %arg4] : memref<?x?xf32>
        %20 = memref.load %arg2[%9, %arg4] : memref<?x?xf32>
        %21:8 = scf.for %arg5 = %c0 to %dim_1 step %c1 iter_args(%arg6 = %13, %arg7 = %14, %arg8 = %15, %arg9 = %16, %arg10 = %17, %arg11 = %18, %arg12 = %19, %arg13 = %20) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
          %22 = memref.load %arg1[%arg5, %arg4] : memref<?x?xf32>
          %23 = memref.load %arg0[%2, %arg5] : memref<?x?xf32>
          %24 = arith.mulf %23, %22 : f32
          %25 = arith.addf %24, %arg6 : f32
          %26 = memref.load %arg0[%3, %arg5] : memref<?x?xf32>
          %27 = arith.mulf %26, %22 : f32
          %28 = arith.addf %27, %arg7 : f32
          %29 = memref.load %arg0[%4, %arg5] : memref<?x?xf32>
          %30 = arith.mulf %29, %22 : f32
          %31 = arith.addf %30, %arg8 : f32
          %32 = memref.load %arg0[%5, %arg5] : memref<?x?xf32>
          %33 = arith.mulf %32, %22 : f32
          %34 = arith.addf %33, %arg9 : f32
          %35 = memref.load %arg0[%6, %arg5] : memref<?x?xf32>
          %36 = arith.mulf %35, %22 : f32
          %37 = arith.addf %36, %arg10 : f32
          %38 = memref.load %arg0[%7, %arg5] : memref<?x?xf32>
          %39 = arith.mulf %38, %22 : f32
          %40 = arith.addf %39, %arg11 : f32
          %41 = memref.load %arg0[%8, %arg5] : memref<?x?xf32>
          %42 = arith.mulf %41, %22 : f32
          %43 = arith.addf %42, %arg12 : f32
          %44 = memref.load %arg0[%9, %arg5] : memref<?x?xf32>
          %45 = arith.mulf %44, %22 : f32
          %46 = arith.addf %45, %arg13 : f32
          scf.yield %25, %28, %31, %34, %37, %40, %43, %46 : f32, f32, f32, f32, f32, f32, f32, f32
        }
        memref.store %21#0, %arg2[%2, %arg4] : memref<?x?xf32>
        memref.store %21#1, %arg2[%3, %arg4] : memref<?x?xf32>
        memref.store %21#2, %arg2[%4, %arg4] : memref<?x?xf32>
        memref.store %21#3, %arg2[%5, %arg4] : memref<?x?xf32>
        memref.store %21#4, %arg2[%6, %arg4] : memref<?x?xf32>
        memref.store %21#5, %arg2[%7, %arg4] : memref<?x?xf32>
        memref.store %21#6, %arg2[%8, %arg4] : memref<?x?xf32>
        memref.store %21#7, %arg2[%9, %arg4] : memref<?x?xf32>
      }
      scf.reduce 
    }
    %c1_3 = arith.constant 1 : index
    scf.parallel (%arg3) = (%1) to (%dim) step (%c1_3) {
      %c0_4 = arith.constant 0 : index
      %2 = arith.addi %arg3, %c0_4 : index
      %3 = arith.subi %dim_0, %c32 : index
      %4 = arith.addi %3, %c1 : index
      %5 = scf.for %arg4 = %c0 to %4 step %c32 iter_args(%arg5 = %c0) -> (index) {
        %6 = vector.load %arg2[%2, %arg4] : memref<?x?xf32>, vector<32xf32>
        %7 = scf.for %arg6 = %c0 to %dim_1 step %c1 iter_args(%arg7 = %6) -> (vector<32xf32>) {
          %9 = vector.load %arg1[%arg6, %arg4] : memref<?x?xf32>, vector<32xf32>
          %10 = memref.load %arg0[%2, %arg6] : memref<?x?xf32>
          %11 = vector.broadcast %10 : f32 to vector<32xf32>
          %12 = vector.fma %11, %9, %arg7 : vector<32xf32>
          scf.yield %12 : vector<32xf32>
        }
        vector.store %7, %arg2[%2, %arg4] : memref<?x?xf32>, vector<32xf32>
        %8 = arith.addi %arg4, %c32 : index
        scf.yield %8 : index
      }
      scf.for %arg4 = %5 to %dim_0 step %c1 {
        %6 = memref.load %arg2[%2, %arg4] : memref<?x?xf32>
        %7 = scf.for %arg5 = %c0 to %dim_1 step %c1 iter_args(%arg6 = %6) -> (f32) {
          %8 = memref.load %arg1[%arg5, %arg4] : memref<?x?xf32>
          %9 = memref.load %arg0[%2, %arg5] : memref<?x?xf32>
          %10 = arith.mulf %9, %8 : f32
          %11 = arith.addf %10, %arg6 : f32
          scf.yield %11 : f32
        }
        memref.store %7, %arg2[%2, %arg4] : memref<?x?xf32>
      }
      scf.reduce 
    }
    return
  }
  func.func @main() {
    %c4 = arith.constant 4 : index
    %c4_0 = arith.constant 4 : index
    %c4_1 = arith.constant 4 : index
    %cst = arith.constant 1.000000e+00 : f32
    %alloc = memref.alloc(%c4, %c4_1) : memref<?x?xf32>
    %alloc_2 = memref.alloc(%c4_1, %c4_0) : memref<?x?xf32>
    %alloc_3 = memref.alloc(%c4, %c4_0) : memref<?x?xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<?x?xf32>)
    linalg.fill ins(%cst : f32) outs(%alloc_2 : memref<?x?xf32>)
    linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<?x?xf32>)
    call @matmul(%alloc, %alloc_2, %alloc_3) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    %cast = memref.cast %alloc_3 : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %alloc_3 : memref<?x?xf32>
    memref.dealloc %alloc_2 : memref<?x?xf32>
    memref.dealloc %alloc : memref<?x?xf32>
    return
  }
}

