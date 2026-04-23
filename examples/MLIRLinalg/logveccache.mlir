#map = affine_map<(d0) -> (d0 - 63)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0)[s0] -> (d0 + 16, s0)>
#map3 = affine_map<(d0) -> (d0 mod 64)>
#set = affine_set<(d0) : (d0 mod 64 - 1 >= 0)>
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @rtclock() -> f64
  func.func @matmul(%arg0: memref<1x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<1x?xf32>) {
    %0 = call @rtclock() : () -> f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %1 = vector.splat %cst : vector<64xf32>
    %dim = memref.dim %arg0, %c0 : memref<1x?xf32>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg1, %c0 : memref<?x?xf32>
    %2 = affine.apply #map(%dim_0)
    affine.parallel (%arg3) = (0) to (%2) step (64) {
      affine.prefetch %arg0[%dim, %dim_1], read, locality<3>, data : memref<1x?xf32>
      affine.for %arg4 = #map1(%c0) to #map1(%dim_1) step 16 {
        %5 = affine.min #map2(%arg4)[%dim_1]
        affine.for %arg5 = #map1(%c0) to #map1(%dim) {
          %6 = affine.vector_load %arg2[%arg5, %arg3] : memref<1x?xf32>, vector<64xf32>
          %7 = scf.for %arg6 = %arg4 to %5 step %c1 iter_args(%arg7 = %6) -> (vector<64xf32>) {
            %8 = vector.load %arg1[%arg6, %arg3] : memref<?x?xf32>, vector<64xf32>
            %9 = memref.load %arg0[%arg5, %arg6] : memref<1x?xf32>
            %10 = vector.broadcast %9 : f32 to vector<64xf32>
            %11 = vector.fma %10, %8, %arg7 : vector<64xf32>
            scf.yield %11 : vector<64xf32>
          }
          vector.store %7, %arg2[%arg5, %arg3] : memref<1x?xf32>, vector<64xf32>
        }
      }
    }
    affine.if #set(%dim_0) {
      %5 = affine.apply #map3(%dim_0)
      %6 = vector.create_mask %5 : vector<64xi1>
      %7 = arith.subi %dim_0, %5 : index
      affine.for %arg3 = #map1(%c0) to #map1(%dim_1) step 16 {
        %8 = affine.min #map2(%arg3)[%dim_1]
        affine.for %arg4 = #map1(%c0) to #map1(%dim) {
          %9 = vector.maskedload %arg2[%arg4, %7], %6, %1 : memref<1x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %10 = scf.for %arg5 = %arg3 to %8 step %c1 iter_args(%arg6 = %9) -> (vector<64xf32>) {
            %11 = vector.maskedload %arg1[%arg5, %7], %6, %1 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
            %12 = memref.load %arg0[%arg4, %arg5] : memref<1x?xf32>
            %13 = vector.broadcast %12 : f32 to vector<64xf32>
            %14 = vector.fma %13, %11, %arg6 : vector<64xf32>
            scf.yield %14 : vector<64xf32>
          }
          vector.maskedstore %arg2[%arg4, %7], %6, %10 : memref<1x?xf32>, vector<64xi1>, vector<64xf32>
        }
      }
    }
    %3 = call @rtclock() : () -> f64
    %4 = arith.subf %3, %0 : f64
    %cast = memref.cast %arg2 : memref<1x?xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    vector.print %4 : f64
    return
  }
  func.func @main() {
    %c1536 = arith.constant 1536 : index
    %c4480 = arith.constant 4480 : index
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 3.000000e+00 : f32
    %alloc = memref.alloc(%c4480) : memref<1x?xf32>
    %alloc_1 = memref.alloc(%c4480, %c1536) : memref<?x?xf32>
    %alloc_2 = memref.alloc(%c1536) : memref<1x?xf32>
    linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<1x?xf32>)
    linalg.fill ins(%cst_0 : f32) outs(%alloc_1 : memref<?x?xf32>)
    linalg.fill ins(%cst : f32) outs(%alloc_2 : memref<1x?xf32>)
    call @matmul(%alloc, %alloc_1, %alloc_2) : (memref<1x?xf32>, memref<?x?xf32>, memref<1x?xf32>) -> ()
    memref.dealloc %alloc_2 : memref<1x?xf32>
    memref.dealloc %alloc_1 : memref<?x?xf32>
    memref.dealloc %alloc : memref<1x?xf32>
    return
  }
}

