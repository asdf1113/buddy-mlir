#map = affine_map<(d0) -> (d0 - 63)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0)[s0] -> (d0 + 32, s0)>
#map3 = affine_map<(d0) -> (d0 mod 64)>
#set = affine_set<(d0) : (d0 mod 64 - 1 >= 0)>
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.splat %cst : vector<64xf32>
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg1, %c0 : memref<?x?xf32>
    %1 = affine.apply #map(%dim_0)
    affine.parallel (%arg3) = (0) to (%1) step (64) {
      affine.prefetch %arg0[%dim, %dim_1], read, locality<3>, data : memref<?x?xf32>
      affine.for %arg4 = #map1(%c0) to #map1(%dim_1) step 32 {
        %2 = affine.min #map2(%arg4)[%dim_1]
        affine.for %arg5 = #map1(%c0) to #map1(%dim) {
          %3 = affine.vector_load %arg2[%arg5, %arg3] : memref<?x?xf32>, vector<64xf32>
          %4 = scf.for %arg6 = %arg4 to %2 step %c1 iter_args(%arg7 = %3) -> (vector<64xf32>) {
            %5 = vector.load %arg1[%arg6, %arg3] : memref<?x?xf32>, vector<64xf32>
            %6 = memref.load %arg0[%arg5, %arg6] : memref<?x?xf32>
            %7 = vector.broadcast %6 : f32 to vector<64xf32>
            %8 = vector.fma %7, %5, %arg7 : vector<64xf32>
            scf.yield %8 : vector<64xf32>
          }
          vector.store %4, %arg2[%arg5, %arg3] : memref<?x?xf32>, vector<64xf32>
        }
      }
    }
    affine.if #set(%dim_0) {
      %2 = affine.apply #map3(%dim_0)
      %3 = vector.create_mask %2 : vector<64xi1>
      %4 = arith.subi %dim_0, %2 : index
      affine.for %arg3 = #map1(%c0) to #map1(%dim_1) step 32 {
        %5 = affine.min #map2(%arg3)[%dim_1]
        affine.for %arg4 = #map1(%c0) to #map1(%dim) {
          %6 = vector.maskedload %arg2[%arg4, %4], %3, %0 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %7 = scf.for %arg5 = %arg3 to %5 step %c1 iter_args(%arg6 = %6) -> (vector<64xf32>) {
            %8 = vector.maskedload %arg1[%arg5, %4], %3, %0 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
            %9 = memref.load %arg0[%arg4, %arg5] : memref<?x?xf32>
            %10 = vector.broadcast %9 : f32 to vector<64xf32>
            %11 = vector.fma %10, %8, %arg6 : vector<64xf32>
            scf.yield %11 : vector<64xf32>
          }
          vector.maskedstore %arg2[%arg4, %4], %3, %7 : memref<?x?xf32>, vector<64xi1>, vector<64xf32>
        }
      }
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

