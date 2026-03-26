#map = affine_map<(d0) -> (d0 mod 64)>
#map1 = affine_map<(d0) -> (d0 ceildiv 64)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> (d0 * 64)>
#set = affine_set<(d0)[s0] : (d0 * -64 + s0 - 64 >= 0)>
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.splat %cst : vector<64xf32>
    %c0_0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0_0 : memref<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_1 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %c0_2 = arith.constant 0 : index
    %dim_3 = memref.dim %arg1, %c0_2 : memref<?x?xf32>
    %1 = affine.apply #map(%dim_1)
    %2 = vector.create_mask %1 : vector<64xi1>
    %3 = affine.apply #map1(%dim_1)
    affine.parallel (%arg3) = (0) to (%3) {
      affine.prefetch %arg0[%dim, %dim_3], read, locality<3>, data : memref<?x?xf32>
      affine.if #set(%arg3)[%dim_1] {
        affine.for %arg4 = #map2(%c0) to #map2(%dim_3) {
          %4 = affine.vector_load %arg1[%arg4, %arg3 * 64] : memref<?x?xf32>, vector<64xf32>
          affine.for %arg5 = #map2(%c0) to #map2(%dim) {
            %5 = memref.load %arg0[%arg5, %arg4] : memref<?x?xf32>
            %6 = vector.broadcast %5 : f32 to vector<64xf32>
            %7 = affine.vector_load %arg2[%arg5, %arg3 * 64] : memref<?x?xf32>, vector<64xf32>
            %8 = vector.fma %6, %4, %7 : vector<64xf32>
            affine.vector_store %8, %arg2[%arg5, %arg3 * 64] : memref<?x?xf32>, vector<64xf32>
          }
        }
      } else {
        affine.for %arg4 = #map2(%c0) to #map2(%dim_3) {
          %4 = affine.apply #map3(%arg3)
          %5 = vector.maskedload %arg1[%arg4, %4], %2, %0 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          affine.for %arg5 = #map2(%c0) to #map2(%dim) {
            %6 = memref.load %arg0[%arg5, %arg4] : memref<?x?xf32>
            %7 = vector.broadcast %6 : f32 to vector<64xf32>
            %8 = vector.maskedload %arg2[%arg5, %4], %2, %0 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
            %9 = vector.fma %7, %5, %8 : vector<64xf32>
            vector.maskedstore %arg2[%arg5, %4], %2, %9 : memref<?x?xf32>, vector<64xi1>, vector<64xf32>
          }
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

