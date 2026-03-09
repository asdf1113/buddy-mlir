#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0) -> (d0)>
module {
  func.func @subgraph0_prefill134(%arg0: tensor<4096x2048xf32>, %arg1: tensor<4096x2048xf32>, %arg2: tensor<4096x2048xf32>, %arg3: tensor<2048x4096xf32>, %arg4: tensor<1x1x1024x1024xi1>, %arg5: tensor<1x1024x128xf32>, %arg6: tensor<1x1024x128xf32>, %arg7: tensor<1x1024x4096xf32>) -> (tensor<1x16x1024x128xf32>, tensor<1x16x1024x128xf32>, tensor<1024x4096xf32>) {
    %0 = tosa.const_shape  {values = dense<[1024, 4096]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.reshape %arg7, %0 : (tensor<1x1024x4096xf32>, !tosa.shape<2>) -> tensor<1024x4096xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1024x2048xf32>
    %2 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1, %arg0 : tensor<1024x4096xf32>, tensor<4096x2048xf32>) outs(%cst : tensor<1024x2048xf32>) -> tensor<1024x2048xf32>
    %3 = tosa.const_shape  {values = dense<[1, 1024, 2048]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.reshape %2, %3 : (tensor<1024x2048xf32>, !tosa.shape<3>) -> tensor<1x1024x2048xf32>
    %5 = tosa.const_shape  {values = dense<[1, 1024, 16, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %6 = tosa.reshape %4, %5 : (tensor<1x1024x2048xf32>, !tosa.shape<4>) -> tensor<1x1024x16x128xf32>
    %7 = tosa.transpose %6 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x1024x16x128xf32>) -> tensor<1x16x1024x128xf32>
    %8 = tosa.const_shape  {values = dense<[1024, 4096]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %9 = tosa.reshape %arg7, %8 : (tensor<1x1024x4096xf32>, !tosa.shape<2>) -> tensor<1024x4096xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1024x2048xf32>
    %10 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%9, %arg1 : tensor<1024x4096xf32>, tensor<4096x2048xf32>) outs(%cst_0 : tensor<1024x2048xf32>) -> tensor<1024x2048xf32>
    %11 = tosa.const_shape  {values = dense<[1, 1024, 2048]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %12 = tosa.reshape %10, %11 : (tensor<1024x2048xf32>, !tosa.shape<3>) -> tensor<1x1024x2048xf32>
    %13 = tosa.const_shape  {values = dense<[1, 1024, 16, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %14 = tosa.reshape %12, %13 : (tensor<1x1024x2048xf32>, !tosa.shape<4>) -> tensor<1x1024x16x128xf32>
    %15 = tosa.transpose %14 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x1024x16x128xf32>) -> tensor<1x16x1024x128xf32>
    %16 = tosa.const_shape  {values = dense<[1024, 4096]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %17 = tosa.reshape %arg7, %16 : (tensor<1x1024x4096xf32>, !tosa.shape<2>) -> tensor<1024x4096xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1024x2048xf32>
    %18 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%17, %arg2 : tensor<1024x4096xf32>, tensor<4096x2048xf32>) outs(%cst_1 : tensor<1024x2048xf32>) -> tensor<1024x2048xf32>
    %19 = tosa.const_shape  {values = dense<[1, 1024, 2048]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %20 = tosa.reshape %18, %19 : (tensor<1024x2048xf32>, !tosa.shape<3>) -> tensor<1x1024x2048xf32>
    %21 = tosa.const_shape  {values = dense<[1, 1024, 16, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %22 = tosa.reshape %20, %21 : (tensor<1x1024x2048xf32>, !tosa.shape<4>) -> tensor<1x1024x16x128xf32>
    %23 = tosa.transpose %22 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x1024x16x128xf32>) -> tensor<1x16x1024x128xf32>
    %24 = tosa.const_shape  {values = dense<[1, 1, 1024, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %25 = tosa.reshape %arg5, %24 : (tensor<1x1024x128xf32>, !tosa.shape<4>) -> tensor<1x1x1024x128xf32>
    %26 = tosa.const_shape  {values = dense<[1, 1, 1024, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %27 = tosa.reshape %arg6, %26 : (tensor<1x1024x128xf32>, !tosa.shape<4>) -> tensor<1x1x1024x128xf32>
    %28 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %29 = tosa.mul %7, %25, %28 : (tensor<1x16x1024x128xf32>, tensor<1x1x1024x128xf32>, tensor<1xi8>) -> tensor<1x16x1024x128xf32>
    %extracted_slice = tensor.extract_slice %7[0, 0, 0, 0] [1, 16, 1024, 64] [1, 1, 1, 1] : tensor<1x16x1024x128xf32> to tensor<1x16x1024x64xf32>
    %extracted_slice_2 = tensor.extract_slice %7[0, 0, 0, 64] [1, 16, 1024, 64] [1, 1, 1, 1] : tensor<1x16x1024x128xf32> to tensor<1x16x1024x64xf32>
    %30 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %31 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %32 = tosa.negate %extracted_slice_2, %30, %31 : (tensor<1x16x1024x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x16x1024x64xf32>
    %33 = tensor.empty() : tensor<1x16x1024x128xf32>
    %inserted_slice = tensor.insert_slice %32 into %33[0, 0, 0, 0] [1, 16, 1024, 64] [1, 1, 1, 1] : tensor<1x16x1024x64xf32> into tensor<1x16x1024x128xf32>
    %inserted_slice_3 = tensor.insert_slice %extracted_slice into %inserted_slice[0, 0, 0, 64] [1, 16, 1024, 64] [1, 1, 1, 1] : tensor<1x16x1024x64xf32> into tensor<1x16x1024x128xf32>
    %34 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %35 = tosa.mul %inserted_slice_3, %27, %34 : (tensor<1x16x1024x128xf32>, tensor<1x1x1024x128xf32>, tensor<1xi8>) -> tensor<1x16x1024x128xf32>
    %36 = tosa.add %29, %35 : (tensor<1x16x1024x128xf32>, tensor<1x16x1024x128xf32>) -> tensor<1x16x1024x128xf32>
    %37 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %38 = tosa.mul %15, %25, %37 : (tensor<1x16x1024x128xf32>, tensor<1x1x1024x128xf32>, tensor<1xi8>) -> tensor<1x16x1024x128xf32>
    %extracted_slice_4 = tensor.extract_slice %15[0, 0, 0, 0] [1, 16, 1024, 64] [1, 1, 1, 1] : tensor<1x16x1024x128xf32> to tensor<1x16x1024x64xf32>
    %extracted_slice_5 = tensor.extract_slice %15[0, 0, 0, 64] [1, 16, 1024, 64] [1, 1, 1, 1] : tensor<1x16x1024x128xf32> to tensor<1x16x1024x64xf32>
    %39 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %40 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %41 = tosa.negate %extracted_slice_5, %39, %40 : (tensor<1x16x1024x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x16x1024x64xf32>
    %42 = tensor.empty() : tensor<1x16x1024x128xf32>
    %inserted_slice_6 = tensor.insert_slice %41 into %42[0, 0, 0, 0] [1, 16, 1024, 64] [1, 1, 1, 1] : tensor<1x16x1024x64xf32> into tensor<1x16x1024x128xf32>
    %inserted_slice_7 = tensor.insert_slice %extracted_slice_4 into %inserted_slice_6[0, 0, 0, 64] [1, 16, 1024, 64] [1, 1, 1, 1] : tensor<1x16x1024x64xf32> into tensor<1x16x1024x128xf32>
    %43 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %44 = tosa.mul %inserted_slice_7, %27, %43 : (tensor<1x16x1024x128xf32>, tensor<1x1x1024x128xf32>, tensor<1xi8>) -> tensor<1x16x1024x128xf32>
    %45 = tosa.add %38, %44 : (tensor<1x16x1024x128xf32>, tensor<1x16x1024x128xf32>) -> tensor<1x16x1024x128xf32>
    %cst_8 = arith.constant 0xFF800000 : f32
    %cst_9 = arith.constant 0.000000e+00 : f32
    %46 = tensor.empty() : tensor<1x1x1024x1024xf32>
    %splat = tensor.splat %cst_9 : tensor<1x1x1024x1024xf32>
    %splat_10 = tensor.splat %cst_8 : tensor<1x1x1024x1024xf32>
    %47 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg4, %splat, %splat_10 : tensor<1x1x1024x1024xi1>, tensor<1x1x1024x1024xf32>, tensor<1x1x1024x1024xf32>) outs(%46 : tensor<1x1x1024x1024xf32>) {
    ^bb0(%in: i1, %in_20: f32, %in_21: f32, %out: f32):
      %61 = arith.select %in, %in_20, %in_21 : f32
      linalg.yield %61 : f32
    } -> tensor<1x1x1024x1024xf32>
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst_11 = arith.constant 0.0883883461 : f32
    %cst_12 = arith.constant 0.000000e+00 : f32
    %cst_13 = arith.constant -1.000000e+30 : f32
    %48 = vector.splat %cst_12 : vector<16xf32>
    %c1 = arith.constant 1 : index
    %49 = bufferization.to_buffer %36 : tensor<1x16x1024x128xf32> to memref<1x16x1024x128xf32>
    %50 = bufferization.to_buffer %45 : tensor<1x16x1024x128xf32> to memref<1x16x1024x128xf32>
    %51 = bufferization.to_buffer %23 : tensor<1x16x1024x128xf32> to memref<1x16x1024x128xf32>
    %52 = bufferization.to_buffer %47 : tensor<1x1x1024x1024xf32> to memref<1x1x1024x1024xf32>
    %c1_14 = arith.constant 1 : index
    %c16_15 = arith.constant 16 : index
    %c1024 = arith.constant 1024 : index
    %c128 = arith.constant 128 : index
    %c1024_16 = arith.constant 1024 : index
    %c16_17 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<1x16x1024x128xf32>
    %alloc_18 = memref.alloc() : memref<1x16x1024xf32>
    affine.for %arg8 = 0 to #map1(%c1_14) {
      affine.parallel (%arg9) = (0) to (%c16_15) {
        affine.parallel (%arg10) = (0) to (%c1024) step (16) {
          %alloc_20 = memref.alloc() : memref<16xf32>
          %alloc_21 = memref.alloc() : memref<16xf32>
          %alloc_22 = memref.alloc() : memref<16x128xf32>
          scf.for %arg11 = %c0 to %c16_17 step %c1 {
            memref.store %cst_13, %alloc_20[%arg11] : memref<16xf32>
            memref.store %cst_12, %alloc_21[%arg11] : memref<16xf32>
          }
          scf.for %arg11 = %c0 to %c16_17 step %c1 {
            scf.for %arg12 = %c0 to %c128 step %c16 {
              vector.store %48, %alloc_22[%arg11, %arg12] : memref<16x128xf32>, vector<16xf32>
            }
          }
          affine.parallel (%arg11) = (0) to (%c1024_16) step (64) {
            %alloc_23 = memref.alloc() : memref<16x64xf32>
            scf.for %arg12 = %c0 to %c16_17 step %c1 {
              scf.for %arg13 = %c0 to %c64 step %c1 {
                memref.store %cst_12, %alloc_23[%arg12, %arg13] : memref<16x64xf32>
              }
            }
            scf.for %arg12 = %c0 to %c16_17 step %c1 {
              %61 = arith.addi %arg10, %arg12 : index
              scf.for %arg13 = %c0 to %c64 step %c1 {
                %62 = arith.addi %arg11, %arg13 : index
                %63 = scf.for %arg14 = %c0 to %c128 step %c16 iter_args(%arg15 = %48) -> (vector<16xf32>) {
                  %68 = vector.load %49[%arg8, %arg9, %61, %arg14] : memref<1x16x1024x128xf32>, vector<16xf32>
                  %69 = vector.load %50[%arg8, %arg9, %62, %arg14] : memref<1x16x1024x128xf32>, vector<16xf32>
                  %70 = vector.fma %68, %69, %arg15 : vector<16xf32>
                  scf.yield %70 : vector<16xf32>
                }
                %64 = vector.reduction <add>, %63 : vector<16xf32> into f32
                %65 = arith.mulf %64, %cst_11 : f32
                %66 = memref.load %52[%arg8, %c0, %61, %62] : memref<1x1x1024x1024xf32>
                %67 = arith.addf %65, %66 : f32
                memref.store %67, %alloc_23[%arg12, %arg13] : memref<16x64xf32>
              }
            }
            scf.for %arg12 = %c0 to %c16_17 step %c1 {
              %61 = scf.for %arg13 = %c0 to %c64 step %c1 iter_args(%arg14 = %cst_13) -> (f32) {
                %76 = memref.load %alloc_23[%arg12, %arg13] : memref<16x64xf32>
                %77 = arith.cmpf ogt, %76, %arg14 : f32
                %78 = arith.select %77, %76, %arg14 : f32
                scf.yield %78 : f32
              }
              %alloc_24 = memref.alloc() : memref<128xf32>
              scf.for %arg13 = %c0 to %c128 step %c1 {
                memref.store %cst_12, %alloc_24[%arg13] : memref<128xf32>
              }
              %62 = scf.for %arg13 = %c0 to %c64 step %c1 iter_args(%arg14 = %cst_12) -> (f32) {
                %76 = arith.addi %arg11, %arg13 : index
                %77 = memref.load %alloc_23[%arg12, %arg13] : memref<16x64xf32>
                %78 = arith.subf %77, %61 : f32
                %79 = math.exp %78 : f32
                %80 = vector.splat %79 : vector<16xf32>
                %81 = arith.addf %arg14, %79 : f32
                scf.for %arg15 = %c0 to %c128 step %c16 {
                  %82 = vector.load %51[%arg8, %arg9, %76, %arg15] : memref<1x16x1024x128xf32>, vector<16xf32>
                  %83 = vector.load %alloc_24[%arg15] : memref<128xf32>, vector<16xf32>
                  %84 = vector.fma %82, %80, %83 : vector<16xf32>
                  vector.store %84, %alloc_24[%arg15] : memref<128xf32>, vector<16xf32>
                }
                scf.yield %81 : f32
              }
              %63 = memref.load %alloc_20[%arg12] : memref<16xf32>
              %64 = arith.cmpf ogt, %61, %63 : f32
              %65 = arith.select %64, %61, %63 : f32
              %66 = arith.subf %63, %65 : f32
              %67 = math.exp %66 : f32
              %68 = vector.splat %67 : vector<16xf32>
              %69 = arith.subf %61, %65 : f32
              %70 = math.exp %69 : f32
              %71 = vector.splat %70 : vector<16xf32>
              scf.for %arg13 = %c0 to %c128 step %c16 {
                %76 = vector.load %alloc_22[%arg12, %arg13] : memref<16x128xf32>, vector<16xf32>
                %77 = vector.load %alloc_24[%arg13] : memref<128xf32>, vector<16xf32>
                %78 = arith.mulf %76, %68 : vector<16xf32>
                %79 = arith.mulf %77, %71 : vector<16xf32>
                %80 = arith.addf %78, %79 : vector<16xf32>
                vector.store %80, %alloc_22[%arg12, %arg13] : memref<16x128xf32>, vector<16xf32>
              }
              %72 = memref.load %alloc_21[%arg12] : memref<16xf32>
              %73 = arith.mulf %72, %67 : f32
              %74 = arith.mulf %62, %70 : f32
              %75 = arith.addf %73, %74 : f32
              memref.store %75, %alloc_21[%arg12] : memref<16xf32>
              memref.store %65, %alloc_20[%arg12] : memref<16xf32>
            }
          }
          scf.for %arg11 = %c0 to %c16_17 step %c1 {
            %61 = arith.addi %arg10, %arg11 : index
            %62 = memref.load %alloc_21[%arg11] : memref<16xf32>
            %63 = vector.splat %62 : vector<16xf32>
            memref.store %62, %alloc_18[%arg8, %arg9, %61] : memref<1x16x1024xf32>
            scf.for %arg12 = %c0 to %c128 step %c16 {
              %64 = vector.load %alloc_22[%arg11, %arg12] : memref<16x128xf32>, vector<16xf32>
              %65 = arith.divf %64, %63 : vector<16xf32>
              vector.store %65, %alloc[%arg8, %arg9, %61, %arg12] : memref<1x16x1024x128xf32>, vector<16xf32>
            }
          }
        }
      }
    }
    %53 = bufferization.to_tensor %alloc restrict : memref<1x16x1024x128xf32> to tensor<1x16x1024x128xf32>
    %54 = bufferization.to_tensor %alloc_18 restrict : memref<1x16x1024xf32> to tensor<1x16x1024xf32>
    %55 = tosa.transpose %53 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x16x1024x128xf32>) -> tensor<1x1024x16x128xf32>
    %56 = tosa.const_shape  {values = dense<[1, 1024, 2048]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %57 = tosa.reshape %55, %56 : (tensor<1x1024x16x128xf32>, !tosa.shape<3>) -> tensor<1x1024x2048xf32>
    %58 = tosa.const_shape  {values = dense<[1024, 2048]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %59 = tosa.reshape %57, %58 : (tensor<1x1024x2048xf32>, !tosa.shape<2>) -> tensor<1024x2048xf32>
    %cst_19 = arith.constant dense<0.000000e+00> : tensor<1024x4096xf32>
    %60 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%59, %arg3 : tensor<1024x2048xf32>, tensor<2048x4096xf32>) outs(%cst_19 : tensor<1024x4096xf32>) -> tensor<1024x4096xf32>
    return %45, %23, %60 : tensor<1x16x1024x128xf32>, tensor<1x16x1024x128xf32>, tensor<1024x4096xf32>
  }
}

