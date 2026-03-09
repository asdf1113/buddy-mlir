  func.func @subgraph0_decode2(
    %arg0: tensor<4096x4096xf32>, 
    %arg1: tensor<4096x4096xf32>, 
    %arg2: tensor<4096x4096xf32>, 
    %arg3: tensor<4096x4096xf32>, 
    %arg4: tensor<1xi64>, 
    %arg5: tensor<1x32x1024x128xf32>, 
    %arg6: tensor<1x32x1024x128xf32>, 
    %arg7: tensor<1x1x1x1024xi1>, 
    %arg8: tensor<1x1x128xf32>, 
    %arg9: tensor<1x1x128xf32>, 
    %arg10: tensor<1x1x4096xf32>) -> 
    (tensor<1x32x1024x128xf32>, 
    tensor<1x32x1024x128xf32>, tensor<1x4096xf32>) {
    
    %0 = tosa.const_shape  {values = dense<[1, 4096]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.reshape %arg10, %0 : (tensor<1x1x4096xf32>, !tosa.shape<2>) -> tensor<1x4096xf32>
    
    %cst = arith.constant dense<0.000000e+00> : tensor<1x4096xf32>
    %2 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1, %arg0 : tensor<1x4096xf32>, tensor<4096x4096xf32>) outs(%cst : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    
    %3 = tosa.const_shape  {values = dense<[1, 1, 4096]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.reshape %2, %3 : (tensor<1x4096xf32>, !tosa.shape<3>) -> tensor<1x1x4096xf32>
    %5 = tosa.const_shape  {values = dense<[1, 1, 32, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %6 = tosa.reshape %4, %5 : (tensor<1x1x4096xf32>, !tosa.shape<4>) -> tensor<1x1x32x128xf32>
    /////////////////////////////////////////////////
    %7 = tosa.const_shape  {values = dense<[1, 32, 1, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %8 = tosa.reshape %6, %7 : (tensor<1x1x32x128xf32>, !tosa.shape<4>) -> tensor<1x32x1x128xf32>
    
    %9 = tosa.const_shape  {values = dense<[1, 4096]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %10 = tosa.reshape %arg10, %9 : (tensor<1x1x4096xf32>, !tosa.shape<2>) -> tensor<1x4096xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x4096xf32>
    %11 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%10, %arg1 : tensor<1x4096xf32>, tensor<4096x4096xf32>) outs(%cst_0 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %12 = tosa.const_shape  {values = dense<[1, 1, 4096]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %13 = tosa.reshape %11, %12 : (tensor<1x4096xf32>, !tosa.shape<3>) -> tensor<1x1x4096xf32>
    %14 = tosa.const_shape  {values = dense<[1, 1, 32, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %15 = tosa.reshape %13, %14 : (tensor<1x1x4096xf32>, !tosa.shape<4>) -> tensor<1x1x32x128xf32>
    %16 = tosa.const_shape  {values = dense<[1, 32, 1, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %17 = tosa.reshape %15, %16 : (tensor<1x1x32x128xf32>, !tosa.shape<4>) -> tensor<1x32x1x128xf32>
    %18 = tosa.const_shape  {values = dense<[1, 4096]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %19 = tosa.reshape %arg10, %18 : (tensor<1x1x4096xf32>, !tosa.shape<2>) -> tensor<1x4096xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x4096xf32>
    %20 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%19, %arg2 : tensor<1x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %21 = tosa.const_shape  {values = dense<[1, 1, 4096]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %22 = tosa.reshape %20, %21 : (tensor<1x4096xf32>, !tosa.shape<3>) -> tensor<1x1x4096xf32>
    %23 = tosa.const_shape  {values = dense<[1, 1, 32, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %24 = tosa.reshape %22, %23 : (tensor<1x1x4096xf32>, !tosa.shape<4>) -> tensor<1x1x32x128xf32>
    %25 = tosa.const_shape  {values = dense<[1, 32, 1, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %26 = tosa.reshape %24, %25 : (tensor<1x1x32x128xf32>, !tosa.shape<4>) -> tensor<1x32x1x128xf32>
    %27 = tosa.const_shape  {values = dense<[1, 1, 1, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %28 = tosa.reshape %arg8, %27 : (tensor<1x1x128xf32>, !tosa.shape<4>) -> tensor<1x1x1x128xf32>
    %29 = tosa.const_shape  {values = dense<[1, 1, 1, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %30 = tosa.reshape %arg9, %29 : (tensor<1x1x128xf32>, !tosa.shape<4>) -> tensor<1x1x1x128xf32>
    %31 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %32 = tosa.mul %8, %28, %31 : (tensor<1x32x1x128xf32>, tensor<1x1x1x128xf32>, tensor<1xi8>) -> tensor<1x32x1x128xf32>
    %extracted_slice = tensor.extract_slice %8[0, 0, 0, 0] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<1x32x1x128xf32> to tensor<1x32x1x64xf32>
    %extracted_slice_2 = tensor.extract_slice %8[0, 0, 0, 64] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<1x32x1x128xf32> to tensor<1x32x1x64xf32>
    %33 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %34 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %35 = tosa.negate %extracted_slice_2, %33, %34 : (tensor<1x32x1x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x1x64xf32>
    %36 = tensor.empty() : tensor<1x32x1x128xf32>
    %inserted_slice = tensor.insert_slice %35 into %36[0, 0, 0, 0] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<1x32x1x64xf32> into tensor<1x32x1x128xf32>
    %inserted_slice_3 = tensor.insert_slice %extracted_slice into %inserted_slice[0, 0, 0, 64] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<1x32x1x64xf32> into tensor<1x32x1x128xf32>
    %37 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %38 = tosa.mul %inserted_slice_3, %30, %37 : (tensor<1x32x1x128xf32>, tensor<1x1x1x128xf32>, tensor<1xi8>) -> tensor<1x32x1x128xf32>
    %39 = tosa.add %32, %38 : (tensor<1x32x1x128xf32>, tensor<1x32x1x128xf32>) -> tensor<1x32x1x128xf32>
    %40 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %41 = tosa.mul %17, %28, %40 : (tensor<1x32x1x128xf32>, tensor<1x1x1x128xf32>, tensor<1xi8>) -> tensor<1x32x1x128xf32>
    %extracted_slice_4 = tensor.extract_slice %17[0, 0, 0, 0] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<1x32x1x128xf32> to tensor<1x32x1x64xf32>
    %extracted_slice_5 = tensor.extract_slice %17[0, 0, 0, 64] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<1x32x1x128xf32> to tensor<1x32x1x64xf32>
    %42 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %43 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %44 = tosa.negate %extracted_slice_5, %42, %43 : (tensor<1x32x1x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x1x64xf32>
    %45 = tensor.empty() : tensor<1x32x1x128xf32>
    %inserted_slice_6 = tensor.insert_slice %44 into %45[0, 0, 0, 0] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<1x32x1x64xf32> into tensor<1x32x1x128xf32>
    %inserted_slice_7 = tensor.insert_slice %extracted_slice_4 into %inserted_slice_6[0, 0, 0, 64] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<1x32x1x64xf32> into tensor<1x32x1x128xf32>
    %46 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %47 = tosa.mul %inserted_slice_7, %30, %46 : (tensor<1x32x1x128xf32>, tensor<1x1x1x128xf32>, tensor<1xi8>) -> tensor<1x32x1x128xf32>
    %48 = tosa.add %41, %47 : (tensor<1x32x1x128xf32>, tensor<1x32x1x128xf32>) -> tensor<1x32x1x128xf32>
    %49 = bufferization.to_buffer %arg5 : tensor<1x32x1024x128xf32> to memref<1x32x1024x128xf32>
    %50 = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
    %51 = tosa.reshape %arg4, %50 : (tensor<1xi64>, !tosa.shape<4>) -> tensor<1x1x1x1xi64>
    %52 = "tosa.const"() <{values = dense<0> : tensor<1x32x1x128xi64>}> : () -> tensor<1x32x1x128xi64>
    %53 = tosa.add %51, %52 : (tensor<1x1x1x1xi64>, tensor<1x32x1x128xi64>) -> tensor<1x32x1x128xi64>
    %54 = bufferization.to_buffer %53 : tensor<1x32x1x128xi64> to memref<1x32x1x128xi64>
    %55 = bufferization.to_buffer %48 : tensor<1x32x1x128xf32> to memref<1x32x1x128xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_8 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c1_9 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    scf.for %arg11 = %c0 to %c1_8 step %c1 {
      scf.for %arg12 = %c0 to %c32 step %c1 {
        scf.for %arg13 = %c0 to %c1_9 step %c1 {
          scf.for %arg14 = %c0 to %c128 step %c1 {
            %102 = memref.load %55[%arg11, %arg12, %arg13, %arg14] : memref<1x32x1x128xf32>
            %103 = memref.load %54[%arg11, %arg12, %arg13, %arg14] : memref<1x32x1x128xi64>
            %104 = arith.index_cast %103 : i64 to index
            memref.store %102, %49[%arg11, %arg12, %104, %arg14] : memref<1x32x1024x128xf32>
          }
        }
      }
    }
    %56 = bufferization.to_tensor %49 restrict : memref<1x32x1024x128xf32> to tensor<1x32x1024x128xf32>
    %57 = bufferization.to_buffer %arg6 : tensor<1x32x1024x128xf32> to memref<1x32x1024x128xf32>
    %58 = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
    %59 = tosa.reshape %arg4, %58 : (tensor<1xi64>, !tosa.shape<4>) -> tensor<1x1x1x1xi64>
    %60 = "tosa.const"() <{values = dense<0> : tensor<1x32x1x128xi64>}> : () -> tensor<1x32x1x128xi64>
    %61 = tosa.add %59, %60 : (tensor<1x1x1x1xi64>, tensor<1x32x1x128xi64>) -> tensor<1x32x1x128xi64>
    %62 = bufferization.to_buffer %61 : tensor<1x32x1x128xi64> to memref<1x32x1x128xi64>
    %63 = bufferization.to_buffer %26 : tensor<1x32x1x128xf32> to memref<1x32x1x128xf32>
    %c0_10 = arith.constant 0 : index
    %c1_11 = arith.constant 1 : index
    %c1_12 = arith.constant 1 : index
    %c32_13 = arith.constant 32 : index
    %c1_14 = arith.constant 1 : index
    %c128_15 = arith.constant 128 : index
    scf.for %arg11 = %c0_10 to %c1_12 step %c1_11 {
      scf.for %arg12 = %c0_10 to %c32_13 step %c1_11 {
        scf.for %arg13 = %c0_10 to %c1_14 step %c1_11 {
          scf.for %arg14 = %c0_10 to %c128_15 step %c1_11 {
            %102 = memref.load %63[%arg11, %arg12, %arg13, %arg14] : memref<1x32x1x128xf32>
            %103 = memref.load %62[%arg11, %arg12, %arg13, %arg14] : memref<1x32x1x128xi64>
            %104 = arith.index_cast %103 : i64 to index
            memref.store %102, %57[%arg11, %arg12, %104, %arg14] : memref<1x32x1024x128xf32>
          }
        }
      }
    }
    %64 = bufferization.to_tensor %57 restrict : memref<1x32x1024x128xf32> to tensor<1x32x1024x128xf32>
    %cst_16 = arith.constant 0xFF800000 : f32
    %cst_17 = arith.constant 0.000000e+00 : f32
    %65 = tensor.empty() : tensor<1x1x1x1024xf32>
    %splat = tensor.splat %cst_17 : tensor<1x1x1x1024xf32>
    %splat_18 = tensor.splat %cst_16 : tensor<1x1x1x1024xf32>
    %66 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg7, %splat, %splat_18 : tensor<1x1x1x1024xi1>, tensor<1x1x1x1024xf32>, tensor<1x1x1x1024xf32>) outs(%65 : tensor<1x1x1x1024xf32>) {
    ^bb0(%in: i1, %in_25: f32, %in_26: f32, %out: f32):
      %102 = arith.select %in, %in_25, %in_26 : f32
      linalg.yield %102 : f32
    } -> tensor<1x1x1x1024xf32>
    %cst_19 = arith.constant 0.000000e+00 : f32
    %splat_20 = tensor.splat %cst_19 : tensor<1x1024xf32>
    %67 = tosa.const_shape  {values = dense<[1, 1024]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %68 = tosa.reshape %66, %67 : (tensor<1x1x1x1024xf32>, !tosa.shape<2>) -> tensor<1x1024xf32>
    %69 = tosa.add %splat_20, %68 : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %70 = tosa.const_shape  {values = dense<[32, 1, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %71 = tosa.reshape %39, %70 : (tensor<1x32x1x128xf32>, !tosa.shape<3>) -> tensor<32x1x128xf32>
    %72 = tosa.const_shape  {values = dense<[32, 1024, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %73 = tosa.reshape %56, %72 : (tensor<1x32x1024x128xf32>, !tosa.shape<3>) -> tensor<32x1024x128xf32>
    %cst_21 = arith.constant dense<0.000000e+00> : tensor<32x1x1024xf32>
    %74 = linalg.batch_matmul_transpose_b ins(%71, %73 : tensor<32x1x128xf32>, tensor<32x1024x128xf32>) outs(%cst_21 : tensor<32x1x1024xf32>) -> tensor<32x1x1024xf32>
    %cst_22 = arith.constant 0.0883883461 : f32
    %splat_23 = tensor.splat %cst_22 : tensor<32x1x1024xf32>
    %75 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %76 = tosa.mul %74, %splat_23, %75 : (tensor<32x1x1024xf32>, tensor<32x1x1024xf32>, tensor<1xi8>) -> tensor<32x1x1024xf32>
    %77 = tosa.const_shape  {values = dense<[1, 1, 1024]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %78 = tosa.reshape %69, %77 : (tensor<1x1024xf32>, !tosa.shape<3>) -> tensor<1x1x1024xf32>
    %79 = tosa.add %76, %78 : (tensor<32x1x1024xf32>, tensor<1x1x1024xf32>) -> tensor<32x1x1024xf32>
    %80 = tosa.reduce_max %79 {axis = 2 : i32} : (tensor<32x1x1024xf32>) -> tensor<32x1x1xf32>
    %81 = tosa.sub %79, %80 : (tensor<32x1x1024xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1024xf32>
    %82 = math.exp %81 : tensor<32x1x1024xf32>
    %83 = tosa.reduce_sum %82 {axis = 2 : i32} : (tensor<32x1x1024xf32>) -> tensor<32x1x1xf32>
    %84 = tosa.log %83 : (tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
    %85 = tosa.add %80, %84 : (tensor<32x1x1xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
    %86 = tosa.sub %79, %85 : (tensor<32x1x1024xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1024xf32>
    %87 = math.exp %86 : tensor<32x1x1024xf32>
    %88 = tosa.const_shape  {values = dense<[1, 32, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %89 = tosa.reshape %85, %88 : (tensor<32x1x1xf32>, !tosa.shape<3>) -> tensor<1x32x1xf32>
    %90 = tosa.const_shape  {values = dense<[32, 1024, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %91 = tosa.reshape %64, %90 : (tensor<1x32x1024x128xf32>, !tosa.shape<3>) -> tensor<32x1024x128xf32>
    %92 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %93 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %94 = tosa.matmul %87, %91, %92, %93 : (tensor<32x1x1024xf32>, tensor<32x1024x128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<32x1x128xf32>
    %95 = tosa.const_shape  {values = dense<[1, 32, 1, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %96 = tosa.reshape %94, %95 : (tensor<32x1x128xf32>, !tosa.shape<4>) -> tensor<1x32x1x128xf32>
    %97 = tosa.const_shape  {values = dense<[1, 1, 4096]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %98 = tosa.reshape %96, %97 : (tensor<1x32x1x128xf32>, !tosa.shape<3>) -> tensor<1x1x4096xf32>
    %99 = tosa.const_shape  {values = dense<[1, 4096]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %100 = tosa.reshape %98, %99 : (tensor<1x1x4096xf32>, !tosa.shape<2>) -> tensor<1x4096xf32>
    %cst_24 = arith.constant dense<0.000000e+00> : tensor<1x4096xf32>
    %101 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%100, %arg3 : tensor<1x4096xf32>, tensor<4096x4096xf32>) outs(%cst_24 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    return %56, %64, %101 : tensor<1x32x1024x128xf32>, tensor<1x32x1024x128xf32>, tensor<1x4096xf32>
  }
}

