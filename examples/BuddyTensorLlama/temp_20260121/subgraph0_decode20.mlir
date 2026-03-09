#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
"builtin.module"() ({
  "func.func"() <{function_type = (tensor<4096x2048xf32>, tensor<4096x2048xf32>, tensor<4096x2048xf32>, tensor<2048x4096xf32>, tensor<1xi64>, tensor<1x16x1024x128xf32>, tensor<1x16x1024x128xf32>, tensor<1x1x1x1024xi1>, tensor<1x1x128xf32>, tensor<1x1x128xf32>, tensor<1x1x4096xf32>) -> (tensor<1x16x1024x128xf32>, tensor<1x16x1024x128xf32>, tensor<1x4096xf32>), sym_name = "subgraph0_decode20"}> ({
  ^bb0(%arg0: tensor<4096x2048xf32>, %arg1: tensor<4096x2048xf32>, %arg2: tensor<4096x2048xf32>, %arg3: tensor<2048x4096xf32>, %arg4: tensor<1xi64>, %arg5: tensor<1x16x1024x128xf32>, %arg6: tensor<1x16x1024x128xf32>, %arg7: tensor<1x1x1x1024xi1>, %arg8: tensor<1x1x128xf32>, %arg9: tensor<1x1x128xf32>, %arg10: tensor<1x1x4096xf32>):
    %0 = "tosa.const_shape"() <{values = dense<[1, 4096]> : tensor<2xindex>}> : () -> !tosa.shape<2>
    %1 = "tosa.reshape"(%arg10, %0) : (tensor<1x1x4096xf32>, !tosa.shape<2>) -> tensor<1x4096xf32>
    %2 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1x2048xf32>}> : () -> tensor<1x2048xf32>
    %3 = "linalg.matmul"(%1, %arg0, %2) <{cast = #linalg.type_fn<cast_signed>, indexing_maps = [#map, #map1, #map2], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg35: f32, %arg36: f32, %arg37: f32):
      %144 = "arith.mulf"(%arg35, %arg36) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %145 = "arith.addf"(%arg37, %144) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%145) : (f32) -> ()
    }) : (tensor<1x4096xf32>, tensor<4096x2048xf32>, tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %4 = "tosa.const_shape"() <{values = dense<[1, 1, 2048]> : tensor<3xindex>}> : () -> !tosa.shape<3>
    %5 = "tosa.reshape"(%3, %4) : (tensor<1x2048xf32>, !tosa.shape<3>) -> tensor<1x1x2048xf32>
    %6 = "tosa.const_shape"() <{values = dense<[1, 1, 16, 128]> : tensor<4xindex>}> : () -> !tosa.shape<4>
    %7 = "tosa.reshape"(%5, %6) : (tensor<1x1x2048xf32>, !tosa.shape<4>) -> tensor<1x1x16x128xf32>
    %8 = "tosa.const_shape"() <{values = dense<[1, 4096]> : tensor<2xindex>}> : () -> !tosa.shape<2>
    %9 = "tosa.reshape"(%arg10, %8) : (tensor<1x1x4096xf32>, !tosa.shape<2>) -> tensor<1x4096xf32>
    %10 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1x2048xf32>}> : () -> tensor<1x2048xf32>
    %11 = "linalg.matmul"(%9, %arg1, %10) <{cast = #linalg.type_fn<cast_signed>, indexing_maps = [#map, #map1, #map2], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg32: f32, %arg33: f32, %arg34: f32):
      %142 = "arith.mulf"(%arg32, %arg33) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %143 = "arith.addf"(%arg34, %142) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%143) : (f32) -> ()
    }) : (tensor<1x4096xf32>, tensor<4096x2048xf32>, tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %12 = "tosa.const_shape"() <{values = dense<[1, 1, 2048]> : tensor<3xindex>}> : () -> !tosa.shape<3>
    %13 = "tosa.reshape"(%11, %12) : (tensor<1x2048xf32>, !tosa.shape<3>) -> tensor<1x1x2048xf32>
    %14 = "tosa.const_shape"() <{values = dense<[1, 1, 16, 128]> : tensor<4xindex>}> : () -> !tosa.shape<4>
    %15 = "tosa.reshape"(%13, %14) : (tensor<1x1x2048xf32>, !tosa.shape<4>) -> tensor<1x1x16x128xf32>
    %16 = "tosa.const_shape"() <{values = dense<[1, 4096]> : tensor<2xindex>}> : () -> !tosa.shape<2>
    %17 = "tosa.reshape"(%arg10, %16) : (tensor<1x1x4096xf32>, !tosa.shape<2>) -> tensor<1x4096xf32>
    %18 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1x2048xf32>}> : () -> tensor<1x2048xf32>
    %19 = "linalg.matmul"(%17, %arg2, %18) <{cast = #linalg.type_fn<cast_signed>, indexing_maps = [#map, #map1, #map2], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg29: f32, %arg30: f32, %arg31: f32):
      %140 = "arith.mulf"(%arg29, %arg30) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %141 = "arith.addf"(%arg31, %140) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%141) : (f32) -> ()
    }) : (tensor<1x4096xf32>, tensor<4096x2048xf32>, tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %20 = "tosa.const_shape"() <{values = dense<[1, 1, 2048]> : tensor<3xindex>}> : () -> !tosa.shape<3>
    %21 = "tosa.reshape"(%19, %20) : (tensor<1x2048xf32>, !tosa.shape<3>) -> tensor<1x1x2048xf32>
    %22 = "tosa.const_shape"() <{values = dense<[1, 1, 16, 128]> : tensor<4xindex>}> : () -> !tosa.shape<4>
    %23 = "tosa.reshape"(%21, %22) : (tensor<1x1x2048xf32>, !tosa.shape<4>) -> tensor<1x1x16x128xf32>
    %24 = "tosa.const_shape"() <{values = dense<[1, 1, 1, 128]> : tensor<4xindex>}> : () -> !tosa.shape<4>
    %25 = "tosa.reshape"(%arg8, %24) : (tensor<1x1x128xf32>, !tosa.shape<4>) -> tensor<1x1x1x128xf32>
    %26 = "tosa.const_shape"() <{values = dense<[1, 1, 1, 128]> : tensor<4xindex>}> : () -> !tosa.shape<4>
    %27 = "tosa.reshape"(%arg9, %26) : (tensor<1x1x128xf32>, !tosa.shape<4>) -> tensor<1x1x1x128xf32>
    %28 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %29 = "tosa.mul"(%7, %25, %28) : (tensor<1x1x16x128xf32>, tensor<1x1x1x128xf32>, tensor<1xi8>) -> tensor<1x1x16x128xf32>
    %30 = "tensor.extract_slice"(%7) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 16, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1x16x128xf32>) -> tensor<1x1x16x64xf32>
    %31 = "tensor.extract_slice"(%7) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 1, 16, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1x16x128xf32>) -> tensor<1x1x16x64xf32>
    %32 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %33 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %34 = "tosa.negate"(%31, %32, %33) : (tensor<1x1x16x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x16x64xf32>
    %35 = "tensor.empty"() : () -> tensor<1x1x16x128xf32>
    %36 = "tensor.insert_slice"(%34, %35) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 16, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1x16x64xf32>, tensor<1x1x16x128xf32>) -> tensor<1x1x16x128xf32>
    %37 = "tensor.insert_slice"(%30, %36) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 1, 16, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1x16x64xf32>, tensor<1x1x16x128xf32>) -> tensor<1x1x16x128xf32>
    %38 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %39 = "tosa.mul"(%37, %27, %38) : (tensor<1x1x16x128xf32>, tensor<1x1x1x128xf32>, tensor<1xi8>) -> tensor<1x1x16x128xf32>
    %40 = "tosa.add"(%29, %39) : (tensor<1x1x16x128xf32>, tensor<1x1x16x128xf32>) -> tensor<1x1x16x128xf32>
    %41 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %42 = "tosa.mul"(%15, %25, %41) : (tensor<1x1x16x128xf32>, tensor<1x1x1x128xf32>, tensor<1xi8>) -> tensor<1x1x16x128xf32>
    %43 = "tensor.extract_slice"(%15) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 16, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1x16x128xf32>) -> tensor<1x1x16x64xf32>
    %44 = "tensor.extract_slice"(%15) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 1, 16, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1x16x128xf32>) -> tensor<1x1x16x64xf32>
    %45 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %46 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %47 = "tosa.negate"(%44, %45, %46) : (tensor<1x1x16x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x16x64xf32>
    %48 = "tensor.empty"() : () -> tensor<1x1x16x128xf32>
    %49 = "tensor.insert_slice"(%47, %48) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 16, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1x16x64xf32>, tensor<1x1x16x128xf32>) -> tensor<1x1x16x128xf32>
    %50 = "tensor.insert_slice"(%43, %49) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 1, 16, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1x16x64xf32>, tensor<1x1x16x128xf32>) -> tensor<1x1x16x128xf32>
    %51 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %52 = "tosa.mul"(%50, %27, %51) : (tensor<1x1x16x128xf32>, tensor<1x1x1x128xf32>, tensor<1xi8>) -> tensor<1x1x16x128xf32>
    %53 = "tosa.add"(%42, %52) : (tensor<1x1x16x128xf32>, tensor<1x1x16x128xf32>) -> tensor<1x1x16x128xf32>
    %54 = "bufferization.to_buffer"(%arg5) : (tensor<1x16x1024x128xf32>) -> memref<1x16x1024x128xf32>
    %55 = "tosa.const_shape"() <{values = dense<1> : tensor<4xindex>}> : () -> !tosa.shape<4>
    %56 = "tosa.reshape"(%arg4, %55) : (tensor<1xi64>, !tosa.shape<4>) -> tensor<1x1x1x1xi64>
    %57 = "tosa.const"() <{values = dense<0> : tensor<1x1x16x128xi64>}> : () -> tensor<1x1x16x128xi64>
    %58 = "tosa.add"(%56, %57) : (tensor<1x1x1x1xi64>, tensor<1x1x16x128xi64>) -> tensor<1x1x16x128xi64>
    %59 = "bufferization.to_buffer"(%58) : (tensor<1x1x16x128xi64>) -> memref<1x1x16x128xi64>
    %60 = "bufferization.to_buffer"(%53) : (tensor<1x1x16x128xf32>) -> memref<1x1x16x128xf32>
    %61 = "arith.constant"() <{value = 0 : index}> : () -> index
    %62 = "arith.constant"() <{value = 1 : index}> : () -> index
    %63 = "arith.constant"() <{value = 1 : index}> : () -> index
    %64 = "arith.constant"() <{value = 1 : index}> : () -> index
    %65 = "arith.constant"() <{value = 16 : index}> : () -> index
    %66 = "arith.constant"() <{value = 128 : index}> : () -> index
    "scf.for"(%61, %63, %62) ({
    ^bb0(%arg25: index):
      "scf.for"(%61, %64, %62) ({
      ^bb0(%arg26: index):
        "scf.for"(%61, %65, %62) ({
        ^bb0(%arg27: index):
          "scf.for"(%61, %66, %62) ({
          ^bb0(%arg28: index):
            %137 = "memref.load"(%60, %arg25, %arg26, %arg27, %arg28) : (memref<1x1x16x128xf32>, index, index, index, index) -> f32
            %138 = "memref.load"(%59, %arg25, %arg26, %arg27, %arg28) : (memref<1x1x16x128xi64>, index, index, index, index) -> i64
            %139 = "arith.index_cast"(%138) : (i64) -> index
            "memref.store"(%137, %54, %arg25, %arg26, %139, %arg28) : (f32, memref<1x16x1024x128xf32>, index, index, index, index) -> ()
            "scf.yield"() : () -> ()
          }) : (index, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %67 = "bufferization.to_tensor"(%54) <{restrict}> : (memref<1x16x1024x128xf32>) -> tensor<1x16x1024x128xf32>
    %68 = "bufferization.to_buffer"(%arg6) : (tensor<1x16x1024x128xf32>) -> memref<1x16x1024x128xf32>
    %69 = "tosa.const_shape"() <{values = dense<1> : tensor<4xindex>}> : () -> !tosa.shape<4>
    %70 = "tosa.reshape"(%arg4, %69) : (tensor<1xi64>, !tosa.shape<4>) -> tensor<1x1x1x1xi64>
    %71 = "tosa.const"() <{values = dense<0> : tensor<1x1x16x128xi64>}> : () -> tensor<1x1x16x128xi64>
    %72 = "tosa.add"(%70, %71) : (tensor<1x1x1x1xi64>, tensor<1x1x16x128xi64>) -> tensor<1x1x16x128xi64>
    %73 = "bufferization.to_buffer"(%72) : (tensor<1x1x16x128xi64>) -> memref<1x1x16x128xi64>
    %74 = "bufferization.to_buffer"(%23) : (tensor<1x1x16x128xf32>) -> memref<1x1x16x128xf32>
    %75 = "arith.constant"() <{value = 0 : index}> : () -> index
    %76 = "arith.constant"() <{value = 1 : index}> : () -> index
    %77 = "arith.constant"() <{value = 1 : index}> : () -> index
    %78 = "arith.constant"() <{value = 1 : index}> : () -> index
    %79 = "arith.constant"() <{value = 16 : index}> : () -> index
    %80 = "arith.constant"() <{value = 128 : index}> : () -> index
    "scf.for"(%75, %77, %76) ({
    ^bb0(%arg21: index):
      "scf.for"(%75, %78, %76) ({
      ^bb0(%arg22: index):
        "scf.for"(%75, %79, %76) ({
        ^bb0(%arg23: index):
          "scf.for"(%75, %80, %76) ({
          ^bb0(%arg24: index):
            %134 = "memref.load"(%74, %arg21, %arg22, %arg23, %arg24) : (memref<1x1x16x128xf32>, index, index, index, index) -> f32
            %135 = "memref.load"(%73, %arg21, %arg22, %arg23, %arg24) : (memref<1x1x16x128xi64>, index, index, index, index) -> i64
            %136 = "arith.index_cast"(%135) : (i64) -> index
            "memref.store"(%134, %68, %arg21, %arg22, %136, %arg24) : (f32, memref<1x16x1024x128xf32>, index, index, index, index) -> ()
            "scf.yield"() : () -> ()
          }) : (index, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %81 = "bufferization.to_tensor"(%68) <{restrict}> : (memref<1x16x1024x128xf32>) -> tensor<1x16x1024x128xf32>
    %82 = "arith.constant"() <{value = 0xFF800000 : f32}> : () -> f32
    %83 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %84 = "tensor.empty"() : () -> tensor<1x1x1x1024xf32>
    %85 = "tensor.splat"(%83) : (f32) -> tensor<1x1x1x1024xf32>
    %86 = "tensor.splat"(%82) : (f32) -> tensor<1x1x1x1024xf32>
    %87 = "linalg.generic"(%arg7, %85, %86, %84) <{indexing_maps = [#map3, #map3, #map3, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 3, 1>}> ({
    ^bb0(%arg17: i1, %arg18: f32, %arg19: f32, %arg20: f32):
      %133 = "arith.select"(%arg17, %arg18, %arg19) : (i1, f32, f32) -> f32
      "linalg.yield"(%133) : (f32) -> ()
    }) : (tensor<1x1x1x1024xi1>, tensor<1x1x1x1024xf32>, tensor<1x1x1x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<1x1x1x1024xf32>
    %88 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %89 = "tensor.splat"(%88) : (f32) -> tensor<16x1024xf32>
    %90 = "tosa.const_shape"() <{values = dense<[16, 1024]> : tensor<2xindex>}> : () -> !tosa.shape<2>
    %91 = "tosa.reshape"(%87, %90) : (tensor<1x1x1x1024xf32>, !tosa.shape<2>) -> tensor<16x1024xf32>
    %92 = "tosa.add"(%89, %91) : (tensor<16x1024xf32>, tensor<16x1024xf32>) -> tensor<16x1024xf32>
    %93 = "tosa.const_shape"() <{values = dense<[1, 16, 128]> : tensor<3xindex>}> : () -> !tosa.shape<3>
    %94 = "tosa.reshape"(%40, %93) : (tensor<1x1x16x128xf32>, !tosa.shape<3>) -> tensor<1x16x128xf32>
    %95 = "tosa.const_shape"() <{values = dense<[16, 1024, 128]> : tensor<3xindex>}> : () -> !tosa.shape<3>
    %96 = "tosa.reshape"(%67, %95) : (tensor<1x16x1024x128xf32>, !tosa.shape<3>) -> tensor<16x1024x128xf32>
    %97 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<16x16x1024xf32>}> : () -> tensor<16x16x1024xf32>
    %98 = "linalg.batch_matmul_transpose_b"(%94, %96, %97) <{operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg14: f32, %arg15: f32, %arg16: f32):
      %131 = "arith.mulf"(%arg14, %arg15) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %132 = "arith.addf"(%arg16, %131) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%132) : (f32) -> ()
    }) : (tensor<1x16x128xf32>, tensor<16x1024x128xf32>, tensor<16x16x1024xf32>) -> tensor<16x16x1024xf32>
    %99 = "arith.constant"() <{value = 0.0883883461 : f32}> : () -> f32
    %100 = "tensor.splat"(%99) : (f32) -> tensor<16x16x1024xf32>
    %101 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %102 = "tosa.mul"(%98, %100, %101) : (tensor<16x16x1024xf32>, tensor<16x16x1024xf32>, tensor<1xi8>) -> tensor<16x16x1024xf32>
    %103 = "tosa.const_shape"() <{values = dense<[1, 16, 1024]> : tensor<3xindex>}> : () -> !tosa.shape<3>
    %104 = "tosa.reshape"(%92, %103) : (tensor<16x1024xf32>, !tosa.shape<3>) -> tensor<1x16x1024xf32>
    %105 = "tosa.add"(%102, %104) : (tensor<16x16x1024xf32>, tensor<1x16x1024xf32>) -> tensor<16x16x1024xf32>
    %106 = "tosa.reduce_max"(%105) <{axis = 2 : i32, nan_mode = "PROPAGATE"}> : (tensor<16x16x1024xf32>) -> tensor<16x16x1xf32>
    %107 = "tosa.sub"(%105, %106) : (tensor<16x16x1024xf32>, tensor<16x16x1xf32>) -> tensor<16x16x1024xf32>
    %108 = "math.exp"(%107) <{fastmath = #arith.fastmath<none>}> : (tensor<16x16x1024xf32>) -> tensor<16x16x1024xf32>
    %109 = "tosa.reduce_sum"(%108) <{axis = 2 : i32}> : (tensor<16x16x1024xf32>) -> tensor<16x16x1xf32>
    %110 = "tosa.log"(%109) : (tensor<16x16x1xf32>) -> tensor<16x16x1xf32>
    %111 = "tosa.add"(%106, %110) : (tensor<16x16x1xf32>, tensor<16x16x1xf32>) -> tensor<16x16x1xf32>
    %112 = "tosa.sub"(%105, %111) : (tensor<16x16x1024xf32>, tensor<16x16x1xf32>) -> tensor<16x16x1024xf32>
    %113 = "math.exp"(%112) <{fastmath = #arith.fastmath<none>}> : (tensor<16x16x1024xf32>) -> tensor<16x16x1024xf32>
    %114 = "tosa.const_shape"() <{values = dense<[1, 1, 16]> : tensor<3xindex>}> : () -> !tosa.shape<3>
    %115 = "tosa.reshape"(%111, %114) : (tensor<16x16x1xf32>, !tosa.shape<3>) -> tensor<1x1x16xf32>
    %116 = "tosa.const_shape"() <{values = dense<[16, 1024, 128]> : tensor<3xindex>}> : () -> !tosa.shape<3>
    %117 = "tosa.reshape"(%81, %116) : (tensor<1x16x1024x128xf32>, !tosa.shape<3>) -> tensor<16x1024x128xf32>
    %118 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %119 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %120 = "tosa.matmul"(%113, %117, %118, %119) : (tensor<16x16x1024xf32>, tensor<16x1024x128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<16x16x128xf32>
    %121 = "tosa.const_shape"() <{values = dense<[1, 16, 16, 128]> : tensor<4xindex>}> : () -> !tosa.shape<4>
    %122 = "tosa.reshape"(%120, %121) : (tensor<16x16x128xf32>, !tosa.shape<4>) -> tensor<1x16x16x128xf32>
    %123 = "tosa.const_shape"() <{values = dense<[1, 1, 2048]> : tensor<3xindex>}> : () -> !tosa.shape<3>
    %124 = "tosa.reshape"(%122, %123) : (tensor<1x16x16x128xf32>, !tosa.shape<3>) -> tensor<1x1x2048xf32>
    %125 = "tosa.const_shape"() <{values = dense<[1, 2048]> : tensor<2xindex>}> : () -> !tosa.shape<2>
    %126 = "tosa.reshape"(%124, %125) : (tensor<1x1x2048xf32>, !tosa.shape<2>) -> tensor<1x2048xf32>
    %127 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1x4096xf32>}> : () -> tensor<1x4096xf32>
    %128 = "linalg.matmul"(%126, %arg3, %127) <{cast = #linalg.type_fn<cast_signed>, indexing_maps = [#map, #map1, #map2], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg11: f32, %arg12: f32, %arg13: f32):
      %129 = "arith.mulf"(%arg11, %arg12) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %130 = "arith.addf"(%arg13, %129) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%130) : (f32) -> ()
    }) : (tensor<1x2048xf32>, tensor<2048x4096xf32>, tensor<1x4096xf32>) -> tensor<1x4096xf32>
    "func.return"(%67, %81, %128) : (tensor<1x16x1024x128xf32>, tensor<1x16x1024x128xf32>, tensor<1x4096xf32>) -> ()
  }) : () -> ()
}) : () -> ()

