#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @subgraph0_decode163(%arg0: tensor<4096xf32>, %arg1: tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32> {
    %0 = tensor.empty() : tensor<1x1x4096xf32>
    %c2_i32 = arith.constant 2 : i32
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x1x4096xf32>) outs(%0 : tensor<1x1x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %18 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %18 : f32
    } -> tensor<1x1x4096xf32>
    %2 = tosa.reduce_sum %1 {axis = 2 : i32} : (tensor<1x1x4096xf32>) -> tensor<1x1x1xf32>
    %3 = "tosa.const"() <{values = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4 = tosa.reciprocal %3 : (tensor<1xf32>) -> tensor<1xf32>
    %5 = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
    %6 = tosa.reshape %4, %5 : (tensor<1xf32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
    %7 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %8 = tosa.mul %6, %2, %7 : (tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<1x1x1xf32>
    %9 = "tosa.const"() <{values = dense<9.99999974E-6> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
    %10 = tosa.add %8, %9 : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %11 = tosa.rsqrt %10 : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %12 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %13 = tosa.mul %arg1, %11, %12 : (tensor<1x1x4096xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<1x1x4096xf32>
    %14 = tosa.const_shape  {values = dense<[1, 1, 4096]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %15 = tosa.reshape %arg0, %14 : (tensor<4096xf32>, !tosa.shape<3>) -> tensor<1x1x4096xf32>
    %16 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %17 = tosa.mul %15, %13, %16 : (tensor<1x1x4096xf32>, tensor<1x1x4096xf32>, tensor<1xi8>) -> tensor<1x1x4096xf32>
    return %17 : tensor<1x1x4096xf32>
  }
}

