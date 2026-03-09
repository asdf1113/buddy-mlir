module {
  func.func @subgraph0_decode71(%arg0: tensor<4096x5504xf32>, %arg1: tensor<4096x5504xf32>, %arg2: tensor<5504x4096xf32>, %arg3: tensor<1x1x4096xf32>) -> tensor<1x4096xf32> {
    %0 = tosa.const_shape  {values = dense<[1, 4096]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.reshape %arg3, %0 : (tensor<1x1x4096xf32>, !tosa.shape<2>) -> tensor<1x4096xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x5504xf32>
    %2 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1, %arg0 : tensor<1x4096xf32>, tensor<4096x5504xf32>) outs(%cst : tensor<1x5504xf32>) -> tensor<1x5504xf32>
    %3 = tosa.const_shape  {values = dense<[1, 1, 5504]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.reshape %2, %3 : (tensor<1x5504xf32>, !tosa.shape<3>) -> tensor<1x1x5504xf32>
    %5 = tosa.sigmoid %4 : (tensor<1x1x5504xf32>) -> tensor<1x1x5504xf32>
    %6 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = tosa.mul %4, %5, %6 : (tensor<1x1x5504xf32>, tensor<1x1x5504xf32>, tensor<1xi8>) -> tensor<1x1x5504xf32>
    %8 = tosa.const_shape  {values = dense<[1, 4096]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %9 = tosa.reshape %arg3, %8 : (tensor<1x1x4096xf32>, !tosa.shape<2>) -> tensor<1x4096xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x5504xf32>
    %10 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%9, %arg1 : tensor<1x4096xf32>, tensor<4096x5504xf32>) outs(%cst_0 : tensor<1x5504xf32>) -> tensor<1x5504xf32>
    %11 = tosa.const_shape  {values = dense<[1, 1, 5504]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %12 = tosa.reshape %10, %11 : (tensor<1x5504xf32>, !tosa.shape<3>) -> tensor<1x1x5504xf32>
    %13 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %14 = tosa.mul %7, %12, %13 : (tensor<1x1x5504xf32>, tensor<1x1x5504xf32>, tensor<1xi8>) -> tensor<1x1x5504xf32>
    %15 = tosa.const_shape  {values = dense<[1, 5504]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %16 = tosa.reshape %14, %15 : (tensor<1x1x5504xf32>, !tosa.shape<2>) -> tensor<1x5504xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x4096xf32>
    %17 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%16, %arg2 : tensor<1x5504xf32>, tensor<5504x4096xf32>) outs(%cst_1 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    return %17 : tensor<1x4096xf32>
  }
}

