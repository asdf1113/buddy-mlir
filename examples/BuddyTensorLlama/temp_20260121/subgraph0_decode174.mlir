module {
  func.func @subgraph0_decode174(%arg0: tensor<1x1x4096xf32>, %arg1: tensor<1x4096xf32>) -> tensor<1x1x4096xf32> {
    %0 = tosa.const_shape  {values = dense<[1, 1, 4096]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.reshape %arg1, %0 : (tensor<1x4096xf32>, !tosa.shape<3>) -> tensor<1x1x4096xf32>
    %2 = tosa.add %arg0, %1 : (tensor<1x1x4096xf32>, tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    return %2 : tensor<1x1x4096xf32>
  }
}

