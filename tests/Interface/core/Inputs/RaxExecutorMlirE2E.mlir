module {
  func.func private @subgraph0(%input: memref<4xf32>, %factor: f32,
                               %scratch: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %input_value = memref.load %input[%i] : memref<4xf32>
      %scratch_value = arith.mulf %input_value, %factor : f32
      memref.store %scratch_value, %scratch[%i] : memref<4xf32>
    }
    return
  }

  func.func private @subgraph1(%scratch: memref<4xf32>, %bias: f32,
                               %output: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %scratch_value = memref.load %scratch[%i] : memref<4xf32>
      %output_value = arith.addf %scratch_value, %bias : f32
      memref.store %output_value, %output[%i] : memref<4xf32>
    }
    return
  }

  func.func @forward0(%params: memref<2xf32>, %input: memref<4xf32>,
                      %scratch: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %factor = memref.load %params[%c0] : memref<2xf32>
    func.call @subgraph0(%input, %factor, %scratch)
        : (memref<4xf32>, f32, memref<4xf32>) -> ()
    return
  }

  func.func @forward1(%params: memref<2xf32>, %scratch: memref<4xf32>,
                      %output: memref<4xf32>) {
    %c1 = arith.constant 1 : index
    %bias = memref.load %params[%c1] : memref<2xf32>
    func.call @subgraph1(%scratch, %bias, %output)
        : (memref<4xf32>, f32, memref<4xf32>) -> ()
    return
  }
}
