module {
  func.func private @rtclock() -> f64

  // 单行 + vector<32xf32> + vector.reduction <add>
  //
  // src0 : [N, K] = [4480, 1536]
  // src1 : [1, K] = [1, 1536]
  // dst  : [1, N] = [1, 4480]
  //
  // 语义：
  //   for n in 0..N-1:
  //     dst[0, n] = dot(src0[n, :], src1[0, :])
  //
  // 假设：
  //   K 是 32 的倍数（本例 K=1536 成立）
  func.func @ggml_matmul_rowwise_v32(
      %src0: memref<?x?xf32>,   // [N, K]
      %src1: memref<1x?xf32>,   // [1, K]
      %dst:  memref<1x?xf32>) { // [1, N]

    %t0 = call @rtclock() : () -> f64

    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c32 = arith.constant 32 : index

    %f0 = arith.constant 0.000000e+00 : f32
    %vzero = vector.splat %f0 : vector<32xf32>

    %N = memref.dim %src0, %c0 : memref<?x?xf32>
    %K = memref.dim %src0, %c1 : memref<?x?xf32>

    // 外层并行遍历 src0 的每一行（每个输出通道）
    affine.parallel (%n) = (0) to (%N) {
      affine.prefetch %src0[%n, %c0], read, locality<3>, data : memref<?x?xf32>
      affine.prefetch %src1[%c0, %c0], read, locality<3>, data : memref<1x?xf32>

      %vacc = scf.for %k0 = %c0 to %K step %c32
          iter_args(%acc = %vzero) -> (vector<32xf32>) {
        %w = vector.load %src0[%n, %k0] : memref<?x?xf32>, vector<32xf32>
        %x = vector.load %src1[%c0, %k0] : memref<1x?xf32>, vector<32xf32>
        %acc1 = vector.fma %w, %x, %acc : vector<32xf32>
        scf.yield %acc1 : vector<32xf32>
      }

      %sum = vector.reduction <add>, %vacc : vector<32xf32> into f32
      memref.store %sum, %dst[%c0, %n] : memref<1x?xf32>
    }

    %t1 = call @rtclock() : () -> f64
    %dt = arith.subf %t1, %t0 : f64
    vector.print %dt : f64
    return
  }

  func.func @main() {
    %c1536 = arith.constant 1536 : index
    %c4480 = arith.constant 4480 : index

    %f0 = arith.constant 0.000000e+00 : f32
    %f2 = arith.constant 2.000000e+00 : f32
    %f3 = arith.constant 3.000000e+00 : f32

    // src0 = weight [N, K] = [4480, 1536]
    %src0 = memref.alloc(%c4480, %c1536) : memref<?x?xf32>

    // src1 = input [1, K] = [1, 1536]
    %src1 = memref.alloc(%c1536) : memref<1x?xf32>

    // dst = output [1, N] = [1, 4480]
    %dst  = memref.alloc(%c4480) : memref<1x?xf32>

    linalg.fill ins(%f3 : f32) outs(%src0 : memref<?x?xf32>)
    linalg.fill ins(%f2 : f32) outs(%src1 : memref<1x?xf32>)
    linalg.fill ins(%f0 : f32) outs(%dst  : memref<1x?xf32>)

    call @ggml_matmul_rowwise_v32(%src0, %src1, %dst)
      : (memref<?x?xf32>, memref<1x?xf32>, memref<1x?xf32>) -> ()

    memref.dealloc %dst  : memref<1x?xf32>
    memref.dealloc %src1 : memref<1x?xf32>
    memref.dealloc %src0 : memref<?x?xf32>
    return
  }
}