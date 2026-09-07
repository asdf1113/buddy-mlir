// This focused fixture preserves the real rank-0 DeepSeek TP=2
// forward_decode_layer0_seg1 wrapper ABI, parameter-pack offsets, and wrapper
// view plumbing.  The private subgraph uses a small deterministic computation
// so the ABI regression does not require checked-in model weights.
module {
  func.func private @subgraph0_decode1_seg1(
      %arg0: memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>,
      %arg1: memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>,
      %arg2: memref<1536xf32, strided<[1], offset: ?>>,
      %arg3: memref<1536x4480xf32, strided<[4480, 1], offset: ?>>,
      %arg4: memref<1536x4480xf32, strided<[4480, 1], offset: ?>>,
      %arg5: memref<4480x1536xf32, strided<[1536, 1], offset: ?>>)
      -> (memref<1x1x1536xf32>, memref<1x1x1536xf32>) {
    %output0 = memref.alloc() : memref<1x1x1536xf32>
    %output1 = memref.alloc() : memref<1x1x1536xf32>
    %c0 = arith.constant 0 : index
    %c1536 = arith.constant 1536 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1536 step %c1 {
      %lhs = memref.load %arg0[%c0, %c0, %i] : memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>
      %rhs = memref.load %arg1[%c0, %c0, %i] : memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>
      %bias = memref.load %arg2[%i] : memref<1536xf32, strided<[1], offset: ?>>
      %sum = arith.addf %lhs, %rhs : f32
      %first = arith.addf %sum, %bias : f32
      %difference = arith.subf %lhs, %rhs : f32
      %second = arith.addf %difference, %bias : f32
      memref.store %first, %output0[%c0, %c0, %i] : memref<1x1x1536xf32>
      memref.store %second, %output1[%c0, %c0, %i] : memref<1x1x1536xf32>
    }
    return %output0, %output1 : memref<1x1x1536xf32>, memref<1x1x1536xf32>
  }

  func.func @forward_decode_layer0_seg1(
      %arg0: memref<1121961536xf32>,
      %arg1: memref<1x1x1536xf32>,
      %arg2: memref<1x1x1536xf32>)
      -> (memref<1x1x1536xf32>, memref<1x1x1536xf32>) {
    %subview = memref.subview %arg0[236128768] [1536] [1] : memref<1121961536xf32> to memref<1536xf32, strided<[1], offset: 236128768>>
    %subview_0 = memref.subview %arg0[236130304] [6881280] [1] : memref<1121961536xf32> to memref<6881280xf32, strided<[1], offset: 236130304>>
    %expand_shape = memref.expand_shape %subview_0 [[0, 1]] output_shape [1536, 4480] : memref<6881280xf32, strided<[1], offset: 236130304>> into memref<1536x4480xf32, strided<[4480, 1], offset: 236130304>>
    %subview_1 = memref.subview %arg0[243011584] [6881280] [1] : memref<1121961536xf32> to memref<6881280xf32, strided<[1], offset: 243011584>>
    %expand_shape_2 = memref.expand_shape %subview_1 [[0, 1]] output_shape [1536, 4480] : memref<6881280xf32, strided<[1], offset: 243011584>> into memref<1536x4480xf32, strided<[4480, 1], offset: 243011584>>
    %subview_3 = memref.subview %arg0[249892864] [6881280] [1] : memref<1121961536xf32> to memref<6881280xf32, strided<[1], offset: 249892864>>
    %expand_shape_4 = memref.expand_shape %subview_3 [[0, 1]] output_shape [4480, 1536] : memref<6881280xf32, strided<[1], offset: 249892864>> into memref<4480x1536xf32, strided<[1536, 1], offset: 249892864>>
    %cast = memref.cast %arg1 : memref<1x1x1536xf32> to memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>
    %cast_5 = memref.cast %arg2 : memref<1x1x1536xf32> to memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>
    %cast_6 = memref.cast %subview : memref<1536xf32, strided<[1], offset: 236128768>> to memref<1536xf32, strided<[1], offset: ?>>
    %cast_7 = memref.cast %expand_shape : memref<1536x4480xf32, strided<[4480, 1], offset: 236130304>> to memref<1536x4480xf32, strided<[4480, 1], offset: ?>>
    %cast_8 = memref.cast %expand_shape_2 : memref<1536x4480xf32, strided<[4480, 1], offset: 243011584>> to memref<1536x4480xf32, strided<[4480, 1], offset: ?>>
    %cast_9 = memref.cast %expand_shape_4 : memref<4480x1536xf32, strided<[1536, 1], offset: 249892864>> to memref<4480x1536xf32, strided<[1536, 1], offset: ?>>
    %0:2 = call @subgraph0_decode1_seg1(%cast, %cast_5, %cast_6, %cast_7, %cast_8, %cast_9) : (memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>, memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>, memref<1536xf32, strided<[1], offset: ?>>, memref<1536x4480xf32, strided<[4480, 1], offset: ?>>, memref<1536x4480xf32, strided<[4480, 1], offset: ?>>, memref<4480x1536xf32, strided<[1536, 1], offset: ?>>) -> (memref<1x1x1536xf32>, memref<1x1x1536xf32>)
    return %0#0, %0#1 : memref<1x1x1536xf32>, memref<1x1x1536xf32>
  }
}
