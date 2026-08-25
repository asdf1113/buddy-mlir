#include "mlir/ExecutionEngine/CRunnerUtils.h"

using MemRef1D = StridedMemRefType<float, 1>;

static float *data(MemRef1D *descriptor) {
  return descriptor->data + descriptor->offset;
}

extern "C" void init_shared(void **args) {
  float *scale = data(static_cast<MemRef1D *>(args[0]));
  float *bias = data(static_cast<MemRef1D *>(args[1]));
  scale[0] = 2.0f;
  scale[1] = 3.0f;
  bias[0] = 1.0f;
  bias[1] = 1.0f;
}

extern "C" void compute_partial(void **args) {
  float *input = data(static_cast<MemRef1D *>(args[0]));
  float *scale = data(static_cast<MemRef1D *>(args[1]));
  float *bias = data(static_cast<MemRef1D *>(args[2]));
  float *partial = data(static_cast<MemRef1D *>(args[3]));
  for (int i = 0; i < 2; ++i)
    partial[i] = input[i] * scale[i] + bias[i];
}

extern "C" void finish(void **args) {
  float *partial = data(static_cast<MemRef1D *>(args[0]));
  float *bias = data(static_cast<MemRef1D *>(args[1]));
  float *output = data(static_cast<MemRef1D *>(args[2]));
  for (int i = 0; i < 2; ++i)
    output[i] = partial[i] + bias[i];
}
