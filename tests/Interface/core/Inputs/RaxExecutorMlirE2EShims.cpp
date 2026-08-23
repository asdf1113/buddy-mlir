#include "mlir/ExecutionEngine/CRunnerUtils.h"

using MemRef1D = StridedMemRefType<float, 1>;

extern "C" void _mlir_ciface_forward0(MemRef1D *params, MemRef1D *input,
                                      MemRef1D *scratch);
extern "C" void _mlir_ciface_forward1(MemRef1D *params, MemRef1D *scratch,
                                      MemRef1D *output);

extern "C" void rax_forward0(void **args) {
  _mlir_ciface_forward0(static_cast<MemRef1D *>(args[0]),
                        static_cast<MemRef1D *>(args[1]),
                        static_cast<MemRef1D *>(args[2]));
}

extern "C" void rax_forward1(void **args) {
  _mlir_ciface_forward1(static_cast<MemRef1D *>(args[0]),
                        static_cast<MemRef1D *>(args[1]),
                        static_cast<MemRef1D *>(args[2]));
}
