#include "mlir/ExecutionEngine/CRunnerUtils.h"

using MemRef1D = StridedMemRefType<float, 1>;
using MemRef2D = StridedMemRefType<float, 2>;

extern "C" void _mlir_ciface_stage2e_tp_layer0_seg0(MemRef2D *result,
                                                    MemRef1D *params,
                                                    MemRef2D *input);
extern "C" void _mlir_ciface_stage2e_tp_layer0_seg1(MemRef2D *result,
                                                    MemRef1D *params,
                                                    MemRef2D *boundary);

extern "C" void rax_stage2e_tp_layer0_seg0(void **args) {
  _mlir_ciface_stage2e_tp_layer0_seg0(static_cast<MemRef2D *>(args[2]),
                                      static_cast<MemRef1D *>(args[0]),
                                      static_cast<MemRef2D *>(args[1]));
}

extern "C" void rax_stage2e_tp_layer0_seg1(void **args) {
  _mlir_ciface_stage2e_tp_layer0_seg1(static_cast<MemRef2D *>(args[2]),
                                      static_cast<MemRef1D *>(args[0]),
                                      static_cast<MemRef2D *>(args[1]));
}
