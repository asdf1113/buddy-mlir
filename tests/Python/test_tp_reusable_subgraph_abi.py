# RUN: %PYTHON %s buddy-opt mlir-opt

import ctypes
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
COMPILE_PIPELINE = (
    SOURCE_ROOT / "tools" / "buddy-codegen" / "compile_pipeline.py"
)

compile_pipeline_spec = importlib.util.spec_from_file_location(
    "compile_pipeline", COMPILE_PIPELINE
)
assert (
    compile_pipeline_spec is not None
    and compile_pipeline_spec.loader is not None
)
compile_pipeline = importlib.util.module_from_spec(compile_pipeline_spec)
compile_pipeline_spec.loader.exec_module(compile_pipeline)

SUBGRAPH = r"""
module {
  func.func @abi_subgraph(%arg0: tensor<4xf32>, %arg1: tensor<4xi1>)
      -> (tensor<4xf32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) {
    return %arg0, %arg1, %arg0, %arg0
        : tensor<4xf32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>
  }
}
"""

WRAPPER = r"""
module {
  func.func private @abi_subgraph(
      memref<4xf32, strided<[1], offset: ?>>,
      memref<4xi1, strided<[1], offset: ?>>)
      -> (memref<4xf32>, memref<4xi1>, memref<4xf32>, memref<4xf32>)

  func.func @abi_wrapper(%arg0: memref<4xf32>, %arg1: memref<4xi1>)
      -> (memref<4xf32>, memref<4xi1>, memref<4xf32>, memref<4xf32>) {
    %cast0 = memref.cast %arg0
        : memref<4xf32> to memref<4xf32, strided<[1], offset: ?>>
    %cast1 = memref.cast %arg1
        : memref<4xi1> to memref<4xi1, strided<[1], offset: ?>>
    %0:4 = call @abi_subgraph(%cast0, %cast1)
        : (memref<4xf32, strided<[1], offset: ?>>,
           memref<4xi1, strided<[1], offset: ?>>)
        -> (memref<4xf32>, memref<4xi1>, memref<4xf32>, memref<4xf32>)
    return %0#0, %0#1, %0#2, %0#3
        : memref<4xf32>, memref<4xi1>, memref<4xf32>, memref<4xf32>
  }
}
"""


class MemRef1DF32(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.POINTER(ctypes.c_float)),
        ("aligned", ctypes.POINTER(ctypes.c_float)),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 1),
        ("strides", ctypes.c_int64 * 1),
    ]


class MemRef1DI1(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.POINTER(ctypes.c_bool)),
        ("aligned", ctypes.POINTER(ctypes.c_bool)),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 1),
        ("strides", ctypes.c_int64 * 1),
    ]


def descriptor(descriptor_type, payload):
    pointer = ctypes.cast(payload, descriptor_type._fields_[0][1])
    return descriptor_type(
        pointer,
        pointer,
        0,
        (ctypes.c_int64 * 1)(4),
        (ctypes.c_int64 * 1)(1),
    )


buddy_opt = sys.argv[1]
mlir_opt = shutil.which(sys.argv[2])
cxx = shutil.which("c++")
assert Path(buddy_opt).is_file()
assert mlir_opt is not None
assert cxx is not None
llvm_root = Path(mlir_opt).resolve().parent.parent
openmp_runtime = next(
    candidate
    for candidate in (
        llvm_root / "lib" / "libomp.so",
        llvm_root
        / "runtimes"
        / "runtimes-bins"
        / "openmp"
        / "runtime"
        / "src"
        / "libomp.so",
    )
    if candidate.is_file()
)

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    phase_dir = root / "layer_partitioned" / "rank0" / "forward_prefill"
    phase_dir.mkdir(parents=True)
    (phase_dir / "abi_subgraph.mlir").write_text(SUBGRAPH)
    (phase_dir / "abi_wrapper.mlir").write_text(WRAPPER)

    runtime_plan = root / "rank0_forward_prefill.json"
    runtime_plan.write_text(
        json.dumps(
            {
                "graph": "forward_prefill",
                "rank": 0,
                "world_size": 2,
                "operations": [
                    {
                        "kind": "dispatch",
                        "wrapper": "abi_wrapper",
                        "function": "abi_subgraph",
                    }
                ],
            }
        )
    )

    objects = compile_pipeline.compile_partitioned(
        config={"variant": "f32", "compilation": {"num_threads": 1}},
        mlir_dir=str(root / "layer_partitioned"),
        output_dir=str(root / "objects"),
        buddy_opt=buddy_opt,
        llvm_dir=str(Path(mlir_opt).parent),
        llc_attrs="-mcpu=native",
        runtime_plan_paths=[str(runtime_plan)],
    )
    library = root / "libabi.so"
    compile_pipeline.link_shared_lib(
        obj_files=objects,
        output_so=str(library),
        cxx=cxx,
        llvm_lib_dir=os.environ["LLVM_LIBS_DIR"],
        openmp_runtime_lib=str(openmp_runtime),
    )

    module = ctypes.CDLL(str(library))
    wrapper = module._mlir_ciface_abi_wrapper
    wrapper.argtypes = [
        ctypes.POINTER(MemRef1DF32),
        ctypes.POINTER(MemRef1DI1),
        ctypes.POINTER(MemRef1DF32),
        ctypes.POINTER(MemRef1DI1),
        ctypes.POINTER(MemRef1DF32),
        ctypes.POINTER(MemRef1DF32),
    ]
    wrapper.restype = None

    input_f32 = (ctypes.c_float * 4)(1.0, -2.0, 3.5, 4.25)
    input_i1 = (ctypes.c_bool * 4)(True, False, True, False)
    outputs_f32 = [(ctypes.c_float * 4)() for _ in range(3)]
    output_i1 = (ctypes.c_bool * 4)()
    arguments = [
        descriptor(MemRef1DF32, input_f32),
        descriptor(MemRef1DI1, input_i1),
        descriptor(MemRef1DF32, outputs_f32[0]),
        descriptor(MemRef1DI1, output_i1),
        descriptor(MemRef1DF32, outputs_f32[1]),
        descriptor(MemRef1DF32, outputs_f32[2]),
    ]
    wrapper(*(ctypes.byref(argument) for argument in arguments))

    expected_f32 = list(input_f32)
    assert all(list(output) == expected_f32 for output in outputs_f32)
    assert list(output_i1) == list(input_i1)
