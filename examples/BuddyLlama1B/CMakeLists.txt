# ==============================================================================
# Options / basic config
# ==============================================================================
option(CROSS_COMPILE_RVV "Build for RISC-V RVV target" OFF)

# ==============================================================================
# Configurable variables (default from environment, overridable by -D)
# ==============================================================================
set(BUDDY_MLIR_BUILD_DIR "$ENV{BUDDY_MLIR_BUILD_DIR}" CACHE PATH
  "Buddy MLIR host build directory (contains bin/buddy-opt)")
set(LLVM_MLIR_BUILD_DIR "$ENV{LLVM_MLIR_BUILD_DIR}" CACHE PATH
  "LLVM/MLIR host build directory (contains bin/mlir-opt, mlir-translate, llvm-as, llc)")
set(PYTHONPATH "$ENV{PYTHONPATH}" CACHE STRING
  "Python module search path")
set(BUDDY_MLIR_BUILD_CROSS_DIR "$ENV{BUDDY_MLIR_BUILD_CROSS_DIR}" CACHE PATH
  "Buddy MLIR cross build directory (contains target libraries in lib/)")
set(RISCV_GNU_TOOLCHAIN "$ENV{RISCV_GNU_TOOLCHAIN}" CACHE PATH
  "RISC-V GNU toolchain root")
set(RISCV_SYSROOT "$ENV{RISCV_SYSROOT}" CACHE PATH
  "RISC-V sysroot root path (defaults to ${RISCV_GNU_TOOLCHAIN}/sysroot when CROSS_COMPILE_RVV=ON)")
set(RISCV_OMP_SHARED "$ENV{RISCV_OMP_SHARED}" CACHE FILEPATH
  "Full path to target OpenMP shared library for RISC-V")
set(DSTP_EXAMPLE_PATH "$ENV{DSTP_EXAMPLE_PATH}" CACHE STRING
  "Runtime path prefix on target device")
set(DSTP_EXAMPLE_BUILD_PATH "$ENV{DSTP_EXAMPLE_BUILD_PATH}" CACHE STRING
  "Runtime build path prefix on target device")

# ==============================================================================
# Fixed options / pipelines
# ==============================================================================
set(TOSA_PIPELINE
  "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# ==============================================================================
# Resolve build tools (host tools)
# ==============================================================================
if(NOT BUDDY_MLIR_BUILD_DIR STREQUAL "")
  set(BUDDY_TOOLS_BINARY_DIR "${BUDDY_MLIR_BUILD_DIR}/bin")
else()
  set(BUDDY_TOOLS_BINARY_DIR "${BUDDY_BINARY_DIR}")
endif()

if(NOT EXISTS "${BUDDY_TOOLS_BINARY_DIR}/buddy-opt")
  message(FATAL_ERROR "buddy-opt not found: ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt")
endif()

if(NOT LLVM_MLIR_BUILD_DIR STREQUAL "")
  set(LLVM_TOOLS_BIN_DIR "${LLVM_MLIR_BUILD_DIR}/bin")
else()
  set(LLVM_TOOLS_BIN_DIR "${LLVM_TOOLS_BINARY_DIR}")
endif()

if(NOT EXISTS "${LLVM_TOOLS_BIN_DIR}/mlir-opt")
  message(FATAL_ERROR "mlir-opt not found: ${LLVM_TOOLS_BIN_DIR}/mlir-opt")
endif()
if(NOT EXISTS "${LLVM_TOOLS_BIN_DIR}/mlir-translate")
  message(FATAL_ERROR "mlir-translate not found: ${LLVM_TOOLS_BIN_DIR}/mlir-translate")
endif()
if(NOT EXISTS "${LLVM_TOOLS_BIN_DIR}/llvm-as")
  message(FATAL_ERROR "llvm-as not found: ${LLVM_TOOLS_BIN_DIR}/llvm-as")
endif()
if(NOT EXISTS "${LLVM_TOOLS_BIN_DIR}/llc")
  message(FATAL_ERROR "llc not found: ${LLVM_TOOLS_BIN_DIR}/llc")
endif()

set(BUDDY_OPT_DEP)
if(BUDDY_MLIR_BUILD_DIR STREQUAL "${CMAKE_BINARY_DIR}")
  set(BUDDY_OPT_DEP buddy-opt)
endif()

# ==============================================================================
# Platform-specific config
# ==============================================================================
if(CROSS_COMPILE_RVV)
  set(OPENMP_NUM_THREADS "num-threads=8")
  set(LLC_TARGET_FLAGS
    -mtriple=riscv64-unknown-linux-gnu
    -target-abi=lp64d
    -mattr=+m,+d,+v
  )

  set(CMAKE_SYSTEM_NAME Linux)
  set(CMAKE_SYSTEM_PROCESSOR riscv64)

  if(RISCV_GNU_TOOLCHAIN STREQUAL "")
    message(FATAL_ERROR
      "CROSS_COMPILE_RVV=ON requires RISCV_GNU_TOOLCHAIN. "
      "Please export it or pass -DRISCV_GNU_TOOLCHAIN=...")
  endif()

  if(RISCV_SYSROOT STREQUAL "")
    set(RISCV_SYSROOT "${RISCV_GNU_TOOLCHAIN}/sysroot")
  endif()
  set(CMAKE_SYSROOT "${RISCV_SYSROOT}")

  if(NOT EXISTS "${RISCV_SYSROOT}")
    message(FATAL_ERROR "RISCV sysroot not found: ${RISCV_SYSROOT}")
  endif()

  if(BUDDY_MLIR_BUILD_CROSS_DIR STREQUAL "")
    message(FATAL_ERROR
      "CROSS_COMPILE_RVV=ON requires BUDDY_MLIR_BUILD_CROSS_DIR. "
      "Please export it or pass -DBUDDY_MLIR_BUILD_CROSS_DIR=...")
  endif()

  if(RISCV_OMP_SHARED STREQUAL "")
    message(FATAL_ERROR
      "CROSS_COMPILE_RVV=ON requires RISCV_OMP_SHARED. "
      "Please export it or pass -DRISCV_OMP_SHARED=...")
  endif()

  set(CMAKE_FIND_ROOT_PATH
    "${RISCV_SYSROOT}"
    "${BUDDY_MLIR_BUILD_CROSS_DIR}"
  )

  set(RISCV_COMMON_FLAGS
    "--sysroot=${RISCV_SYSROOT} -march=rv64gcv -mabi=lp64d -fPIC")

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${RISCV_COMMON_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${RISCV_COMMON_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${RISCV_COMMON_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${RISCV_COMMON_FLAGS}")

  set(OPENMP_LIB "${RISCV_OMP_SHARED}")
  set(MLIR_RUNNER_UTILS_LIB
    "${BUDDY_MLIR_BUILD_CROSS_DIR}/lib/libStaticMLIRCRunnerUtils.a")

  if(DSTP_EXAMPLE_PATH STREQUAL "")
    set(DSTP_EXAMPLE_PATH ".")
  endif()
  if(DSTP_EXAMPLE_BUILD_PATH STREQUAL "")
    set(DSTP_EXAMPLE_BUILD_PATH ".")
  endif()

else()
  set(OPENMP_NUM_THREADS "num-threads=48")
  set(LLC_TARGET_FLAGS "-mcpu=native")

  set(OPENMP_LIB omp)
  set(MLIR_RUNNER_UTILS_LIB mlir_c_runner_utils)

  set(DSTP_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
  set(DSTP_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ==============================================================================
# Common checks
# ==============================================================================
if(CROSS_COMPILE_RVV)
  if(NOT EXISTS "${OPENMP_LIB}")
    message(FATAL_ERROR "RISC-V OpenMP shared library does not exist: ${OPENMP_LIB}")
  endif()
  if(NOT EXISTS "${MLIR_RUNNER_UTILS_LIB}")
    message(FATAL_ERROR "RISC-V static MLIR runner utils library does not exist: ${MLIR_RUNNER_UTILS_LIB}")
  endif()
endif()

message(STATUS "CROSS_COMPILE_RVV          = ${CROSS_COMPILE_RVV}")
message(STATUS "BUDDY_MLIR_BUILD_DIR       = ${BUDDY_MLIR_BUILD_DIR}")
message(STATUS "LLVM_MLIR_BUILD_DIR        = ${LLVM_MLIR_BUILD_DIR}")
message(STATUS "BUDDY_TOOLS_BINARY_DIR     = ${BUDDY_TOOLS_BINARY_DIR}")
message(STATUS "LLVM_TOOLS_BIN_DIR         = ${LLVM_TOOLS_BIN_DIR}")
message(STATUS "PYTHONPATH                 = ${PYTHONPATH}")
message(STATUS "BUDDY_MLIR_BUILD_CROSS_DIR = ${BUDDY_MLIR_BUILD_CROSS_DIR}")
message(STATUS "RISCV_GNU_TOOLCHAIN        = ${RISCV_GNU_TOOLCHAIN}")
message(STATUS "RISCV_SYSROOT              = ${RISCV_SYSROOT}")
message(STATUS "RISCV_OMP_SHARED           = ${RISCV_OMP_SHARED}")
message(STATUS "OPENMP_LIB                 = ${OPENMP_LIB}")
message(STATUS "MLIR_RUNNER_UTILS_LIB      = ${MLIR_RUNNER_UTILS_LIB}")
message(STATUS "DSTP_EXAMPLE_PATH          = ${DSTP_EXAMPLE_PATH}")
message(STATUS "DSTP_EXAMPLE_BUILD_PATH    = ${DSTP_EXAMPLE_BUILD_PATH}")

# add_custom_command(
#   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
#          ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
#          ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
#   COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-llama2.py
#           --output-dir ${CMAKE_CURRENT_BINARY_DIR}
#   COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
# )

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  COMMAND ${LLVM_TOOLS_BIN_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            # -matmul-parallel-vectorization-optimize
            -matmul-vectorization #unroll
            # -matmul-rvv #my
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BIN_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BIN_DIR}/llvm-as |
        ${LLVM_TOOLS_BIN_DIR}/llc ${LLC_TARGET_FLAGS} -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  DEPENDS ${BUDDY_OPT_DEP} ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph.o
  COMMAND ${LLVM_TOOLS_BIN_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt
            -convert-elementwise-to-linalg
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            # -matmul-parallel-vectorization-optimize #old
            -matmul-vectorization #unroll
            # -matmul-rvv #my
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
            -convert-vector-to-scf
            -expand-strided-metadata
            -lower-affine
            -cse
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BIN_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BIN_DIR}/llvm-as |
        ${LLVM_TOOLS_BIN_DIR}/llc ${LLC_TARGET_FLAGS} -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph.o
  DEPENDS ${BUDDY_OPT_DEP} ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
  COMMENT "Building subgraph.o"
  VERBATIM
)

add_library(LLAMA STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph.o
)

set_target_properties(
  LLAMA
  PROPERTIES
  LINKER_LANGUAGE C
)

add_executable(buddy-llama-run llama-main.cpp)

target_compile_definitions(buddy-llama-run PRIVATE
  LLAMA_EXAMPLE_PATH="${DSTP_EXAMPLE_PATH}"
  LLAMA_EXAMPLE_BUILD_PATH="${DSTP_EXAMPLE_BUILD_PATH}"
)

if(NOT CROSS_COMPILE_RVV)
  target_link_directories(buddy-llama-run PRIVATE ${LLVM_LIBRARY_DIR})
endif()

set(BUDDY_LLAMA_LIBS
  LLAMA
  "${MLIR_RUNNER_UTILS_LIB}"
  "${OPENMP_LIB}"
)

if(BUDDY_MLIR_USE_MIMALLOC AND NOT CROSS_COMPILE_RVV)
  list(APPEND BUDDY_LLAMA_LIBS mimalloc)
endif()

target_link_libraries(buddy-llama-run PRIVATE ${BUDDY_LLAMA_LIBS})




# # add_custom_command(
# #   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
# #          ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
# #          ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
# #   COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-llama2.py
# #           --output-dir ${CMAKE_CURRENT_BINARY_DIR}
# #   COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
# # )



# add_custom_command(
#   OUTPUT forward.o
#   COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir
#             -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
#           ${BUDDY_BINARY_DIR}/buddy-opt
#             -arith-expand
#             -eliminate-empty-tensors
#             -empty-tensor-to-alloc-tensor
#             -one-shot-bufferize="bufferize-function-boundaries"
#             -ownership-based-buffer-deallocation
#             -buffer-deallocation-simplification
#             -bufferization-lower-deallocations
#             -matmul-parallel-vectorization-optimize
#             -batchmatmul-optimize
#             -convert-linalg-to-affine-loops
#             -affine-loop-fusion
#             -affine-parallelize
#             -lower-affine
#             -convert-scf-to-openmp=num-threads=48b
#             -convert-vector-to-scf
#             -expand-strided-metadata
#             -convert-vector-to-llvm
#             -memref-expand
#             -arith-expand
#             -convert-arith-to-llvm
#             -finalize-memref-to-llvm
#             -convert-scf-to-cf
#             -convert-cf-to-llvm
#             -llvm-request-c-wrappers
#             -convert-openmp-to-llvm
#             -convert-arith-to-llvm
#             -convert-math-to-llvm
#             -convert-math-to-libm
#             -convert-func-to-llvm
#             -reconcile-unrealized-casts |
#         ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
#         ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
#         ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
#           -o ${CMAKE_CURRENT_BINARY_DIR}/forward.o
#   DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir
#   COMMENT "Building forward.o "
#   VERBATIM)

# add_custom_command(
#     OUTPUT subgraph.o
#     COMMAND ti  ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
#               -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
#             ${BUDDY_BINARY_DIR}/buddy-opt
#             -convert-elementwise-to-linalg
#             -arith-expand
#             -eliminate-empty-tensors
#             -empty-tensor-to-alloc-tensor
#             -one-shot-bufferize="bufferize-function-boundaries"
#             -ownership-based-buffer-deallocation
#             -buffer-deallocation-simplification
#             -bufferization-lower-deallocations
#             -matmul-parallel-vectorization-optimize
#             -batchmatmul-optimize
#             -convert-linalg-to-affine-loops
#             -affine-loop-fusion
#             -affine-parallelize
#             -lower-affine
#             -convert-scf-to-openmp=num-threads=48
#             -convert-vector-to-scf
#             -expand-strided-metadata
#             -lower-affine
#             -cse
#             -convert-vector-to-llvm
#             -memref-expand
#             -arith-expand
#             -convert-arith-to-llvm
#             -finalize-memref-to-llvm
#             -convert-scf-to-cf
#             -convert-cf-to-llvm
#             -llvm-request-c-wrappers
#             -convert-openmp-to-llvm
#             -convert-arith-to-llvm
#             -convert-math-to-llvm
#             -convert-math-to-libm
#             -convert-func-to-llvm
#             -reconcile-unrealized-casts |
#           ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
#           ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
#           ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
#             -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph.o
#     DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
#     COMMENT "Building subgraph.o "
#     VERBATIM)

# add_library(LLAMA STATIC forward.o subgraph.o)

# SET_SOURCE_FILES_PROPERTIES(
#   template.o
#   PROPERTIES
#   EXTERNAL_OBJECT true
#   GENERATED true)

# SET_TARGET_PROPERTIES(
#   LLAMA
#   PROPERTIES
#   LINKER_LANGUAGE C)

# add_executable(buddy-llama-run llama-main.cpp)

# set(LLAMA_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
# set(LLAMA_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# target_compile_definitions(buddy-llama-run PRIVATE
#   LLAMA_EXAMPLE_PATH="${LLAMA_EXAMPLE_PATH}"
#   LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
# )

# target_link_directories(buddy-llama-run PRIVATE ${LLVM_LIBRARY_DIR})

# set(BUDDY_LLAMA_LIBS
#   LLAMA
#   mlir_c_runner_utils
#   omp
# )
# if(BUDDY_MLIR_USE_MIMALLOC)
#   list(APPEND BUDDY_LLAMA_LIBS mimalloc)
# endif()

# target_link_libraries(buddy-llama-run ${BUDDY_LLAMA_LIBS})
