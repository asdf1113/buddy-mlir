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
set(RISCV_OMP_STATIC "$ENV{RISCV_OMP_STATIC}" CACHE FILEPATH
  "Full path to target libomp.a for RISC-V")
# set(MPICH_ROOT "$ENV{MPICH_ROOT}" CACHE PATH
#   "MPICH installation root (contains include/ and lib/)")

# ==============================================================================
# Fixed options / pipelines
# ==============================================================================
set(BUFFERIZE_FULL_OPTS
  "unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries")
set(BUFFERIZE_SIMPLE_OPTS
  "bufferize-function-boundaries")
set(TOSA_PIPELINE
  "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

# Avoid try_compile link checks during configure in cross mode.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# ==============================================================================
# Resolve build tools (host tools)
# ==============================================================================
# buddy-opt:
# default: ${BUDDY_BINARY_DIR}/buddy-opt
# override: ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
if(NOT BUDDY_MLIR_BUILD_DIR STREQUAL "")
  set(BUDDY_TOOLS_BINARY_DIR "${BUDDY_MLIR_BUILD_DIR}/bin")
else()
  set(BUDDY_TOOLS_BINARY_DIR "${BUDDY_BINARY_DIR}")
endif()

if(NOT EXISTS "${BUDDY_TOOLS_BINARY_DIR}/buddy-opt")
  message(FATAL_ERROR "buddy-opt not found: ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt")
endif()

# llvm tools:
# default: ${LLVM_TOOLS_BINARY_DIR}/<tool>
# override: ${LLVM_MLIR_BUILD_DIR}/bin/<tool>
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
  set(OPENMP_NUM_THREADS "num-threads=32")
  set(LLC_TARGET_FLAGS
    -mtriple=riscv64-unknown-linux-gnu
    -target-abi=lp64d
    -mattr=+m,+a,+f,+d,+c,+v
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

  if(RISCV_OMP_STATIC STREQUAL "")
    message(FATAL_ERROR
      "CROSS_COMPILE_RVV=ON requires RISCV_OMP_STATIC. "
      "Please export it or pass -DRISCV_OMP_STATIC=...")
  endif()

  # if(MPICH_ROOT STREQUAL "")
  #   message(FATAL_ERROR
  #     "CROSS_COMPILE_RVV=ON requires MPICH_ROOT. "
  #     "Please export it or pass -DMPICH_ROOT=...")
  # endif()

  # Search target headers/libraries from target roots.
  set(CMAKE_FIND_ROOT_PATH
    "${RISCV_SYSROOT}"
    # "${MPICH_ROOT}"
    "${BUDDY_MLIR_BUILD_CROSS_DIR}"
  )

  set(RISCV_COMMON_FLAGS
    "--sysroot=${RISCV_SYSROOT} -march=rv64gcv -mabi=lp64d -fPIC")

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${RISCV_COMMON_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${RISCV_COMMON_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${RISCV_COMMON_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${RISCV_COMMON_FLAGS}")

  # set(MPICH_INCLUDE_DIR "${MPICH_ROOT}/include")
  # set(MPI_C_LIB "${MPICH_ROOT}/lib/libmpi.so")
  # set(MPI_CXX_LIB "${MPICH_ROOT}/lib/libmpicxx.so")

  set(OPENMP_LIB "${RISCV_OMP_STATIC}")
  set(MLIR_RUNNER_UTILS_LIB
    "${BUDDY_MLIR_BUILD_CROSS_DIR}/lib/libStaticMLIRCRunnerUtils.a")

  # Compile/link options for final executable
  set(TARGET_COMPILE_OPTS
    -march=rv64gcv
    -mabi=lp64d
    -O3
    -Wall
  )
  set(TARGET_LINK_OPTS
    -march=rv64gcv
    -mabi=lp64d
  )

  message(STATUS "CMAKE_C_COMPILER       = ${CMAKE_C_COMPILER}")
  message(STATUS "CMAKE_CXX_COMPILER     = ${CMAKE_CXX_COMPILER}")

else()
  set(OPENMP_NUM_THREADS "num-threads=48")
  set(LLC_TARGET_FLAGS "-mcpu=native")

  # Native compile/link opts for final executable
  set(TARGET_COMPILE_OPTS -O3 -Wall)
  set(TARGET_LINK_OPTS "")

  # # MPICH: support manual root, otherwise search default locations.
  # if(NOT MPICH_ROOT STREQUAL "")
  #   set(MPICH_INCLUDE_DIR "${MPICH_ROOT}/include")
  #   find_library(MPI_C_LIB mpi
  #     HINTS "${MPICH_ROOT}"
  #     PATH_SUFFIXES lib lib64
  #     NO_DEFAULT_PATH)
  #   find_library(MPI_CXX_LIB mpicxx
  #     HINTS "${MPICH_ROOT}"
  #     PATH_SUFFIXES lib lib64
  #     NO_DEFAULT_PATH)
  # else()
  #   find_path(MPICH_INCLUDE_DIR mpi.h
  #     HINTS /usr /usr/local /opt/homebrew /opt/local
  #     PATH_SUFFIXES include)
  #   find_library(MPI_C_LIB mpi
  #     HINTS /usr /usr/local /opt/homebrew /opt/local
  #     PATH_SUFFIXES lib lib64)
  #   find_library(MPI_CXX_LIB mpicxx
  #     HINTS /usr /usr/local /opt/homebrew /opt/local
  #     PATH_SUFFIXES lib lib64)
  # endif()

  # Native mode: keep original successful link style
  set(OPENMP_LIB omp)
  set(MLIR_RUNNER_UTILS_LIB mlir_c_runner_utils)
endif()

# ==============================================================================
# Common checks
# ==============================================================================
# if(NOT EXISTS "${MPICH_INCLUDE_DIR}/mpi.h")
#   message(FATAL_ERROR "MPICH header not found: ${MPICH_INCLUDE_DIR}/mpi.h")
# endif()

# if(NOT EXISTS "${MPI_C_LIB}")
#   message(FATAL_ERROR "MPI C shared library not found: ${MPI_C_LIB}")
# endif()

# if(NOT EXISTS "${MPI_CXX_LIB}")
#   message(FATAL_ERROR "MPI CXX shared library not found: ${MPI_CXX_LIB}")
# endif()

if(CROSS_COMPILE_RVV)
  if(NOT OPENMP_LIB)
    message(FATAL_ERROR "RISC-V OpenMP static library not found")
  endif()
  if(NOT EXISTS "${OPENMP_LIB}")
    message(FATAL_ERROR "RISC-V OpenMP static library does not exist: ${OPENMP_LIB}")
  endif()

  if(NOT MLIR_RUNNER_UTILS_LIB)
    message(FATAL_ERROR "RISC-V static MLIR runner utils library not found")
  endif()
  if(NOT EXISTS "${MLIR_RUNNER_UTILS_LIB}")
    message(FATAL_ERROR "RISC-V static MLIR runner utils library does not exist: ${MLIR_RUNNER_UTILS_LIB}")
  endif()
endif()

message(STATUS "CROSS_COMPILE_RVV           = ${CROSS_COMPILE_RVV}")
message(STATUS "BUDDY_MLIR_BUILD_DIR        = ${BUDDY_MLIR_BUILD_DIR}")
message(STATUS "LLVM_MLIR_BUILD_DIR         = ${LLVM_MLIR_BUILD_DIR}")
message(STATUS "BUDDY_TOOLS_BINARY_DIR      = ${BUDDY_TOOLS_BINARY_DIR}")
message(STATUS "LLVM_TOOLS_BIN_DIR          = ${LLVM_TOOLS_BIN_DIR}")
message(STATUS "PYTHONPATH                  = ${PYTHONPATH}")
message(STATUS "BUDDY_MLIR_BUILD_CROSS_DIR  = ${BUDDY_MLIR_BUILD_CROSS_DIR}")
message(STATUS "RISCV_GNU_TOOLCHAIN         = ${RISCV_GNU_TOOLCHAIN}")
message(STATUS "RISCV_SYSROOT               = ${RISCV_SYSROOT}")
message(STATUS "RISCV_OMP_STATIC            = ${RISCV_OMP_STATIC}")
# message(STATUS "MPICH_ROOT                  = ${MPICH_ROOT}")
# message(STATUS "MPICH_INCLUDE_DIR           = ${MPICH_INCLUDE_DIR}")
message(STATUS "MPI_C_LIB                   = ${MPI_C_LIB}")
message(STATUS "MPI_CXX_LIB                 = ${MPI_CXX_LIB}")
message(STATUS "OPENMP_LIB                  = ${OPENMP_LIB}")
message(STATUS "MLIR_RUNNER_UTILS_LIB       = ${MLIR_RUNNER_UTILS_LIB}")

# ==============================================================================
# Generate MLIR files
# ==============================================================================
add_custom_command(
  OUTPUT
  ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode.mlir
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_arg0.data
  COMMAND ${CMAKE_COMMAND} -E env
  PYTHONPATH=${PYTHONPATH}
  ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1.py
  --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating forward_prefill.mlir, subgraph0_prefill.mlir, forward_decode.mlir, subgraph0_decode.mlir and arg0.data..."
  VERBATIM
)

# ==============================================================================
# Pass pipelines
# ==============================================================================
set(FORWARD_PASSES
  -eliminate-empty-tensors
  -empty-tensor-to-alloc-tensor
  -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
  -expand-strided-metadata
  -ownership-based-buffer-deallocation
  -buffer-deallocation-simplification
  -bufferization-lower-deallocations
  -matmul-vectorization-blis
  -batchmatmul-optimize
  -convert-linalg-to-affine-loops
  -affine-parallelize
  -convert-vector-to-scf
  -lower-affine
  -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
  -cse
  -memref-expand
  -arith-expand
  -convert-vector-to-llvm
  -convert-arith-to-llvm
  -finalize-memref-to-llvm
  -convert-scf-to-cf
  -convert-cf-to-llvm
  -convert-ub-to-llvm
  -llvm-request-c-wrappers
  -convert-openmp-to-llvm
  -convert-arith-to-llvm
  -convert-math-to-llvm
  -convert-math-to-libm
  -convert-func-to-llvm
  -reconcile-unrealized-casts
)

set(SUBGRAPH_PREFILL_PASSES
  -eliminate-empty-tensors
  -empty-tensor-to-alloc-tensor
  -convert-elementwise-to-linalg
  -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
  -expand-strided-metadata
  -ownership-based-buffer-deallocation
  -buffer-deallocation-simplification
  -bufferization-lower-deallocations
  -matmul-vectorization-blis
  -batchmatmul-optimize
  -batchmatmul-transpose-b-vectorization
  -convert-linalg-to-affine-loops
  -affine-parallelize
  -convert-vector-to-scf
  -lower-affine
  -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
  -cse
  -memref-expand
  -arith-expand
  -convert-vector-to-llvm
  -convert-arith-to-llvm
  -finalize-memref-to-llvm
  -convert-scf-to-cf
  -convert-cf-to-llvm
  -convert-ub-to-llvm
  -llvm-request-c-wrappers
  -convert-openmp-to-llvm
  -convert-arith-to-llvm
  -convert-math-to-llvm
  -convert-math-to-libm
  -convert-func-to-llvm
  -reconcile-unrealized-casts
)

set(SUBGRAPH_DECODE_PASSES
  -eliminate-empty-tensors
  -empty-tensor-to-alloc-tensor
  -convert-elementwise-to-linalg
  -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
  -expand-strided-metadata
  -ownership-based-buffer-deallocation
  -buffer-deallocation-simplification
  -bufferization-lower-deallocations
  -assume-tight-memref-layout
  -staticize-memref-layout
  -matmul-vectorization-decode=vector-size=128
  -batch-matmul-vectorization-decode=vector-size=128
  -batchmatmul-transpose-b-vectorization=vector-size=16
  -convert-linalg-to-affine-loops
  -convert-vector-to-scf
  -lower-affine
  -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
  -cse
  -memref-expand
  -arith-expand
  -convert-vector-to-llvm
  -convert-arith-to-llvm
  -finalize-memref-to-llvm
  -convert-scf-to-cf
  -convert-cf-to-llvm
  -convert-ub-to-llvm
  -llvm-request-c-wrappers
  -convert-openmp-to-llvm
  -convert-arith-to-llvm
  -convert-math-to-llvm
  -convert-math-to-libm
  -convert-func-to-llvm
  -reconcile-unrealized-casts
)

# ==============================================================================
# Build object files
# ==============================================================================
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.o
  COMMAND ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt
  ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.mlir
  -simplify-tosa-reshape |
  ${LLVM_TOOLS_BIN_DIR}/mlir-opt
  -pass-pipeline ${TOSA_PIPELINE} |
  ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt
  ${FORWARD_PASSES} |
  ${LLVM_TOOLS_BIN_DIR}/mlir-translate -mlir-to-llvmir |
  ${LLVM_TOOLS_BIN_DIR}/llvm-as |
  ${LLVM_TOOLS_BIN_DIR}/llc ${LLC_TARGET_FLAGS} -filetype=obj -relocation-model=pic -O3
  -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.o
  DEPENDS ${BUDDY_OPT_DEP} ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.mlir
  COMMENT "Building forward_prefill.o"
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill.o
  COMMAND ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill.mlir
  -simplify-tosa-reshape |
  ${LLVM_TOOLS_BIN_DIR}/mlir-opt
  -pass-pipeline ${TOSA_PIPELINE} |
  ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt
  ${SUBGRAPH_PREFILL_PASSES} |
  ${LLVM_TOOLS_BIN_DIR}/mlir-translate -mlir-to-llvmir |
  ${LLVM_TOOLS_BIN_DIR}/llvm-as |
  ${LLVM_TOOLS_BIN_DIR}/llc ${LLC_TARGET_FLAGS} -filetype=obj -relocation-model=pic -O3
  -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill.o
  DEPENDS ${BUDDY_OPT_DEP} ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill.mlir
  COMMENT "Building subgraph_prefill.o"
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.o
  COMMAND ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt
  ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.mlir
  -simplify-tosa-reshape |
  ${LLVM_TOOLS_BIN_DIR}/mlir-opt
  -pass-pipeline ${TOSA_PIPELINE} |
  ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt
  ${FORWARD_PASSES} |
  ${LLVM_TOOLS_BIN_DIR}/mlir-translate -mlir-to-llvmir |
  ${LLVM_TOOLS_BIN_DIR}/llvm-as |
  ${LLVM_TOOLS_BIN_DIR}/llc ${LLC_TARGET_FLAGS} -filetype=obj -relocation-model=pic -O3
  -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.o
  DEPENDS ${BUDDY_OPT_DEP} ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.mlir
  COMMENT "Building forward_decode.o"
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode.o
  COMMAND ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode.mlir
  -simplify-tosa-reshape
  -simplify-tosa-matmul-scalar |
  ${LLVM_TOOLS_BIN_DIR}/mlir-opt
  -pass-pipeline ${TOSA_PIPELINE} |
  ${BUDDY_TOOLS_BINARY_DIR}/buddy-opt
  ${SUBGRAPH_DECODE_PASSES} |
  ${LLVM_TOOLS_BIN_DIR}/mlir-translate -mlir-to-llvmir |
  ${LLVM_TOOLS_BIN_DIR}/llvm-as |
  ${LLVM_TOOLS_BIN_DIR}/llc ${LLC_TARGET_FLAGS} -filetype=obj -relocation-model=pic -O3
  -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode.o
  DEPENDS ${BUDDY_OPT_DEP} ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode.mlir
  COMMENT "Building subgraph_decode.o"
  VERBATIM
)

# ==============================================================================
# Build library
# ==============================================================================
add_library(DSR1TP STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.o
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill.o
  ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.o
  ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode.o
)

set_target_properties(
  DSR1TP
  PROPERTIES
  LINKER_LANGUAGE C
)

# ==============================================================================
# Final executable
# ==============================================================================
add_executable(buddy-dsr1-slpit-run buddy-deepseek-r1-main.cpp)

set(DEEPSEEKR1_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(DEEPSEEKR1_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(buddy-dsr1-slpit-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="/home/user/lishiyang/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="/home/user/lishiyang/"
)

# target_include_directories(buddy-dsr1-slpit-run PRIVATE
#   "${MPICH_INCLUDE_DIR}"
# )

target_compile_options(buddy-dsr1-slpit-run PRIVATE ${TARGET_COMPILE_OPTS})

if(CROSS_COMPILE_RVV)
  target_link_options(buddy-dsr1-slpit-run PRIVATE ${TARGET_LINK_OPTS})
else()
  target_link_directories(buddy-dsr1-slpit-run PRIVATE ${LLVM_LIBRARY_DIR})
endif()

set(BUDDY_DEEPSEEKR1_LIBS
  DSR1TP
  "${MLIR_RUNNER_UTILS_LIB}"
  "${OPENMP_LIB}"
  # "${MPI_CXX_LIB}"
  # "${MPI_C_LIB}"
)

if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_DEEPSEEKR1_LIBS mimalloc)
endif()

target_link_libraries(buddy-dsr1-slpit-run PRIVATE
  ${BUDDY_DEEPSEEKR1_LIBS}
)
