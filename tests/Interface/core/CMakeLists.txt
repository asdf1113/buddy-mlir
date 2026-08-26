_add_test_executable(buddy-container-test
  ContainerTest.cpp
)

if(BUDDY_ENABLE_OPENCV)
  find_package(OpenCV REQUIRED CONFIG)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

if(BUDDY_MLIR_ENABLE_DIP_LIB AND BUDDY_ENABLE_OPENCV)
  set(DIP_LIBS ${JPEG_LIBRARY} ${PNG_LIBRARY})
  _add_test_executable(buddy-image-container-test
    ImageContainerTest.cpp
    LINK_LIBS
      ${OpenCV_LIBS}
      ${DIP_LIBS}
  )
endif()

if(BUDDY_MLIR_ENABLE_DIP_LIB)
  set(NEW_DIP_LIBS "")
  if(BUDDY_ENABLE_PNG)
    list(APPEND NEW_DIP_LIBS ${PNG_LIBRARIES})
    _add_test_executable(buddy-new-image-container-test-png
      NewImageContainerTestPng.cpp
      LINK_LIBS
        ${NEW_DIP_LIBS}
    )
  endif()
  _add_test_executable(buddy-new-image-container-test-bmp
    NewImageContainerTestBmp.cpp
    LINK_LIBS
      ${NEW_DIP_LIBS}
  )
endif()

_add_test_executable(buddy-audio-container-test
  AudioContainerTest.cpp
)

_add_test_executable(buddy-text-container-test
  TextContainerTest.cpp
)

add_library(buddy-rax-executor-test-kernels SHARED
  Inputs/RaxExecutorTestKernels.cpp
)

_add_test_executable(buddy-rax-executor-test
  RaxExecutorTest.cpp
  LINK_LIBS
    buddy_runtime_execution
)
target_compile_definitions(buddy-rax-executor-test PRIVATE
  RAX_EXECUTOR_TEST_LIBRARY="$<TARGET_FILE:buddy-rax-executor-test-kernels>"
)
add_dependencies(buddy-rax-executor-test buddy-rax-executor-test-kernels)

_add_test_executable(buddy-rax-collective-executor-test
  RaxCollectiveExecutorTest.cpp
  LINK_LIBS
    buddy_runtime_execution
)

set(RAX_EXECUTOR_MLIR_E2E_OBJECT
  "${CMAKE_CURRENT_BINARY_DIR}/RaxExecutorMlirE2E.o")
add_custom_command(
  OUTPUT "${RAX_EXECUTOR_MLIR_E2E_OBJECT}"
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
          ${CMAKE_CURRENT_SOURCE_DIR}/Inputs/RaxExecutorMlirE2E.mlir
          -convert-scf-to-cf
          -llvm-request-c-wrappers
          -convert-arith-to-llvm
          -finalize-memref-to-llvm
          -convert-cf-to-llvm
          -convert-func-to-llvm
          -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic
          -O0 -o "${RAX_EXECUTOR_MLIR_E2E_OBJECT}"
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Inputs/RaxExecutorMlirE2E.mlir
  COMMENT "Building RaxExecutorMlirE2E.o"
  VERBATIM
)
set_source_files_properties("${RAX_EXECUTOR_MLIR_E2E_OBJECT}" PROPERTIES
  EXTERNAL_OBJECT TRUE
  GENERATED TRUE
)

add_library(buddy-rax-executor-mlir-e2e-test-kernels SHARED
  Inputs/RaxExecutorMlirE2EShims.cpp
  "${RAX_EXECUTOR_MLIR_E2E_OBJECT}"
)

set(RAX_EXECUTOR_MLIR_E2E_LIBRARY
  "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}buddy-rax-executor-mlir-e2e-test-kernels${CMAKE_SHARED_LIBRARY_SUFFIX}")
configure_file(
  Inputs/RaxExecutorMlirE2E.rhal.mlir.in
  Inputs/RaxExecutorMlirE2E.rhal.mlir
  @ONLY
)

_add_test_executable(buddy-rax-executor-mlir-e2e-test
  RaxExecutorMlirE2ETest.cpp
  LINK_LIBS
    buddy_runtime_execution
)
add_dependencies(buddy-rax-executor-mlir-e2e-test
  buddy-rax-executor-mlir-e2e-test-kernels
)

if(BUDDY_RUNTIME_ENABLE_MPI)
  add_library(buddy-rax-executor-mpi-e2e-test-kernels SHARED
    Inputs/RaxExecutorMpiE2EKernels.cpp
  )

  set(RAX_EXECUTOR_MPI_E2E_LIBRARY
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}buddy-rax-executor-mpi-e2e-test-kernels${CMAKE_SHARED_LIBRARY_SUFFIX}")
  configure_file(
    Inputs/RaxExecutorMpiE2ERank0.rhal.mlir.in
    Inputs/RaxExecutorMpiE2ERank0.rhal.mlir
    @ONLY
  )
  configure_file(
    Inputs/RaxExecutorMpiE2ERank1.rhal.mlir.in
    Inputs/RaxExecutorMpiE2ERank1.rhal.mlir
    @ONLY
  )

  _add_test_executable(buddy-rax-executor-mpi-e2e-test
    RaxExecutorMpiE2ETest.cpp
    LINK_LIBS
      buddy_runtime_execution
      buddy_runtime_mpi
  )
  add_dependencies(buddy-rax-executor-mpi-e2e-test
    buddy-rax-executor-mpi-e2e-test-kernels
  )

  _add_test_executable(buddy-rax-variable-collectives-mpi-e2e-test
    RaxVariableCollectivesMpiE2ETest.cpp
    LINK_LIBS
      buddy_runtime_execution
      buddy_runtime_mpi
  )

  if(BUDDY_MLIR_ENABLE_PYTHON_PACKAGES)
    set(STAGE2E_ARTIFACT_DIR "${CMAKE_CURRENT_BINARY_DIR}/Stage2E")
    set(STAGE2E_RANK0_LIBRARY
      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}buddy-rax-tp-frontend-rank0${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(STAGE2E_RANK1_LIBRARY
      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}buddy-rax-tp-frontend-rank1${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(STAGE2E_GENERATED_FILES
      "${STAGE2E_ARTIFACT_DIR}/materialization.json"
      "${STAGE2E_ARTIFACT_DIR}/rank0.rhal.mlir"
      "${STAGE2E_ARTIFACT_DIR}/rank1.rhal.mlir"
    )
    foreach(rank RANGE 0 1)
      list(APPEND STAGE2E_GENERATED_FILES
        "${STAGE2E_ARTIFACT_DIR}/rank${rank}/rank${rank}_params_float32.data"
        "${STAGE2E_ARTIFACT_DIR}/rank${rank}/segment0.mlir"
        "${STAGE2E_ARTIFACT_DIR}/rank${rank}/segment1.mlir"
        "${STAGE2E_ARTIFACT_DIR}/rank${rank}/wrapper0.mlir"
        "${STAGE2E_ARTIFACT_DIR}/rank${rank}/wrapper1.mlir"
      )
    endforeach()
    add_custom_command(
      OUTPUT ${STAGE2E_GENERATED_FILES}
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${BUDDY_MLIR_PYTHON_PACKAGES_DIR}:$ENV{PYTHONPATH}"
              "CUDA_VISIBLE_DEVICES="
              ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/Inputs/Stage2EGenerate.py
              --output-dir "${STAGE2E_ARTIFACT_DIR}"
              --rank0-library "${STAGE2E_RANK0_LIBRARY}"
              --rank1-library "${STAGE2E_RANK1_LIBRARY}"
      DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/Inputs/Stage2EGenerate.py
        python-package-buddy
        python-package-buddy-mlir
      COMMENT "Generating Stage 2E frontend segments, wrappers, packs, and RHAL"
      VERBATIM
    )
    add_custom_target(buddy-rax-tp-frontend-artifacts
      DEPENDS ${STAGE2E_GENERATED_FILES})

    function(_stage2e_compile_object rank stem output_variable)
      set(object "${STAGE2E_ARTIFACT_DIR}/rank${rank}/${stem}.o")
      add_custom_command(
        OUTPUT "${object}"
        COMMAND $<TARGET_FILE:buddy-opt>
                "${STAGE2E_ARTIFACT_DIR}/rank${rank}/${stem}.mlir"
                "-pass-pipeline=builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))" |
                $<TARGET_FILE:buddy-opt>
                -eliminate-empty-tensors
                -empty-tensor-to-alloc-tensor
                -convert-elementwise-to-linalg
                "-one-shot-bufferize=bufferize-function-boundaries"
                -expand-strided-metadata
                -bufferization-lower-deallocations
                -convert-bufferization-to-memref
                -convert-linalg-to-loops
                -lower-affine
                -convert-scf-to-cf
                -llvm-request-c-wrappers
                -convert-arith-to-llvm
                -finalize-memref-to-llvm
                -convert-cf-to-llvm
                -convert-func-to-llvm
                -reconcile-unrealized-casts |
                ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
                ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj
                -relocation-model=pic -O0 -o "${object}"
        DEPENDS
          "${STAGE2E_ARTIFACT_DIR}/rank${rank}/${stem}.mlir"
          buddy-opt
        COMMENT "Compiling Stage 2E rank ${rank} ${stem}.mlir"
        VERBATIM
      )
      set_source_files_properties("${object}" PROPERTIES
        EXTERNAL_OBJECT TRUE
        GENERATED TRUE
      )
      set(${output_variable} "${object}" PARENT_SCOPE)
    endfunction()

    foreach(rank RANGE 0 1)
      _stage2e_compile_object(${rank} segment0 stage2e_segment0_object)
      _stage2e_compile_object(${rank} segment1 stage2e_segment1_object)
      _stage2e_compile_object(${rank} wrapper0 stage2e_wrapper0_object)
      _stage2e_compile_object(${rank} wrapper1 stage2e_wrapper1_object)
      add_library(buddy-rax-tp-frontend-rank${rank} SHARED
        Inputs/Stage2EFrontendMpiE2EShims.cpp
        "${stage2e_segment0_object}"
        "${stage2e_segment1_object}"
        "${stage2e_wrapper0_object}"
        "${stage2e_wrapper1_object}"
      )
      add_dependencies(buddy-rax-tp-frontend-rank${rank}
        buddy-rax-tp-frontend-artifacts)
    endforeach()

    set(STAGE2E_RANK0_RAX "${STAGE2E_ARTIFACT_DIR}/rank0.rax")
    set(STAGE2E_RANK1_RAX "${STAGE2E_ARTIFACT_DIR}/rank1.rax")
    add_custom_command(
      OUTPUT "${STAGE2E_RANK0_RAX}" "${STAGE2E_RANK1_RAX}"
      COMMAND $<TARGET_FILE:rax-pack>
              "${STAGE2E_ARTIFACT_DIR}/rank0.rhal.mlir"
              -o "${STAGE2E_RANK0_RAX}"
      COMMAND $<TARGET_FILE:rax-pack>
              "${STAGE2E_ARTIFACT_DIR}/rank1.rhal.mlir"
              -o "${STAGE2E_RANK1_RAX}"
      DEPENDS
        "${STAGE2E_ARTIFACT_DIR}/rank0.rhal.mlir"
        "${STAGE2E_ARTIFACT_DIR}/rank1.rhal.mlir"
        buddy-rax-tp-frontend-rank0
        buddy-rax-tp-frontend-rank1
        rax-pack
      COMMENT "Packing Stage 2E rank-specific RAX files"
      VERBATIM
    )
    add_custom_target(buddy-rax-tp-frontend-rax
      DEPENDS "${STAGE2E_RANK0_RAX}" "${STAGE2E_RANK1_RAX}")

    _add_test_executable(buddy-rax-tp-frontend-mpi-e2e-test
      RaxTpFrontendMpiE2ETest.cpp
      LINK_LIBS
        buddy_runtime_execution
        buddy_runtime_mpi
    )
    add_dependencies(buddy-rax-tp-frontend-mpi-e2e-test
      buddy-rax-tp-frontend-rax)
  endif()
endif()
