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
