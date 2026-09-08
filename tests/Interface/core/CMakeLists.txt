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

_add_test_executable(buddy-rax-execution-session-test
  RaxExecutionSessionTest.cpp
  "${CMAKE_SOURCE_DIR}/models/deepseek_r1/DeepSeekR1RaxSession.cpp"
  LINK_LIBS
    buddy_runtime_execution
)
target_include_directories(buddy-rax-execution-session-test PRIVATE
  "${CMAKE_SOURCE_DIR}/models/deepseek_r1/include"
)
target_compile_definitions(buddy-rax-execution-session-test PRIVATE
  RAX_EXECUTION_SESSION_TEST_LIBRARY="$<TARGET_FILE:buddy-rax-executor-test-kernels>"
)
add_dependencies(buddy-rax-execution-session-test
  buddy-rax-executor-test-kernels
)

_add_test_executable(buddy-deepseek-r1-rax-generation-test
  DeepSeekR1RaxGenerationTest.cpp
  "${CMAKE_SOURCE_DIR}/models/deepseek_r1/DeepSeekR1RaxSession.cpp"
  LINK_LIBS
    buddy_runtime_execution
    buddy_runtime_llm
)
target_include_directories(buddy-deepseek-r1-rax-generation-test PRIVATE
  "${CMAKE_SOURCE_DIR}/models/deepseek_r1/include"
)
target_compile_definitions(buddy-deepseek-r1-rax-generation-test PRIVATE
  DEEPSEEK_RAX_GENERATION_TEST_LIBRARY="$<TARGET_FILE:buddy-rax-executor-test-kernels>"
)
add_dependencies(buddy-deepseek-r1-rax-generation-test
  buddy-rax-executor-test-kernels
)

set(RAX_WRAPPER_ABI_TEST_OBJECT
  "${CMAKE_CURRENT_BINARY_DIR}/RaxWrapperAbiTest.o")
add_custom_command(
  OUTPUT "${RAX_WRAPPER_ABI_TEST_OBJECT}"
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_SOURCE_DIR}/tools/buddy-codegen/compile_pipeline.py"
          --config
          "${CMAKE_CURRENT_SOURCE_DIR}/Inputs/RaxWrapperAbiTestConfig.json"
          --input
          "${CMAKE_CURRENT_SOURCE_DIR}/Inputs/RaxWrapperAbiTest.mlir"
          --output "${RAX_WRAPPER_ABI_TEST_OBJECT}"
          --pipeline forward
          --tp-wrapper-out-params
          --buddy-opt "$<TARGET_FILE:buddy-opt>"
          --llvm-tools-dir "${LLVM_TOOLS_BINARY_DIR}"
          --llc-attrs=-mcpu=native
  DEPENDS
    buddy-opt
    "${CMAKE_SOURCE_DIR}/tools/buddy-codegen/compile_pipeline.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/Inputs/RaxWrapperAbiTestConfig.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/Inputs/RaxWrapperAbiTest.mlir"
  COMMENT "Building focused DeepSeek TP wrapper ABI object"
  VERBATIM
)
set_source_files_properties("${RAX_WRAPPER_ABI_TEST_OBJECT}" PROPERTIES
  EXTERNAL_OBJECT TRUE
  GENERATED TRUE
)

add_library(buddy-rax-wrapper-abi-test-kernels SHARED
  Inputs/RaxWrapperAbiTestShim.cpp
  "${RAX_WRAPPER_ABI_TEST_OBJECT}"
)

_add_test_executable(buddy-rax-wrapper-abi-test
  RaxWrapperAbiTest.cpp
  LINK_LIBS
    buddy_runtime_execution
)
target_compile_definitions(buddy-rax-wrapper-abi-test PRIVATE
  RAX_WRAPPER_ABI_TEST_LIBRARY="$<TARGET_FILE:buddy-rax-wrapper-abi-test-kernels>"
)
add_dependencies(buddy-rax-wrapper-abi-test
  buddy-rax-wrapper-abi-test-kernels
)

_add_test_executable(buddy-rax-collective-executor-test
  RaxCollectiveExecutorTest.cpp
  LINK_LIBS
    buddy_runtime_execution
)
target_compile_definitions(buddy-rax-collective-executor-test PRIVATE
  RAX_EXECUTOR_TEST_LIBRARY="$<TARGET_FILE:buddy-rax-executor-test-kernels>"
)
add_dependencies(buddy-rax-collective-executor-test
  buddy-rax-executor-test-kernels
)

if(BUDDY_RUNTIME_ENABLE_MPI)
  _add_test_executable(buddy-deepseek-r1-rax-runner-test
    DeepSeekR1RaxRunnerTest.cpp
    "${CMAKE_SOURCE_DIR}/models/deepseek_r1/DeepSeekR1RaxRunner.cpp"
    "${CMAKE_SOURCE_DIR}/models/deepseek_r1/DeepSeekR1RaxSession.cpp"
    LINK_LIBS
      buddy_runtime_execution
      buddy_runtime_mpi
  )
  target_include_directories(buddy-deepseek-r1-rax-runner-test PRIVATE
    "${CMAKE_SOURCE_DIR}/models/deepseek_r1/include"
  )
  target_compile_definitions(buddy-deepseek-r1-rax-runner-test PRIVATE
    DEEPSEEK_RAX_RUNNER_TEST_LIBRARY="$<TARGET_FILE:buddy-rax-executor-test-kernels>"
  )
  add_dependencies(buddy-deepseek-r1-rax-runner-test
    buddy-rax-executor-test-kernels
  )
endif()
