add_library(buddy_runtime_execution STATIC
  RaxExecutor.cpp
)
target_link_libraries(buddy_runtime_execution PUBLIC
  buddy_runtime_core
  ${CMAKE_DL_LIBS}
)
target_compile_features(buddy_runtime_execution PUBLIC cxx_std_17)

install(TARGETS buddy_runtime_execution
        EXPORT BuddyMLIRTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT buddy_runtime)
