cmake_minimum_required(VERSION 3.25)
project(dpa4c_runtime LANGUAGES NONE)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(DPA_WORK_DIR "" CACHE PATH "Disposable build directory under /workspace")
set(DPA_STAGE_DIR "" CACHE PATH "Independent runtime directory under /workspace")
set(DPA_JOBS 8 CACHE STRING "Maximum compiler concurrency")
set(DPA_INCREMENTAL OFF CACHE BOOL "Reuse this project's own build tree")
add_custom_target(runtime_all
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/build/build_runtime.py
    --root ${CMAKE_CURRENT_SOURCE_DIR} --build-dir ${DPA_WORK_DIR}
    --stage-dir ${DPA_STAGE_DIR} --config ${CMAKE_BUILD_TYPE}
    --jobs ${DPA_JOBS} $<$<BOOL:${DPA_INCREMENTAL}>:--incremental>
  USES_TERMINAL VERBATIM COMMAND_EXPAND_LISTS)
add_custom_target(verify_runtime
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/manifest/verify_runtime.py
    --root ${CMAKE_CURRENT_SOURCE_DIR} --stage-dir ${DPA_STAGE_DIR}
  USES_TERMINAL VERBATIM)
