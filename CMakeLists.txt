cmake_minimum_required(VERSION 3.12)

set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # For LLVM tooling
# set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()


if(DEFINED WARP_PATH)
  message(STATUS "WARP_PATH is defined with value: ${WARP_PATH}")
else()
  set(WARP_PATH "D:/dev/warp_cpp/third_party/warp")
  message(STATUS "WARP_PATH is not defined, using hard-coded path: ${WARP_PATH}")
endif()



if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
	cmake_policy(SET CMP0017 NEW)
	cmake_policy(SET CMP0042 NEW)
	cmake_policy(SET CMP0057 NEW)
endif(COMMAND cmake_policy)
    
#this line has to appear before 'PROJECT' in order to be able to disable incremental linking
SET(MSVC_INCREMENTAL_DEFAULT ON)

PROJECT(warp_cpp)

add_executable(example_add_float_array_cpu example_add_float_array_cpu.cpp )
target_link_libraries(example_add_float_array_cpu  ${CMAKE_DL_LIBS})
target_include_directories(example_add_float_array_cpu PRIVATE ${WARP_PATH} . )

# target_compile_definitions(example_add_float_array_cpu PRIVATE . )

add_executable(example_add_float_array_cuda example_add_float_array_cuda.cpp )
target_link_libraries(example_add_float_array_cuda  ${CMAKE_DL_LIBS})
target_include_directories(example_add_float_array_cuda PRIVATE ${WARP_PATH} . )

# target_compile_definitions(example_add_float_array_cuda PRIVATE . )

