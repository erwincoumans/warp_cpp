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



if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
	cmake_policy(SET CMP0017 NEW)
	cmake_policy(SET CMP0042 NEW)
	cmake_policy(SET CMP0057 NEW)
endif(COMMAND cmake_policy)
    
#this line has to appear before 'PROJECT' in order to be able to disable incremental linking
SET(MSVC_INCREMENTAL_DEFAULT ON)

PROJECT(warp_cpp)

add_executable(example_add_float_array example_add_float_array.cpp )

target_link_libraries(example_add_float_array  ${CMAKE_DL_LIBS})
# target_compile_definitions(pendulum_example_tiny_gui PRIVATE USE_TINY )
# target_include_directories(pendulum_example_tiny_gui PRIVATE ../src )

