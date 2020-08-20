cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(RLFramework)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(reinforceNative SHARED reinforce.cpp buffer.cpp)
target_compile_options (reinforceNative PRIVATE -fPIC -Wall)
target_link_libraries(reinforceNative "${TORCH_LIBRARIES}")
set_property(TARGET reinforceNative PROPERTY CXX_STANDARD 14)
