cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(RLFramework)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB SOURCES "*.cpp")

add_library(deepQNative SHARED ${SOURCES})
target_compile_options (deepQNative PRIVATE -fPIC -Wall)
target_link_libraries(deepQNative "${TORCH_LIBRARIES}")
set_property(TARGET deepQNative PROPERTY CXX_STANDARD 14)
