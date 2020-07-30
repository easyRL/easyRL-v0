cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(RLFramework)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(drqnNative SHARED drqnNative.cpp drqnReplayBuffer.cpp)
target_compile_options (drqnNative PRIVATE -fPIC -Wall)
target_link_libraries(drqnNative "${TORCH_LIBRARIES}")
set_property(TARGET drqnNative PROPERTY CXX_STANDARD 14)
