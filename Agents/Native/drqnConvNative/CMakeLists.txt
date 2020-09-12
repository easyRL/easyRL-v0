cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(RLFramework)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_library(drqnConvNative SHARED drqnConvNative.cpp drqnConvReplayBuffer.cpp)
target_compile_options (drqnConvNative PRIVATE -fPIC -Wall)
target_link_libraries(drqnConvNative "${TORCH_LIBRARIES}")
set_property(TARGET drqnConvNative PROPERTY CXX_STANDARD 14)
