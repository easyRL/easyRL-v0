cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(RLFramework)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tensorboard_logger)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_library(ppoNative SHARED ppo.cpp buffer.cpp)
target_compile_options (ppoNative PRIVATE -fPIC -Wall)
target_link_libraries(ppoNative "${TORCH_LIBRARIES}" tensorboard_logger protobuf)
set_property(TARGET ppoNative PROPERTY CXX_STANDARD 14)
