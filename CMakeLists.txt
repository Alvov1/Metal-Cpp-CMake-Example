cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(MetalCppExample)

add_subdirectory(metal-cmake)
add_executable(MetalCppExample main.cpp)
target_link_libraries(MetalCppExample METAL_CPP)

