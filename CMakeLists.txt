cmake_minimum_required(VERSION 3.15)
project(garrido_brain LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(pybind11 REQUIRED)

pybind11_add_module(garrido_brain src/bindings.cpp)
target_include_directories(garrido_brain PRIVATE include)
