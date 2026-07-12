#!/bin/bash
set -e
cd "$(dirname "$0")/.."   # adjust path to wherever cerebellum.hpp/bindings.cpp live
c++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) \
  -I/usr/include/eigen3 \
  bindings.cpp -o garrido_brain_cpp$(python3-config --extension-suffix)