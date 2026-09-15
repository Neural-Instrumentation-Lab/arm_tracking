mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())") ..
cmake --build . -j
mv *.so ../.