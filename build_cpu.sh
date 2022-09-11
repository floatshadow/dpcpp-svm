rm -rf build
CC=dpcpp

mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=$CC -DUSE_GPU=ON -DUSE_CUDA=OFF -DUSE_PAPI=OFF -DCMAKE_CXX_FLAGS=" -O3" -DOpenMP_CXX_FLAGS="-fopenmp" -DOpenMP_CXX_LIB_NAMES="libiomp5"  -DOpenMP_libiomp5_LIBRARY="/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64/libiomp5.so" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_MODULE_PATH=. .. && make -j
