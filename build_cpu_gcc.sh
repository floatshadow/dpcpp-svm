rm -rf build

CC=g++

mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=$CC -DUSE_GPU=OFF -DUSE_CUDA=OFF -DUSE_PAPI=OFF -DCMAKE_CXX_FLAGS="-fopenmp -O3" -DCMAKE_MODULE_PATH=. .. && make -j

