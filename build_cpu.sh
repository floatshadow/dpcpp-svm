rm -rf build

CC=icpx

mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=$CC -DUSE_GPU=OFF -DUSE_CUDA=OFF -DUSE_PAPI=OFF -DCMAKE_CXX_FLAGS="-fiopenmp" -DCMAKE_MODULE_PATH=. .. && make -j


