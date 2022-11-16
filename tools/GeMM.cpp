#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <chrono>
#include <iostream>

using namespace sycl;
using namespace oneapi::mkl;
using namespace std::chrono;

constexpr size_t N = 50000;
constexpr size_t M = 10000;

int main() {
#ifdef USE_GPU
    gpu_selector selector;
#else
    cpu_selector selector;
#endif
    queue q(selector);
    std::cout << "Device: "
            << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
    
    float *A = malloc_device<float>(N * N, q);
    float *B = malloc_device<float>(N * M, q);
    float *C = malloc_device<float>(N * M, q);
    q.memset(A, 0, sizeof(float)*N*N);
    q.memset(B, 0, sizeof(float)*N*M);
    q.memset(C, 0, sizeof(float)*N*M);
    high_resolution_clock::time_point t0,t1;
    double dt0=0.0;

    t0 = high_resolution_clock::now();
    auto gemm = blas::column_major::gemm(
        q,
        transpose::nontrans, transpose::nontrans,
        N, M, N,
        1.0f, A, N,
        B, N, 
        0.0f, C, N, 
        {}
    );
    gemm.wait();
    t1 = high_resolution_clock::now();
    dt0 = dt0 + duration<double>(t1-t0).count();
    printf( "\nFOM: main loop : %11.6lf ms, %11.6lfs \n\n", dt0*1000, dt0 );
    double GFlops = 2 * N * N * M / (dt0 * 1) / 1000000000;
    printf("GFlops: %lf\n", GFlops);
}
