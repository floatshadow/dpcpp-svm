#ifndef THUNDERSVM_SYCL_COMMON_H
#define THUNDERSVM_SYCL_COMMON_H

/// \file sycl_common.h
/// \brief sycl global settings, not in common.h 
/// in case of the pollution thus reduce the compilation time

#include <cstddef>
#include <thundersvm/config.h>

#include <CL/sycl.hpp>
#include <iostream>

namespace thunder{
    inline auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                std::cerr << "Caught asynchronous SYCL exception during GEMM:" << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };
#ifdef USE_GPU
    inline sycl::default_selector selector;
#else 
    inline sycl::cpu_selector selector;
#endif
    inline sycl::queue sycl_q(selector, exception_handler);
    inline sycl::queue &get_sycl_queue() { return sycl_q; }
} // end namespace thunder

#endif // THUNDERSVM_SYCL_COMMON_H