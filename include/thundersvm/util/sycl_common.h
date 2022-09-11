#ifndef THUNDERSVM_SYCL_COMMON_H
#define THUNDERSVM_SYCL_COMMON_H

/// \file sycl_common.h
/// \brief sycl global settings, not in common.h 
/// in case of the pollution thus reduce the compilation time

#include <cstddef>
#include <thundersvm/config.h>

#ifdef USE_ONEAPI
#include <CL/sycl.hpp>
#endif

namespace thunder{
#ifdef USE_ONEAPI
    inline sycl::default_selector selector;
    inline sycl::queue sycl_q(selector);
    inline sycl::queue &get_sycl_queue() { return sycl_q; }
#endif
} // end namespace thunder

#endif // THUNDERSVM_SYCL_COMMON_H