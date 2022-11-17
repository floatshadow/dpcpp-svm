#ifndef THUNDERSVM_SYCL_COMMON_H
#define THUNDERSVM_SYCL_COMMON_H

/// \file sycl_common.h
/// \brief sycl global settings, not in common.h 
/// in case of the pollution thus reduce the compilation time

#include <cstddef>
#include <thundersvm/config.h>

#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

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
    inline sycl::gpu_selector selector;
#else 
    inline sycl::cpu_selector selector;
#endif
    inline sycl::queue sycl_q(selector, exception_handler);


    inline sycl::queue &get_sycl_queue() { return sycl_q; }
        inline void get_device_name() {
        auto &q = get_sycl_queue();
        std::cout << "Device: "
            << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
    }
    inline void get_device_local_memory_size() {
        auto &q = get_sycl_queue();
        std::cout << "Local Memory Size: "
            << q.get_device().get_info<sycl::info::device::local_mem_size>()
            << std::endl;
    }

    /// \brief This namespace serve the purpose for utilizing muti-tile intel GPU
    namespace MutiTile {
        inline std::vector<sycl::queue> sub_queues;

        inline void create_sub_devices() {
            auto &root_queue = get_sycl_queue();
            auto sub_devices = root_queue.get_device().create_sub_devices<
                sycl::info::partition_property::partition_by_affinity_domain>(
                sycl::info::partition_affinity_domain::next_partitionable);
            sycl::context tile_context(sub_devices);
            for (auto &D : sub_devices) {
                sub_queues.emplace_back(tile_context, D, exception_handler);
            }
        }
        inline std::vector<sycl::queue> &get_sub_queues() { return sub_queues; }
    } // end namespace MultiTile
} // end namespace thunder

#endif // THUNDERSVM_SYCL_COMMON_H