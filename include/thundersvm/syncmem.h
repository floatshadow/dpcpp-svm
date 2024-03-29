//
// Created by jiashuai on 17-9-16.
//

#ifndef THUNDERSVM_SYNCMEM_H
#define THUNDERSVM_SYNCMEM_H

#include <thundersvm/thundersvm.h>
#include <thundersvm/util/sycl_common.h>

namespace thunder {
    inline void malloc_host(void **ptr, size_t size) {
#ifdef USE_GPU
        *ptr = sycl::malloc_host(size, thunder::get_sycl_queue());
#else
        *ptr = malloc(size);
#endif
    }

    inline void free_host(void *ptr) {
#ifdef USE_GPU
        sycl::free(ptr, thunder::get_sycl_queue());
#else
        free(ptr);
#endif
    }

    inline void device_mem_copy(void *dst, const void *src, size_t size) {
#ifdef USE_GPU
        thunder::get_sycl_queue().memcpy(dst, src, size).wait();
#else
        NO_GPU;
#endif
    }

    /**
     * @brief Auto-synced memory for CPU and GPU
     */
    class SyncMem {
    public:
        SyncMem();

        /**
         * create a piece of synced memory with given size. The GPU/CPU memory will not be allocated immediately, but
         * allocated when it is used at first time.
         * @param size the size of memory (in Bytes)
         */
        explicit SyncMem(size_t size);

        ~SyncMem();

        ///return raw host pointer
        void *host_data();

        ///return raw device pointer
        void *device_data();

        /**
         * set host data pointer to another host pointer, and its memory will not be managed by this class
         * @param data another host pointer
         */
        void set_host_data(void *data);

        /**
         * set device data pointer to another device pointer, and its memory will not be managed by this class
         * @param data another device pointer
         */
        void set_device_data(void *data);

        ///transfer data to host
        void to_host();

        ///transfer data to device
        void to_device();
        
        /// swap memory and ownership.
        void swap(SyncMem *rhs);

        ///return the size of memory
        size_t size() const;

        ///to determine the where the newest data locates in
        enum HEAD {
            HOST, DEVICE, UNINITIALIZED
        };

        HEAD head() const;

        static size_t get_total_memory_size() { return total_memory_size; }

        static void reset_memory_size() { total_memory_size = 0; }


    private:
        void *device_ptr;
        void *host_ptr;
        bool own_device_data;
        bool own_host_data;
        size_t size_;
        HEAD head_;
        static size_t total_memory_size;
    };
}
using thunder::SyncMem;
#endif //THUNDERSVM_SYNCMEM_H
