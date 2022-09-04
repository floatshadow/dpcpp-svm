//
// Created by jiashuai on 17-9-16.
//

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <thundersvm/syncmem.h>

namespace thunder {
    size_t SyncMem::total_memory_size = 0;
    SyncMem::SyncMem() : device_ptr(nullptr), host_ptr(nullptr), size_(0), head_(UNINITIALIZED), own_device_data(false),
                         own_host_data(false) {

    }

    SyncMem::SyncMem(size_t size) : device_ptr(nullptr), host_ptr(nullptr), size_(size), head_(UNINITIALIZED),
                                    own_device_data(false), own_host_data(false) {

    }

    SyncMem::~SyncMem() try {
        if (this->head_ != UNINITIALIZED) {
            this->head_ = UNINITIALIZED;
            if (own_host_data || own_device_data) total_memory_size -= size_;
            if (host_ptr && own_host_data) {
                free_host(host_ptr);
                host_ptr = nullptr;
            }
#ifdef USE_CUDA
            if (device_ptr && own_device_data) {
                /*
                DPCT1001:16: The statement could not be removed.
                */
                /*
                DPCT1002:17: Special case error handling if-stmt was detected.
                You may need to rewrite this code.
                */
                /*
                DPCT1003:18: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                0;
                device_ptr = nullptr;
            }
#endif
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    void *SyncMem::host_data() {
        to_host();
        return host_ptr;
    }

    void *SyncMem::device_data() {
#ifdef USE_CUDA
        to_device();
#else
        NO_GPU;
#endif
        return device_ptr;
    }

    size_t SyncMem::size() const {
        return size_;
    }

    SyncMem::HEAD SyncMem::head() const {
        return head_;
    }

    void SyncMem::to_host() try {
        switch (head_) {
            case UNINITIALIZED:
                malloc_host(&host_ptr, size_);
                memset(host_ptr, 0, size_);
                head_ = HOST;
                own_host_data = true;
                total_memory_size += size_;
                break;
            case DEVICE:
#ifdef USE_CUDA
                if (nullptr == host_ptr) {
                    /*
                    DPCT1003:21: Migrated API does not return error code. (*, 0)
                    is inserted. You may need to rewrite this code.
                    */
                    /*
                    DPCT1001:19: The statement could not be removed.
                    */
                    /*
                    DPCT1002:20: Special case error handling if-stmt was
                    detected. You may need to rewrite this code.
                    */
                    0;
                    /*
                    DPCT1001:22: The statement could not be removed.
                    */
                    /*
                    DPCT1002:23: Special case error handling if-stmt was
                    detected. You may need to rewrite this code.
                    */
                    /*
                    DPCT1003:24: Migrated API does not return error code. (*, 0)
                    is inserted. You may need to rewrite this code.
                    */
                    0;
                    own_host_data = true;
                }
                /*
                DPCT1001:25: The statement could not be removed.
                */
                /*
                DPCT1002:26: Special case error handling if-stmt was detected.
                You may need to rewrite this code.
                */
                /*
                DPCT1003:27: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                0;
                head_ = HOST;
#else
                NO_GPU;
#endif
                break;
            case HOST:;
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    void SyncMem::to_device() try {
#ifdef USE_CUDA
        switch (head_) {
            case UNINITIALIZED:
                /*
                DPCT1003:30: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                /*
                DPCT1001:28: The statement could not be removed.
                */
                /*
                DPCT1002:29: Special case error handling if-stmt was detected.
                You may need to rewrite this code.
                */
                0;
                /*
                DPCT1001:31: The statement could not be removed.
                */
                /*
                DPCT1002:32: Special case error handling if-stmt was detected.
                You may need to rewrite this code.
                */
                /*
                DPCT1003:33: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                0;
                head_ = DEVICE;
                own_device_data = true;
                total_memory_size += size_;
                break;
            case HOST:
                if (nullptr == device_ptr) {
                    /*
                    DPCT1003:36: Migrated API does not return error code. (*, 0)
                    is inserted. You may need to rewrite this code.
                    */
                    /*
                    DPCT1001:34: The statement could not be removed.
                    */
                    /*
                    DPCT1002:35: Special case error handling if-stmt was
                    detected. You may need to rewrite this code.
                    */
                    0;
                    /*
                    DPCT1001:37: The statement could not be removed.
                    */
                    /*
                    DPCT1002:38: Special case error handling if-stmt was
                    detected. You may need to rewrite this code.
                    */
                    /*
                    DPCT1003:39: Migrated API does not return error code. (*, 0)
                    is inserted. You may need to rewrite this code.
                    */
                    0;
                    own_device_data = true;
                }
                /*
                DPCT1001:40: The statement could not be removed.
                */
                /*
                DPCT1002:41: Special case error handling if-stmt was detected.
                You may need to rewrite this code.
                */
                /*
                DPCT1003:42: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                0;
                head_ = DEVICE;
                break;
            case DEVICE:;
        }
#else
        NO_GPU;
#endif
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    void SyncMem::set_host_data(void *data) {
        CHECK_NOTNULL(data);
        if (own_host_data) {
            free_host(host_ptr);
            total_memory_size -= size_;
        }
        host_ptr = data;
        own_host_data = false;
        head_ = HEAD::HOST;
    }

    void SyncMem::set_device_data(void *data) try {
#ifdef USE_CUDA
        CHECK_NOTNULL(data);
        if (own_device_data) {
            /*
            DPCT1001:43: The statement could not be removed.
            */
            /*
            DPCT1002:44: Special case error handling if-stmt was detected. You
            may need to rewrite this code.
            */
            /*
            DPCT1003:45: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            0;
            total_memory_size -= size_;
        }
        device_ptr = data;
        own_device_data = false;
        head_ = HEAD::DEVICE;
#else
        NO_GPU;
#endif
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
}
