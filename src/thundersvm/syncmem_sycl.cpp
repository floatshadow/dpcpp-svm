// #include "thundersvm/util/common.h"   This file has already been included in thundersvm.h?
#include <thundersvm/syncmem.h>
#include <thundersvm/util/sycl_common.h>

namespace thunder
{
size_t SyncMem::total_memory_size = 0;
SyncMem::SyncMem()
    : device_ptr(nullptr), host_ptr(nullptr), size_(0), head_(UNINITIALIZED), own_device_data(false),
      own_host_data(false)
{
}

SyncMem::SyncMem(size_t size)
    : device_ptr(nullptr), host_ptr(nullptr), size_(size), head_(UNINITIALIZED), own_device_data(false),
      own_host_data(false)
{
}

SyncMem::~SyncMem()
{
    if (this->head_ != UNINITIALIZED)
    {
        this->head_ = UNINITIALIZED;
        if (own_host_data || own_device_data)
            total_memory_size -= size_;
        if (host_ptr && own_host_data)
        {
            free_host(host_ptr);
            host_ptr = nullptr;
        }
        if (device_ptr && own_device_data)
        {
            sycl::free(device_ptr, thunder::get_sycl_queue());
            device_ptr = nullptr;
        }
    }
}

void *SyncMem::host_data()
{
    to_host();
    return host_ptr;
}

void *SyncMem::device_data()
{
    to_device();
    return device_ptr;
}

size_t SyncMem::size() const
{
    return size_;
}

SyncMem::HEAD SyncMem::head() const
{
    return head_;
}

void SyncMem::to_host()
{
#ifdef USE_GPU
    switch (head_)
    {
    case UNINITIALIZED:
        host_ptr = malloc_host(size_, thunder::get_sycl_queue());
        memset(host_ptr, 0, size_);
        head_ = HOST;
        own_host_data = true;
        total_memory_size += size_;
        break;
    case DEVICE:
        if (nullptr == host_ptr)
        {
            host_ptr = sycl::malloc_host(size_, thunder::get_sycl_queue());
            thunder::get_sycl_queue().memset(host_ptr, 0, size_);
            own_host_data = true;
        }
        thunder::get_sycl_queue().memcpy(host_ptr, device_ptr, size_);
        head_ = HOST;
        break;
    case HOST:;
    }
#else
    switch (head_)
    {
    case UNINITIALIZED: {
        auto &q = thunder::get_sycl_queue();
        host_ptr = malloc_host(size_, q);
        q.memset(host_ptr, 0, size_);
        head_ = HOST;
        own_host_data = true;
        total_memory_size += size_;
    }
    case DEVICE:;
    case HOST:;
    }
#endif
}

void SyncMem::to_device()
{
#ifdef USE_GPU
    switch (head_)
    {
    case UNINITIALIZED:
        device_ptr = sycl::malloc_device(size_, thunder::get_sycl_queue());
        thunder::get_sycl_queue().memset(device_ptr, 0, size_);
        head_ = DEVICE;
        own_device_data = true;
        total_memory_size += size_;
        break;
    case HOST:
        if (nullptr == device_ptr)
        {
            device_ptr = sycl::malloc_device(size_, thunder::get_sycl_queue());
            thunder::get_sycl_queue().memset(device_ptr, 0, size_);
            own_device_data = true;
        }
        thunder::get_sycl_queue().memcpy(device_ptr, host_ptr, size_);
        head_ = DEVICE;
        break;
    case DEVICE:;
    }
#else
    switch (head_)
    {
    case UNINITIALIZED: {
        auto &q = thunder::get_sycl_queue();
        host_ptr = device_ptr = sycl::malloc_host(size_, q);
        q.memset(device_ptr, 0, size_);
        head_ = HOST;
        /// @attention Here "device" is CPU
        own_host_data = true;
        own_device_data = false;
        total_memory_size += size_;
        break;
    }
    case HOST:;
    case DEVICE:;
    }
#endif
}

void SyncMem::set_host_data(void *data)
{
    CHECK_NOTNULL(data);
    if (own_host_data)
    {
        free_host(host_ptr);
        total_memory_size -= size_;
    }
    host_ptr = data;
    own_host_data = false;
    head_ = HEAD::HOST;
}

void SyncMem::set_device_data(void *data)
{
    CHECK_NOTNULL(data);
    if (own_device_data)
    {
        sycl::free(device_data(), thunder::get_sycl_queue());
        total_memory_size -= size_;
    }
    device_ptr = data;
    own_device_data = false;
    head_ = HEAD::DEVICE;
}
} // namespace thunder
