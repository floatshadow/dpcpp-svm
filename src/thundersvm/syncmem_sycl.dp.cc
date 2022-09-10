#include <thundersvm/syncmem_sycl.h>

namespace thunder
{
size_t SyncMem::total_memory_size = 0;
SyncMem::SyncMem()
    : device_ptr(nullptr), host_ptr(nullptr), size_(0), head_(UNINITIALIZED), own_device_data(false),
      own_host_data(false)
{
}

SyncMem::SyncMem(sycl::queue q)
    : device_ptr(nullptr), host_ptr(nullptr), size_(0), head_(UNINITIALIZED), own_device_data(false),
      own_host_data(false), q(q)
{
}

SyncMem::SyncMem(sycl::queue q, size_t size)
    : device_ptr(nullptr), host_ptr(nullptr), size_(size), head_(UNINITIALIZED), own_device_data(false),
      own_host_data(false), q(q)
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
            free_host(q, host_ptr);
            host_ptr = nullptr;
        }
        if (device_ptr && own_device_data)
        {
            sycl::free(device_ptr, q);
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
    switch (head_)
    {
    case UNINITIALIZED:
        malloc_host(q, &host_ptr, size_);
        memset(host_ptr, 0, size_);
        head_ = HOST;
        own_host_data = true;
        total_memory_size += size_;
        break;
    case DEVICE:
        if (nullptr == host_ptr)
        {
            host_ptr = sycl::malloc_host(size_, q);
            q.memset(host_ptr, 0, size_);
            own_host_data = true;
        }
        q.memcpy(host_ptr, device_ptr, size_);
        head_ = HOST;
        break;
    case HOST:;
    }
}

void SyncMem::to_device()
{
    switch (head_)
    {
    case UNINITIALIZED:
        device_ptr = sycl::malloc_device(size_, q);
        q.memset(device_ptr, 0, size_);
        head_ = DEVICE;
        own_device_data = true;
        total_memory_size += size_;
        break;
    case HOST:
        if (nullptr == device_ptr)
        {
            device_ptr = sycl::malloc_device(size_, q);
            q.memset(device_ptr, 0, size_);
            own_device_data = true;
        }
        q.memcpy(device_ptr, host_ptr, size_);
        head_ = DEVICE;
        break;
    case DEVICE:;
    }
}

void SyncMem::set_host_data(void *data)
{
    CHECK_NOTNULL(data);
    if (own_host_data)
    {
        free_host(q, host_ptr);
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
        sycl::free(device_data(), q);
        total_memory_size -= size_;
    }
    device_ptr = data;
    own_device_data = false;
    head_ = HEAD::DEVICE;

}
} // namespace thunder
