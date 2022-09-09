//
// Created by jiashuai on 17-9-17.
//
#include "thundersvm/syncarray_sycl.h"

template <typename T> SyncArray<T>::SyncArray(sycl::queue q, size_t count) : mem(new SyncMem(q, sizeof(T) * count)), size_(count)
{
}

template <typename T> SyncArray<T>::~SyncArray()
{
    delete mem;
}

template <typename T> const T *SyncArray<T>::host_data() const
{
    to_host();
    return static_cast<T *>(mem->host_data());
}

template <typename T> const T *SyncArray<T>::device_data() const
{
    to_device();
    return static_cast<T *>(mem->device_data());
}

template <typename T> T *SyncArray<T>::host_data()
{
    to_host();
    return static_cast<T *>(mem->host_data());
}

template <typename T> T *SyncArray<T>::device_data()
{
    to_device();
    return static_cast<T *>(mem->device_data());
}

template <typename T> void SyncArray<T>::resize(sycl::queue q, size_t count)
{
    delete mem;
    mem = new SyncMem(q, sizeof(T) * count);
    this->size_ = count;
}

template <typename T> 
void SyncArray<T>::copy_from(sycl::queue q,const T *source, size_t count)
{
    thunder::device_mem_copy(q, mem->device_data(), source, sizeof(T) * count);
}

template <typename T> 
void SyncArray<T>::log(el::base::type::ostream_t &ostream) const
{
    int i;
    ostream << "[";
    for (i = 0; i < size() - 1 && i < el::base::consts::kMaxLogPerContainer - 1; ++i)
    {
        ostream << host_data()[i] << ",";
    }
    ostream << host_data()[i];
    ostream << "]";
}

template <typename T> void SyncArray<T>::copy_from(sycl::queue q, const SyncArray<T> &source)
{
    CHECK_EQ(size(), source.size()) << "destination and source count doesn't match";
    copy_from(q, source.device_data(), source.size());
}

template <typename T> void SyncArray<T>::mem_set(sycl::queue q, const T &value)
{
    q.memset(device_data(), value, mem_size());
}

template class SyncArray<int>;

template class SyncArray<float>;

template class SyncArray<double>;
