//
// Created by jiashuai on 17-9-21.
//
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "thundersvm/kernel/smo_kernel.h"

#include <dpct/dpl_utils.hpp>

#include <thundersvm/util/sycl_common.h>

namespace svm_kernel
{

    template <typename T>
    int get_block_min(const T *values, int *index, sycl::nd_item<3> item_ct1)
    {
        int tid = item_ct1.get_local_id(2);
        index[tid] = tid;
        /*
        DPCT1065:92: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        // block size is always the power of 2
        for (int offset = item_ct1.get_local_range(2) / 2; offset > 0;
             offset >>= 1)
        {
            if (tid < offset)
            {
                if (values[index[tid + offset]] < values[index[tid]])
                {
                    index[tid] = index[tid + offset];
                }
            }
            /*
            DPCT1065:93: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
        return index[0];
    }

    void
    c_smo_solve_kernel(const int *label, float_type *f_val, float_type *alpha, float_type *alpha_diff,
                       const int *working_set, int ws_size,
                       float_type Cp, float_type Cn, const kernel_type *k_mat_rows, const kernel_type *k_mat_diag, int row_len,
                       float_type eps,
                       float_type *diff, int max_iter,
                       sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
    {
        //"row_len" equals to the number of instances in the original training dataset.
        // allocate shared memory
        auto shared_mem = (int *)dpct_local;
        int *f_idx2reduce = shared_mem;                                                                             // temporary memory for reduction
        float_type *f_val2reduce = (float_type *)&shared_mem[ws_size];                                              // f values used for reduction.
        float_type *alpha_i_diff = (float_type *)&shared_mem[ws_size + ws_size * sizeof(float_type) / sizeof(int)]; // delta alpha_i
        float_type *alpha_j_diff = &alpha_i_diff[1];
        kernel_type *kd = (kernel_type *)&alpha_j_diff[1]; // diagonal elements for kernel matrix

        // index, f value and alpha for each instance
        int tid = item_ct1.get_local_id(2);
        int wsi = working_set[tid];
        kd[tid] = k_mat_diag[wsi];
        float_type y = label[wsi];
        float_type f = f_val[wsi];
        float_type a = alpha[wsi];
        float_type aold = a;
        /*
        DPCT1065:94: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        float_type local_eps;
        int numOfIter = 0;
        while (1)
        {
            // select fUp and fLow
            if (is_I_up(a, y, Cp, Cn))
                f_val2reduce[tid] = f;
            else
                f_val2reduce[tid] = INFINITY;
            int i = get_block_min(f_val2reduce, f_idx2reduce, item_ct1);
            float_type up_value = f_val2reduce[i];
            kernel_type kIwsI = k_mat_rows[row_len * i + wsi]; // K[i, wsi]
            /*
            DPCT1065:95: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            if (is_I_low(a, y, Cp, Cn))
                f_val2reduce[tid] = -f;
            else
                f_val2reduce[tid] = INFINITY;
            int j1 = get_block_min(f_val2reduce, f_idx2reduce, item_ct1);
            float_type low_value = -f_val2reduce[j1];

            float_type local_diff = low_value - up_value;
            if (numOfIter == 0)
            {
                local_eps = sycl::max((double)eps, (double)(0.1f * local_diff));
                if (tid == 0)
                {
                    diff[0] = local_diff;
                }
            }

            if (numOfIter > max_iter || local_diff < local_eps)
            {
                alpha[wsi] = a;
                alpha_diff[tid] = -(a - aold) * y;
                diff[1] = numOfIter;
                break;
            }
            /*
            DPCT1065:96: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // select j2 using second order heuristic
            if (-up_value > -f && (is_I_low(a, y, Cp, Cn)))
            {
                float_type aIJ = kd[i] + kd[tid] - 2 * kIwsI;
                float_type bIJ = -up_value + f;
                f_val2reduce[tid] = (-bIJ * bIJ / aIJ);
            }
            else
                f_val2reduce[tid] = INFINITY;
            int j2 = get_block_min(f_val2reduce, f_idx2reduce, item_ct1);

            // update alpha
            if (tid == i)
                *alpha_i_diff = y > 0 ? Cp - a : a;
            if (tid == j2)
                *alpha_j_diff = sycl::min(
                    (double)(y > 0 ? a : Cn - a),
                    (double)((-up_value + f) / (kd[i] + kd[j2] - 2 * kIwsI)));
            /*
            DPCT1065:97: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            float_type l =
                sycl::min((double)(*alpha_i_diff), (double)(*alpha_j_diff));

            if (tid == i)
                a += l * y;
            if (tid == j2)
                a -= l * y;

            // update f
            kernel_type kJ2wsI = k_mat_rows[row_len * j2 + wsi]; // K[J2, wsi]
            f -= l * (kJ2wsI - kIwsI);
            numOfIter++;
        }
    }

    void
    nu_smo_solve_kernel(const int *label, float_type *f_values, float_type *alpha, float_type *alpha_diff,
                        const int *working_set,
                        int ws_size, float C, const kernel_type *k_mat_rows, const kernel_type *k_mat_diag, int row_len,
                        float_type eps,
                        float_type *diff, int max_iter,
                        sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
    {
        //"row_len" equals to the number of instances in the original training dataset.
        // allocate shared memory
        auto shared_mem = (int *)dpct_local;
        int *f_idx2reduce = shared_mem;                                                                             // temporary memory for reduction
        float_type *f_val2reduce = (float_type *)&shared_mem[ws_size];                                              // f values used for reduction.
        float_type *alpha_i_diff = (float_type *)&shared_mem[ws_size + ws_size * sizeof(float_type) / sizeof(int)]; // delta alpha_i
        float_type *alpha_j_diff = &alpha_i_diff[1];
        kernel_type *kd = (kernel_type *)&alpha_j_diff[1]; // diagonal elements for kernel matrix

        // index, f value and alpha for each instance
        int tid = item_ct1.get_local_id(2);
        int wsi = working_set[tid];
        kd[tid] = k_mat_diag[wsi];
        float_type y = label[wsi];
        float_type f = f_values[wsi];
        float_type a = alpha[wsi];
        float_type aold = a;
        /*
        DPCT1065:98: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        float_type local_eps;
        int numOfIter = 0;
        while (1)
        {
            // select I_up (y=+1)
            if (y > 0 && a < C)
                f_val2reduce[tid] = f;
            else
                f_val2reduce[tid] = INFINITY;
            /*
            DPCT1065:99: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            int ip = get_block_min(f_val2reduce, f_idx2reduce, item_ct1);
            float_type up_value_p = f_val2reduce[ip];
            kernel_type kIpwsI = k_mat_rows[row_len * ip + wsi]; // K[i, wsi]
            /*
            DPCT1065:100: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // select I_up (y=-1)
            if (y < 0 && a > 0)
                f_val2reduce[tid] = f;
            else
                f_val2reduce[tid] = INFINITY;
            int in = get_block_min(f_val2reduce, f_idx2reduce, item_ct1);
            float_type up_value_n = f_val2reduce[in];
            kernel_type kInwsI = k_mat_rows[row_len * in + wsi]; // K[i, wsi]
            /*
            DPCT1065:101: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // select I_low (y=+1)
            if (y > 0 && a > 0)
                f_val2reduce[tid] = -f;
            else
                f_val2reduce[tid] = INFINITY;
            int j1p = get_block_min(f_val2reduce, f_idx2reduce, item_ct1);
            float_type low_value_p = -f_val2reduce[j1p];
            /*
            DPCT1065:102: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // select I_low (y=-1)
            if (y < 0 && a < C)
                f_val2reduce[tid] = -f;
            else
                f_val2reduce[tid] = INFINITY;
            int j1n = get_block_min(f_val2reduce, f_idx2reduce, item_ct1);
            float_type low_value_n = -f_val2reduce[j1n];

            float_type local_diff =
                sycl::max((double)(low_value_p - up_value_p),
                          (double)(low_value_n - up_value_n));

            if (numOfIter == 0)
            {
                local_eps = sycl::max((double)eps, (double)(0.1 * local_diff));
                if (tid == 0)
                {
                    diff[0] = local_diff;
                }
            }

            if (numOfIter > max_iter || local_diff < local_eps)
            {
                alpha[wsi] = a;
                alpha_diff[tid] = -(a - aold) * y;
                diff[1] = numOfIter;
                break;
            }
            /*
            DPCT1065:103: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // select j2p using second order heuristic
            if (-up_value_p > -f && y > 0 && a > 0)
            {
                float_type aIJ = kd[ip] + kd[tid] - 2 * kIpwsI;
                float_type bIJ = -up_value_p + f;
                f_val2reduce[tid] = -bIJ * bIJ / aIJ;
            }
            else
                f_val2reduce[tid] = INFINITY;
            int j2p = get_block_min(f_val2reduce, f_idx2reduce, item_ct1);
            float_type f_val_j2p = f_val2reduce[j2p];
            /*
            DPCT1065:104: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // select j2n using second order heuristic
            if (-up_value_n > -f && y < 0 && a < C)
            {
                float_type aIJ = kd[in] + kd[tid] - 2 * kInwsI;
                float_type bIJ = -up_value_n + f;
                f_val2reduce[tid] = -bIJ * bIJ / aIJ;
            }
            else
                f_val2reduce[tid] = INFINITY;
            int j2n = get_block_min(f_val2reduce, f_idx2reduce, item_ct1);

            int i, j2;
            float_type up_value;
            kernel_type kIwsI;
            if (f_val_j2p < f_val2reduce[j2n])
            {
                i = ip;
                j2 = j2p;
                up_value = up_value_p;
                kIwsI = kIpwsI;
            }
            else
            {
                i = in;
                j2 = j2n;
                kIwsI = kInwsI;
                up_value = up_value_n;
            }
            // update alpha
            if (tid == i)
                *alpha_i_diff = y > 0 ? C - a : a;
            if (tid == j2)
                *alpha_j_diff = sycl::min(
                    (double)(y > 0 ? a : C - a),
                    (double)((-up_value + f) / (kd[i] + kd[j2] - 2 * kIwsI)));
            /*
            DPCT1065:105: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            float_type l =
                sycl::min((double)(*alpha_i_diff), (double)(*alpha_j_diff));

            if (tid == i)
                a += l * y;
            if (tid == j2)
                a -= l * y;

            // update f
            kernel_type kJ2wsI = k_mat_rows[row_len * j2 + wsi]; // K[J2, wsi]
            f -= l * (kJ2wsI - kIwsI);
            numOfIter++;
        }
    }

    void
    c_smo_solve(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                SyncArray<float_type> &alpha_diff,
                const SyncArray<int> &working_set, float_type Cp, float_type Cn, const SyncArray<kernel_type> &k_mat_rows,
                const SyncArray<kernel_type> &k_mat_diag, int row_len, float_type eps, SyncArray<float_type> &diff,
                int max_iter)
    {
        size_t ws_size = working_set.size();
        size_t smem_size = 0;
        smem_size += ws_size * sizeof(int);         // f_idx2reduce
        smem_size += ws_size * sizeof(float_type);  // f_val2reduce
        smem_size += ws_size * sizeof(kernel_type); // kd
        smem_size += 2 * sizeof(float_type);        // alpha diff
        /*
        DPCT1049:106: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        auto &q = thunder::get_sycl_queue();
        q.submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(smem_size), cgh);

                auto y_device_data_ct0 = y.device_data();
                auto f_val_device_data_ct1 = f_val.device_data();
                auto alpha_device_data_ct2 = alpha.device_data();
                auto alpha_diff_device_data_ct3 = alpha_diff.device_data();
                auto working_set_device_data_ct4 = working_set.device_data();
                auto k_mat_rows_device_data_ct8 = k_mat_rows.device_data();
                auto k_mat_diag_device_data_ct9 = k_mat_diag.device_data();
                auto diff_device_data_ct12 = diff.device_data();

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, ws_size),
                                      sycl::range<3>(1, 1, ws_size)),
                    [=](sycl::nd_item<3> item_ct1) {
                            c_smo_solve_kernel(
                                y_device_data_ct0, f_val_device_data_ct1,
                                alpha_device_data_ct2,
                                alpha_diff_device_data_ct3,
                                working_set_device_data_ct4, ws_size, Cp, Cn,
                                k_mat_rows_device_data_ct8,
                                k_mat_diag_device_data_ct9, row_len, eps,
                                diff_device_data_ct12, max_iter, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                    }); });
        q.wait();
    }

    void nu_smo_solve(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                      SyncArray<float_type> &alpha_diff,
                      const SyncArray<int> &working_set, float_type C, const SyncArray<kernel_type> &k_mat_rows,
                      const SyncArray<kernel_type> &k_mat_diag, int row_len, float_type eps, SyncArray<float_type> &diff,
                      int max_iter)
    {
        size_t ws_size = working_set.size();
        size_t smem_size = 0;
        smem_size += ws_size * sizeof(int);         // f_idx2reduce
        smem_size += ws_size * sizeof(float_type);  // f_val2reduce
        smem_size += ws_size * sizeof(kernel_type); // kd
        smem_size += 2 * sizeof(float_type);        // alpha diff
        /*
        DPCT1049:107: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        auto &q = thunder::get_sycl_queue();
        q.submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(smem_size), cgh);

                auto y_device_data_ct0 = y.device_data();
                auto f_val_device_data_ct1 = f_val.device_data();
                auto alpha_device_data_ct2 = alpha.device_data();
                auto alpha_diff_device_data_ct3 = alpha_diff.device_data();
                auto working_set_device_data_ct4 = working_set.device_data();
                auto k_mat_rows_device_data_ct7 = k_mat_rows.device_data();
                auto k_mat_diag_device_data_ct8 = k_mat_diag.device_data();
                auto diff_device_data_ct11 = diff.device_data();

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, ws_size),
                                      sycl::range<3>(1, 1, ws_size)),
                    [=](sycl::nd_item<3> item_ct1) {
                            nu_smo_solve_kernel(
                                y_device_data_ct0, f_val_device_data_ct1,
                                alpha_device_data_ct2,
                                alpha_diff_device_data_ct3,
                                working_set_device_data_ct4, ws_size, C,
                                k_mat_rows_device_data_ct7,
                                k_mat_diag_device_data_ct8, row_len, eps,
                                diff_device_data_ct11, max_iter, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                    }); });
        q.wait();
    }

    void
    update_f_kernel(float_type *f, int ws_size, const float_type *alpha_diff, const kernel_type *k_mat_rows,
                    int n_instances, sycl::nd_item<3> item_ct1)
    {
        //"n_instances" equals to the number of rows of the whole kernel matrix for both SVC and SVR.
        for (int idx = item_ct1.get_local_id(2); idx < (n_instances); idx += 1024)
        { // one thread to update multiple fvalues.
            double sum_diff = 0;
            for (int i = 0; i < ws_size; ++i)
            {
                double d = alpha_diff[i];
                if (d != 0)
                {
                    sum_diff += d * k_mat_rows[i * n_instances + idx];
                }
            }
            f[idx] -= sum_diff;
        }
    }

    void update_f(SyncArray<float_type> &f,
                  const SyncArray<float_type> &alpha_diff,
                  const SyncArray<kernel_type> &k_mat_rows,
                  int n_instances)
    {
        auto &q = thunder::get_sycl_queue();
        q.submit([&](sycl::handler &cgh) { 
            auto f_device_data_ct0 = f.device_data();
            auto alpha_diff_size_ct1 = alpha_diff.size();
            auto alpha_diff_device_data_ct2 = alpha_diff.device_data();
            auto k_mat_rows_device_data_ct3 = k_mat_rows.device_data();
            auto n_instances_ct4 = n_instances;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)),
                [=](sycl::nd_item<3> item_ct1) {
                    update_f_kernel(f_device_data_ct0, alpha_diff_size_ct1, alpha_diff_device_data_ct2, k_mat_rows_device_data_ct3, n_instances_ct4, item_ct1);
                }); 
            });
        q.wait();
    }

    void sort_f(SyncArray<float_type> &f_val2sort, SyncArray<int> &f_idx2sort)
    {
        dpct::sort(oneapi::dpl::execution::make_device_policy(thunder::get_sycl_queue()),
                   f_val2sort.device_data(), f_val2sort.device_data() + f_val2sort.size(),
                   f_idx2sort.device_data(), oneapi::dpl::less<float_type>());
    }
}
