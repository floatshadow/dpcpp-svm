
//
// Created by jiashuai on 17-11-7.
//
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <thundersvm/util/sycl_common.h>

#include <thundersvm/kernel/smo_kernel.h>
#include <omp.h>

namespace svm_kernel {
#ifdef USE_GPU
    template <typename T> int get_block_min(const T *values, int *index, sycl::nd_item<3> item_ct1)
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
        for (int offset = item_ct1.get_local_range(2) / 2; offset > 0; offset >>= 1)
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

    void c_smo_solve_kernel(const int *label, float_type *f_val, float_type *alpha, float_type *alpha_diff,
                            const int *working_set, int ws_size, float_type Cp, float_type Cn,
                            const kernel_type *k_mat_rows, const kernel_type *k_mat_diag, int row_len, float_type eps,
                            float_type *diff, int max_iter, sycl::nd_item<3> item_ct1, volatile uint8_t *dpct_local)
    {
        //"row_len" equals to the number of instances in the original training dataset.
        // allocate shared memory
        auto shared_mem = (int *)dpct_local;
        int *f_idx2reduce = shared_mem;                                // temporary memory for reduction
        float_type *f_val2reduce = (float_type *)&shared_mem[ws_size]; // f values used for reduction.
        float_type *alpha_i_diff =
            (float_type *)&shared_mem[ws_size + ws_size * sizeof(float_type) / sizeof(int)]; // delta alpha_i
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
                *alpha_j_diff =
                    sycl::min((double)(y > 0 ? a : Cn - a), (double)((-up_value + f) / (kd[i] + kd[j2] - 2 * kIwsI)));
            /*
            DPCT1065:97: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            float_type l = sycl::min((double)(*alpha_i_diff), (double)(*alpha_j_diff));

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

    void c_smo_solve(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                     SyncArray<float_type> &alpha_diff, const SyncArray<int> &working_set, float_type Cp, float_type Cn,
                     const SyncArray<kernel_type> &k_mat_rows, const SyncArray<kernel_type> &k_mat_diag, int row_len,
                     float_type eps, SyncArray<float_type> &diff, int max_iter)
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
        thunder::get_sycl_queue()
            .submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(smem_size), cgh);

                auto y_device_data_ct0 = y.device_data();
                auto f_val_device_data_ct1 = f_val.device_data();
                auto alpha_device_data_ct2 = alpha.device_data();
                auto alpha_diff_device_data_ct3 = alpha_diff.device_data();
                auto working_set_device_data_ct4 = working_set.device_data();
                auto k_mat_rows_device_data_ct8 = k_mat_rows.device_data();
                auto k_mat_diag_device_data_ct9 = k_mat_diag.device_data();
                auto diff_device_data_ct12 = diff.device_data();

                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ws_size), sycl::range<3>(1, 1, ws_size)),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     c_smo_solve_kernel(y_device_data_ct0, f_val_device_data_ct1, alpha_device_data_ct2,
                                                        alpha_diff_device_data_ct3, working_set_device_data_ct4,
                                                        ws_size, Cp, Cn, k_mat_rows_device_data_ct8,
                                                        k_mat_diag_device_data_ct9, row_len, eps, diff_device_data_ct12,
                                                        max_iter, item_ct1, dpct_local_acc_ct1.get_pointer());
                                 });
            })
            .wait();
    }
#else
    /* Merrill, 2015*/
    /// @brief: this kernel apply parallel reduction to twice to
    /// find 2 extreme training instance.
    void c_smo_solve_kernel(const int *label, float_type *f_val, float_type *alpha, float_type *alpha_diff,
                            const int *working_set, int ws_size, float_type Cp, float_type Cn,
                            const kernel_type *k_mat_rows, const kernel_type *k_mat_diag, int row_len, float_type eps,
                            float_type *diff, int max_iter)
    {
        // allocate shared memory
        float_type alpha_i_diff; // delta alpha_i
        float_type alpha_j_diff;
        vector<kernel_type> kd(ws_size); // diagonal elements for kernel matrix

        // index, f value and alpha for each instance
        vector<float_type> a_old(ws_size);
        vector<float_type> kIwsI(ws_size);
        vector<float_type> f(ws_size);
        vector<float_type> y(ws_size);
        vector<float_type> a(ws_size);
        for (int tid = 0; tid < ws_size; ++tid)
        {
            int wsi = working_set[tid];
            f[tid] = f_val[wsi];
            a_old[tid] = a[tid] = alpha[wsi];
            y[tid] = label[wsi];
            kd[tid] = k_mat_diag[wsi];
        }
        float_type local_eps;
        int numOfIter = 0;
        while (1)
        {
            // select fUp and fLow
            int i = 0;
            float_type up_value = INFINITY;
            for (int tid = 0; tid < ws_size; ++tid)
            {
                if (is_I_up(a[tid], y[tid], Cp, Cn))
                    if (f[tid] < up_value)
                    {
                        up_value = f[tid];
                        i = tid;
                    }
            }
            for (int tid = 0; tid < ws_size; ++tid)
            {
                /* row_len = n_instances */
                kIwsI[tid] = k_mat_rows[row_len * i + working_set[tid]]; // K[i, wsi]
            }
            float_type low_value = -INFINITY;
            for (int tid = 0; tid < ws_size; ++tid)
            {
                if (is_I_low(a[tid], y[tid], Cp, Cn))
                    if (f[tid] > low_value)
                    {
                        low_value = f[tid];
                    }
            }

            //            printf("up = %lf, low = %lf\n", up_value, low_value);
            float_type local_diff = low_value - up_value;
            if (numOfIter == 0)
            {
                local_eps = max(eps, 0.1f * local_diff);
                diff[0] = local_diff;
            }

            if (numOfIter > max_iter || local_diff < local_eps)
            {
                for (int tid = 0; tid < ws_size; ++tid)
                {
                    int wsi = working_set[tid];
                    alpha_diff[tid] = -(a[tid] - a_old[tid]) * y[tid];
                    alpha[wsi] = a[tid];
                }
                diff[1] = numOfIter;
                break;
            }
            int j2 = 0;
            float_type min_t = INFINITY;
            // select j2 using second order heuristic
            for (int tid = 0; tid < ws_size; ++tid)
            {
                if (-up_value > -f[tid] && (is_I_low(a[tid], y[tid], Cp, Cn)))
                {
                    float_type aIJ = kd[i] + kd[tid] - 2 * kIwsI[tid];
                    float_type bIJ = -up_value + f[tid];
                    float_type ft = -bIJ * bIJ / aIJ;
                    if (ft < min_t)
                    {
                        min_t = ft;
                        j2 = tid;
                    }
                }
            }

            // update alpha
            //            if (tid == i)
            alpha_i_diff = y[i] > 0 ? Cp - a[i] : a[i];
            //            if (tid == j2)
            alpha_j_diff = min(y[j2] > 0 ? a[j2] : Cn - a[j2], (-up_value + f[j2]) / (kd[i] + kd[j2] - 2 * kIwsI[j2]));
            float_type l = min(alpha_i_diff, alpha_j_diff);

            //            if (tid == i)
            a[i] += l * y[i];
            //            if (tid == j2)
            a[j2] -= l * y[j2];

            // update f
            for (int tid = 0; tid < ws_size; ++tid)
            {
                int wsi = working_set[tid];
                float_type kJ2wsI = k_mat_rows[row_len * j2 + wsi]; // K[J2, wsi]
                f[tid] -= l * (kJ2wsI - kIwsI[tid]);
            }
            numOfIter++;
        }
    }

    void c_smo_solve(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                    SyncArray<float_type> &alpha_diff, const SyncArray<int> &working_set, float_type Cp, float_type Cn,
                    const SyncArray<kernel_type> &k_mat_rows, const SyncArray<kernel_type> &k_mat_diag, int row_len,
                    float_type eps, SyncArray<float_type> &diff, int max_iter)
    {
        c_smo_solve_kernel(y.host_data(), f_val.host_data(), alpha.host_data(), alpha_diff.host_data(),
                        working_set.host_data(), working_set.size(), Cp, Cn, k_mat_rows.host_data(),
                        k_mat_diag.host_data(), row_len, eps, diff.host_data(), max_iter);
    }
#endif




    void nu_smo_solve_kernel(const int *y, float_type *f_val, float_type *alpha, float_type *alpha_diff,
                             const int *working_set, int ws_size, float_type C, const kernel_type *k_mat_rows,
                             const kernel_type *k_mat_diag, int row_len, float_type eps, float_type *diff,
                             int max_iter) {
        //allocate shared memory
        float_type alpha_i_diff; //delta alpha_i
        float_type alpha_j_diff;
        kernel_type *kd = new kernel_type[ws_size]; // diagonal elements for kernel matrix

        //index, f value and alpha for each instance
        float_type *a_old = new float_type[ws_size];
        kernel_type *kIpwsI = new kernel_type[ws_size];
        kernel_type *kInwsI = new kernel_type[ws_size];
        float_type *f = new float_type[ws_size];
        for (int tid = 0; tid < ws_size; ++tid) {
            int wsi = working_set[tid];
            f[tid] = f_val[wsi];
            a_old[tid] = alpha[wsi];
            kd[tid] = k_mat_diag[wsi];
        }
        float_type local_eps;
        int numOfIter = 0;
        while (1) {
            //select I_up (y=+1)
            int ip = 0;
            float_type up_value_p = INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] > 0 && alpha[wsi] < C)
                    if (f[tid] < up_value_p) {
                        ip = tid;
                        up_value_p = f[tid];
                    }
            }

            for (int tid = 0; tid < ws_size; ++tid) {
                kIpwsI[tid] = k_mat_rows[row_len * ip + working_set[tid]];//K[i, wsi]
            }

            //select I_up (y=-1)
            int in = 0;
            float_type up_value_n = INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] < 0 && alpha[wsi] > 0)
                    if (f[tid] < up_value_n) {
                        in = tid;
                        up_value_n = f[tid];
                    }
            }
            for (int tid = 0; tid < ws_size; ++tid) {
                kInwsI[tid] = k_mat_rows[row_len * in + working_set[tid]];//K[i, wsi]
            }

            //select I_low (y=+1)
            float_type low_value_p = -INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] > 0 && alpha[wsi] > 0)
                    if (f[tid] > low_value_p) {
                        low_value_p = f[tid];
                    }
            }


            //select I_low (y=-1)
            float_type low_value_n = -INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] < 0 && alpha[wsi] < C)
                    if (f[tid] > low_value_n) {
                        low_value_n = f[tid];
                    }
            }

            float_type local_diff = max(low_value_p - up_value_p, low_value_n - up_value_n);

            if (numOfIter == 0) {
                local_eps = max(eps, 0.1 * local_diff);
                diff[0] = local_diff;
            }

            if (numOfIter > max_iter || local_diff < local_eps) {
                for (int tid = 0; tid < ws_size; ++tid) {
                    int wsi = working_set[tid];
                    alpha_diff[tid] = -(alpha[wsi] - a_old[tid]) * y[wsi];
                }
                break;
            }

            //select j2p using second order heuristic
            int j2p = 0;
            float_type f_val_j2p = INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (-up_value_p > -f[tid] && y[wsi] > 0 && alpha[wsi] > 0) {
                    float_type aIJ = kd[ip] + kd[tid] - 2 * kIpwsI[tid];
                    float_type bIJ = -up_value_p + f[tid];
                    float_type f_t1 = -bIJ * bIJ / aIJ;
                    if (f_t1 < f_val_j2p) {
                        j2p = tid;
                        f_val_j2p = f_t1;
                    }
                }
            }

            //select j2n using second order heuristic
            int j2n = 0;
            float_type f_val_j2n = INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (-up_value_n > -f[tid] && y[wsi] < 0 && alpha[wsi] < C) {
                    float_type aIJ = kd[ip] + kd[tid] - 2 * kIpwsI[tid];
                    float_type bIJ = -up_value_n + f[tid];
                    float_type f_t2 = -bIJ * bIJ / aIJ;
                    if (f_t2 < f_val_j2n) {
                        j2n = tid;
                        f_val_j2n = f_t2;
                    }
                }
            }

            int i, j2;
            float_type up_value;
            kernel_type *kIwsI;
            if (f_val_j2p < f_val_j2n) {
                i = ip;
                j2 = j2p;
                up_value = up_value_p;
                kIwsI = kIpwsI;
            } else {
                i = in;
                j2 = j2n;
                kIwsI = kInwsI;
                up_value = up_value_n;
            }
            //update alpha
//            if (tid == i)
            alpha_i_diff = y[working_set[i]] > 0 ? C - alpha[working_set[i]] : alpha[working_set[i]];
//            if (tid == j2)
            alpha_j_diff = min(y[working_set[j2]] > 0 ? alpha[working_set[j2]] : C - alpha[working_set[j2]],
                               (-up_value + f[j2]) / (kd[i] + kd[j2] - 2 * kIwsI[j2]));
            float_type l = min(alpha_i_diff, alpha_j_diff);

            alpha[working_set[i]] += l * y[working_set[i]];
            alpha[working_set[j2]] -= l * y[working_set[j2]];

            //update f
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                kernel_type kJ2wsI = k_mat_rows[row_len * j2 + wsi];//K[J2, wsi]
                f[tid] -= l * (kJ2wsI - kIwsI[tid]);
            }
            numOfIter++;
        }
        delete[] kd;
        delete[] a_old;
        delete[] f;
        delete[] kIpwsI;
        delete[] kInwsI;
    }

    void nu_smo_solve(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                      SyncArray<float_type> &alpha_diff,
                      const SyncArray<int> &working_set, float_type C, const SyncArray<kernel_type> &k_mat_rows,
                      const SyncArray<kernel_type> &k_mat_diag, int row_len, float_type eps,
                      SyncArray<float_type> &diff,
                      int max_iter) {
        nu_smo_solve_kernel(y.host_data(), f_val.host_data(), alpha.host_data(), alpha_diff.host_data(),
                            working_set.host_data(), working_set.size(), C, k_mat_rows.host_data(),
                            k_mat_diag.host_data(), row_len, eps, diff.host_data(), max_iter);
    }

#ifdef USE_GPU
    void update_f_kernel(float_type *f, int ws_size, const float_type *alpha_diff, const kernel_type *k_mat_rows,
                         int n_instances, sycl::id<1> idx)
    {
        //"n_instances" equals to the number of rows of the whole kernel matrix for both SVC and SVR.
        // int idx = item.get_id();
        // for (int idx = item.get_global_id(); idx < (n_instances); idx += 1)
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

    void update_f(SyncArray<float_type> &f, const SyncArray<float_type> &alpha_diff,
                  const SyncArray<kernel_type> &k_mat_rows, int n_instances)
    {
        thunder::get_sycl_queue()
            .submit([&](sycl::handler &h) {
                auto f_device_data_ct0 = f.device_data();
                auto alpha_diff_size_ct1 = alpha_diff.size();
                auto alpha_diff_device_data_ct2 = alpha_diff.device_data();
                auto k_mat_rows_device_data_ct3 = k_mat_rows.device_data();
                auto n_instances_ct4 = n_instances;
                // h.parallel_for(sycl::nd_range<1>(sycl::range<1>(n_instances), sycl::range<1>(1)),
                //                [=](sycl::nd_item<1> item) {
                //                    update_f_kernel(f_device_data_ct0, alpha_diff_size_ct1, alpha_diff_device_data_ct2,
                //                                    k_mat_rows_device_data_ct3, n_instances, item);
                //                });
                h.parallel_for(sycl::range<1>(n_instances),
                               [=](sycl::id<1> idx) {
                                   update_f_kernel(f_device_data_ct0, alpha_diff_size_ct1, alpha_diff_device_data_ct2,
                                                   k_mat_rows_device_data_ct3, n_instances, idx);
                               });
            })
            .wait();
    }
#else
    void update_f(SyncArray<float_type> &f, const SyncArray<float_type> &alpha_diff,
             const SyncArray<kernel_type> &k_mat_rows,
             int n_instances) {
        //"n_instances" equals to the number of rows of the whole kernel matrix for both SVC and SVR.
        /// @attention Use device_data with modified syncarray.
        float_type *f_data = f.host_data();
        const float_type *alpha_diff_data = alpha_diff.host_data();
        const kernel_type *k_mat_rows_data = k_mat_rows.host_data();
#pragma omp parallel for schedule(guided)
        for (int idx = 0; idx < n_instances; ++idx) {
            double sum_diff = 0;
            for (int i = 0; i < alpha_diff.size(); ++i) {
                float_type d = alpha_diff_data[i];
                if (d != 0) {
                    /// @attention: cache un-friendly.
                    sum_diff += d * k_mat_rows_data[i * n_instances + idx];
                }
            }
            f_data[idx] -= sum_diff;
        }
    }

#endif


    /// @attention Use device_data with modified syncarray.
    void sort_f(SyncArray<float_type> &f_val2sort, SyncArray<int> &f_idx2sort)
    {
        auto &q = thunder::get_sycl_queue();
#ifdef USE_GPU
        dpct::sort(oneapi::dpl::execution::make_device_policy(q), f_val2sort.device_data(),
                   f_val2sort.device_data() + f_val2sort.size(), f_idx2sort.device_data(),
                   oneapi::dpl::less<float_type>());
#else
        dpct::sort(oneapi::dpl::execution::make_device_policy(q),
                   f_val2sort.host_data(), f_val2sort.host_data() + f_val2sort.size(),
                   f_idx2sort.host_data(), oneapi::dpl::less<float_type>());
#endif
    }
}

