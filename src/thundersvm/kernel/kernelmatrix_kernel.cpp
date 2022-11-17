//
// Created by jiashuai on 17-11-7.
//

#include "thundersvm/thundersvm.h"
#include <thundersvm/kernel/kernelmatrix_kernel.h>
#include <thundersvm/util/sycl_common.h>
#include <oneapi/mkl.hpp>
#include <oneapi/dpl/cmath>
#ifndef USE_GPU
#include <Eigen/Dense>
#include <Eigen/Sparse>
#endif

using namespace oneapi::mkl;
using namespace sycl;

template <size_t round_base> 
constexpr size_t round_up(size_t size) {
    static_assert((round_base & (round_base - 1)) == 0);
    return (size + round_base - 1) & (~(round_base - 1));
}

constexpr size_t ceildiv(size_t n, size_t m) {
    return (n + m - 1) / (m);
}

namespace svm_kernel {
    void kernel_get_working_set_ins(const kernel_type *val, const int *col_ind, const int *row_ptr, 
                               const int *data_row_idx, kernel_type *data_rows,
                               int m, int n) 
    {
        auto &q = thunder::get_sycl_queue();
        constexpr size_t work_group_size = 256;
        size_t global_group_size = round_up<work_group_size>(m);
        q.submit([&](handler &h){
            constexpr size_t sub_group_size = 8U;
            h.parallel_for(sycl::nd_range<1>(global_group_size, work_group_size),
                [=](nd_item<1> item)[[intel::reqd_sub_group_size(sub_group_size)]]
            {
                int i = item.get_global_linear_id();
                if (i >= m)
                    return;
                
                int row = data_row_idx[i];
                int col_data_start = row_ptr[row];
                int col_data_end   = row_ptr[row + 1];
            #pragma unroll
                for (int j = col_data_start; j < col_data_end; ++j) {
                    data_rows[i * n + col_ind[j]] = val[j];
                }     
            });
        }).wait();
    }

    /// @brief fill the sparse data into dense matrix.
    void
    get_working_set_ins(const SyncArray<kernel_type> &val, const SyncArray<int> &col_ind, const SyncArray<int> &row_ptr,
                            const SyncArray<int> &data_row_idx, SyncArray<kernel_type> &data_rows, int m, int n) {
        const int *data_row_idx_data = data_row_idx.device_data();
        kernel_type *data_rows_data = data_rows.device_data();
        const int *row_ptr_data = row_ptr.device_data();
        const int *col_ind_data = col_ind.device_data();
        const kernel_type *val_data = val.device_data();
        kernel_get_working_set_ins(val_data, col_ind_data, row_ptr_data, 
                                   data_row_idx_data, data_rows_data,
                                   m, n);
        
    }

    void
    get_working_set_ins(const SyncArray<kernel_type> &val, const SyncArray<int> &col_ind, const SyncArray<int> &row_ptr,
                        const SyncArray<int> &data_row_idx, SyncArray <kernel_type>& ws_val,
                        SyncArray<int> &ws_col_ind, SyncArray<int> &ws_row_ptr, int m){
        const int *data_row_idx_data = data_row_idx.host_data();
        const int *row_ptr_data = row_ptr.host_data();
        const int *col_ind_data = col_ind.host_data();
        const kernel_type *val_data = val.host_data();
//        kernel_type *ws_val_data = ws_val.host_data();
//        int *ws_row_ptr_data = ws_row_ptr.host_data();
//        int *ws_col_ind_data = ws_col_ind.host_data();
        //three arrays for csr representation
        vector<kernel_type> csr_val;
        vector<int> csr_col_ind;//index of each value of all the instances
        vector<int> csr_row_ptr(1, 0);//the start positions of the instances
        //ws_row_ptr_data[0] = 0;
        for(int i = 0; i < m; i++){
            int row = data_row_idx_data[i];
            for(int j = row_ptr_data[row]; j < row_ptr_data[row+1]; ++j){
                csr_col_ind.push_back(col_ind_data[j]);
                csr_val.push_back(val_data[j]);
            }
            csr_row_ptr.push_back(csr_row_ptr.back() + row_ptr_data[row+1] - row_ptr_data[row]);
        }
        //three arrays (on GPU/CPU) for csr representation
        ws_val.resize(csr_val.size());
        ws_col_ind.resize(csr_col_ind.size());
        ws_row_ptr.resize(csr_row_ptr.size());
        //copy data to the three arrays
        ws_val.copy_from(csr_val.data(), ws_val.size());
        ws_col_ind.copy_from(csr_col_ind.data(), ws_col_ind.size());
        ws_row_ptr.copy_from(csr_row_ptr.data(), ws_row_ptr.size());
    }


    /// @attention: Not be called in CPU only train.
    void
    RBF_kernel(const SyncArray<kernel_type> &self_dot0, const SyncArray<kernel_type> &self_dot1,
               SyncArray<kernel_type> &dot_product, int m,
               int n, kernel_type gamma) {
        kernel_type *dot_product_data = dot_product.host_data();
        const kernel_type *self_dot0_data = self_dot0.host_data();
        const kernel_type *self_dot1_data = self_dot1.host_data();
#pragma omp parallel for schedule(guided)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; ++j) {
                dot_product_data[i * n + j] = expf(
                        -(self_dot0_data[i] + self_dot1_data[j] - dot_product_data[i * n + j] * 2) * gamma);
            }
        }
    }

#ifdef USE_GPU
    /// @attention: \p Is called in CPU only train.
    void RBF_kernel(const SyncArray<int> &self_dot0_idx, const SyncArray<kernel_type> &self_dot1,
                    SyncArray<kernel_type> &dot_product, int m, int n, kernel_type gamma)
    {
        kernel_type *dot_product_data = dot_product.device_data();
        const int *self_dot0_idx_data = self_dot0_idx.device_data();
        const kernel_type *self_dot1_data = self_dot1.device_data();
        auto &q = thunder::get_sycl_queue();
        constexpr size_t work_group_size = 512; // based on 64kb local memory, and each dim work_group limit.
        size_t global_group_size = round_up<work_group_size>(n);
        q.submit([&](handler &h) {
             constexpr size_t sub_group_size = 8U;
             sycl::accessor<kernel_type, 1, sycl::access::mode::read_write, sycl::access::target::local> self_dot_j(
                 work_group_size, h);
             h.parallel_for(sycl::nd_range<1>(global_group_size, work_group_size),
                            [=](nd_item<1> item) [[intel::reqd_sub_group_size(sub_group_size)]] {
                                int j = item.get_global_linear_id();
                                if (j >= n)
                                    return;

                                auto sg = item.get_sub_group();
                                int local_j = item.get_local_id(0);
                                self_dot_j[local_j] = self_dot1_data[j]; // writes local memory.

                                using global_ptr =
                                    sycl::multi_ptr<const kernel_type, sycl::access::address_space::global_space>;
#pragma unroll
                                for (int i = 0; i < m; ++i)
                                {
                                    // kernel_type self_dot_i =
                                    // sg.load(global_ptr(&(self_dot1_data[self_dot0_idx_data[i]])));
                                    kernel_type self_dot_i = self_dot1_data[self_dot0_idx_data[i]];
                                    dot_product_data[i * n + j] = expf(
                                        -(self_dot_i + self_dot1_data[j] - dot_product_data[i * n + j] * 2) * gamma);
                                }
                            });
         }).wait();
    }
#else
    void
    RBF_kernel(const SyncArray<int> &self_dot0_idx, const SyncArray<kernel_type> &self_dot1,
               SyncArray<kernel_type> &dot_product, int m,
               int n, kernel_type gamma) {
        kernel_type *dot_product_data = dot_product.host_data();
        const int *self_dot0_idx_data = self_dot0_idx.host_data();
        const kernel_type *self_dot1_data = self_dot1.host_data();
#pragma omp parallel for schedule(guided)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; ++j) {
                dot_product_data[i * n + j] = expf(
                        -(self_dot1_data[self_dot0_idx_data[i]] + self_dot1_data[j] - dot_product_data[i * n + j] * 2) *
                        gamma);
            }
        }
    }
#endif


    void poly_kernel(SyncArray<kernel_type> &dot_product, kernel_type gamma, kernel_type coef0, int degree, int mn) {
        kernel_type *dot_product_data = dot_product.host_data();
#pragma omp parallel for schedule(guided)
        for (int idx = 0; idx < mn; idx++) {
            dot_product_data[idx] = powf(gamma * dot_product_data[idx] + coef0, degree);
        }
    }

    void sigmoid_kernel(SyncArray<kernel_type> &dot_product, kernel_type gamma, kernel_type coef0, int mn) {
        kernel_type *dot_product_data = dot_product.host_data();
#pragma omp parallel for schedule(guided)
        for (int idx = 0; idx < mn; idx++) {
            dot_product_data[idx] = tanhf(gamma * dot_product_data[idx] + coef0);
        }
    }

    void sum_kernel_values(const SyncArray<float_type> &coef, int total_sv, const SyncArray<int> &sv_start,
                           const SyncArray<int> &sv_count, const SyncArray<float_type> &rho,
                           const SyncArray<kernel_type> &k_mat,
                           SyncArray<float_type> &dec_values, int n_classes, int n_instances) {
        const int *sv_start_data = sv_start.host_data();
        const int *sv_count_data = sv_count.host_data();
        const float_type *coef_data = coef.host_data();
        const kernel_type *k_mat_data = k_mat.host_data();
        float_type *dec_values_data = dec_values.host_data();
        const float_type* rho_data = rho.host_data();
#pragma omp parallel for schedule(guided)
        for (int idx = 0; idx < n_instances; idx++) {
            int k = 0;
            int n_binary_models = n_classes * (n_classes - 1) / 2;
            for (int i = 0; i < n_classes; ++i) {
                for (int j = i + 1; j < n_classes; ++j) {
                    int si = sv_start_data[i];
                    int sj = sv_start_data[j];
                    int ci = sv_count_data[i];
                    int cj = sv_count_data[j];
                    const float_type *coef1 = &coef_data[(j - 1) * total_sv];
                    const float_type *coef2 = &coef_data[i * total_sv];
                    const kernel_type *k_values = &k_mat_data[idx * total_sv];
                    double sum = 0;
#pragma omp parallel for reduction(+:sum)
                    for (int l = 0; l < ci; ++l) {
                        sum += coef1[si + l] * k_values[si + l];
                    }
#pragma omp parallel for reduction(+:sum)
                    for (int l = 0; l < cj; ++l) {
                        sum += coef2[sj + l] * k_values[sj + l];
                    }
                    dec_values_data[idx * n_binary_models + k] = sum - rho_data[k];
                    k++;
                }
            }
        }
    }

    /// @attention: Use device_data with modified syncarray.
    void dns_csr_mul(int m, int n, int k, const SyncArray<kernel_type> &dense_mat, const SyncArray<kernel_type> &csr_val,
                     const SyncArray<int> &csr_row_ptr, const SyncArray<int> &csr_col_ind, int nnz,
                     SyncArray<kernel_type> &result) {
        // Eigen::Map<const Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> denseMat(dense_mat.host_data(), k, n);
        // Eigen::Map<const Eigen::SparseMatrix<kernel_type, Eigen::RowMajor>> sparseMat(m, k, nnz, csr_row_ptr.host_data(),
        //                                                                         csr_col_ind.host_data(),
        //                                                                         csr_val.host_data());
        // Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> retMat = sparseMat * denseMat;
        // Eigen::Map<Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(result.host_data(),
        //                                                                                    retMat.rows(),
        //                                                                                    retMat.cols()) = retMat;
        sparse::matrix_handle_t handle;
        kernel_type one(1.0);
        kernel_type zero(0.0);
        auto &q = thunder::get_sycl_queue();

#ifdef USE_GPU
        /// \brief Do Sparse Mat (m x k) @ trans(Dense Mat (n x k)) = Dense Mat (m x n), Column Major.
        // sparse::set_csr_data(handle, m, k, index_base::zero, const_cast<int *>(csr_row_ptr.device_data()),
        //                      const_cast<int *>(csr_col_ind.device_data()),
        //                      const_cast<kernel_type *>(csr_val.device_data()));
        // sparse::set_matrix_property(handle, sparse::property::sorted);
        /// @attention: weired trans.
        // AS MKL do not support dense matrix B `trans` we add a manual trans.
        // kernel_type *dense_mat_trans = (kernel_type *)malloc(sizeof(kernel_type) * dense_mat.size());
        // const kernel_type *dense_mat_data = dense_mat.host_data();
        // for (int j = 0; j < n; ++j)
        //  for (int i = 0; i < k; ++i)
        //      dense_mat_trans[j * k + i] = dense_mat_data[i * n + i];
        // std::cout << m << " " << k << " " << n << std::endl;
        auto &sub_queues = thunder::MutiTile::get_sub_queues();
        size_t n_sub_devices = sub_queues.size();
        size_t n_slice = ceildiv(n, n_sub_devices);
        kernel_type *dense_mat_data = const_cast<kernel_type *>(dense_mat.device_data());
        kernel_type *result_data = const_cast<kernel_type *>(result.device_data());

        std::vector<sparse::matrix_handle_t> handles(n_sub_devices);
        for (size_t i = 0; i < n_sub_devices; ++i) {
            sparse::init_matrix_handle(&handles[i]);
            sparse::set_csr_data(handles[i], m, k, index_base::zero, const_cast<int *>(csr_row_ptr.device_data()),
                                 const_cast<int *>(csr_col_ind.device_data()),
                                 const_cast<kernel_type *>(csr_val.device_data()));
            sparse::set_matrix_property(handles[i], sparse::property::sorted);
        }

        for (size_t i = 0; i < n_sub_devices; ++i) {
            size_t n_cols = std::min(n_slice, n - i * n_slice);
            auto &Q = sub_queues[i];
            auto gemm_event = sparse::gemm(Q, layout::col_major, transpose::nontrans, transpose::nontrans, one, handles[i],
                                           dense_mat_data + i * n_slice * k, n_cols, k, // num_col, ldB
                                           zero, result_data + i * n_slice * m, m, {});
        }
        for (size_t i = 0; i < n_sub_devices; ++i) {
            auto &Q = sub_queues[i];
            Q.wait_and_throw();
        }

        for (size_t i = 0; i < n_sub_devices; ++i) {
            sparse::release_matrix_handle(&handles[i], {});
        }

#else
        sparse::init_matrix_handle(&handle);
        sparse::set_csr_data(handle, m, k, index_base::zero, const_cast<int *>(csr_row_ptr.host_data()),
                             const_cast<int *>(csr_col_ind.host_data()),
                             const_cast<kernel_type *>(csr_val.host_data()));
        sparse::set_matrix_property(handle, sparse::property::sorted);
        auto gemm_event = sparse::gemm(q, layout::col_major, transpose::nontrans, transpose::nontrans, one, handle,
                                       const_cast<kernel_type *>(dense_mat.host_data()), n, k, // num_col, ldB
                                       zero, const_cast<kernel_type *>(result.host_data()), m, {});
        sparse::release_matrix_handle(&handle, {gemm_event});
#endif
        // free(dense_mat_trans);
    }


    void csr_csr_mul(int m, int n, int k, const SyncArray<kernel_type> &ws_val, const SyncArray<int> &ws_col_ind,
                     const SyncArray<int> &ws_row_ptr, const SyncArray<kernel_type> &csr_val,
                     const SyncArray<int> &csr_row_ptr, const SyncArray<int> &csr_col_ind, int nnz, int nnz2,
                     SyncArray<kernel_type> &result) {
        // Eigen::Map<const Eigen::SparseMatrix<kernel_type, Eigen::RowMajor>> sparseMat1(m, k, nnz, csr_row_ptr.host_data(),
        //                                                                               csr_col_ind.host_data(),
        //                                                                               csr_val.host_data());
        // Eigen::Map<const Eigen::SparseMatrix<kernel_type>> sparseMat2(k, n, nnz2, ws_row_ptr.host_data(),
        //                                                               ws_col_ind.host_data(),
        //                                                               ws_val.host_data());
        // Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> retMat = sparseMat1 * sparseMat2;
        // Eigen::Map<Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(result.host_data(),
        //                                                                                          retMat.rows(),
        //                                                                                          retMat.cols()) = retMat;
        auto &q = thunder::get_sycl_queue();
        sparse::matrix_handle_t A, B, C;
        sparse::matmat_descr_t descr;
        sparse::matrix_view_descr viewA, viewB, viewC;

        sparse::init_matrix_handle(&A);
        sparse::init_matrix_handle(&B);
        sparse::init_matrix_handle(&C);
        // before matmat 
        sparse::set_csr_data(A, m, k, index_base::zero, const_cast<int *>(csr_row_ptr.device_data()),
                             const_cast<int *>(csr_col_ind.device_data()),
                             const_cast<kernel_type *>(csr_val.device_data()));
        sparse::set_csr_data(B, k, n, index_base::zero, const_cast<int *>(ws_row_ptr.device_data()),
                             const_cast<int *>(ws_col_ind.device_data()),
                             const_cast<kernel_type *>(ws_val.device_data()));
        // dummy C sparse matrix handle.
        int c_nrows = m;
        int c_ncols = n;
        index_base c_index = index_base::zero;
        int *c_row_ptr = nullptr;
        int *c_col_ind = nullptr;
        kernel_type *c_val = nullptr;
        c_row_ptr = malloc_device<int>(c_nrows + 1, q);
        sparse::set_csr_data(C, c_nrows, c_ncols, c_index, c_row_ptr, (int *)nullptr, (kernel_type *)nullptr);

        // initialize the matmat descriptor.
        viewA = viewB = viewC = sparse::matrix_view_descr::general;
        sparse::init_matmat_descr(&descr);

        sparse::set_matmat_data(descr, viewA, transpose::nontrans, viewB, transpose::nontrans, viewC);
        // work estimation.
        using sparse::matmat_request;
        auto work_estimation_stage = sparse::matmat(q, A, B, C, matmat_request::work_estimation, descr, 
                                                    nullptr, nullptr, {});
        // compute.
        auto compute_stage         = sparse::matmat(q, A, B, C, matmat_request::compute, descr,
                                                    nullptr, nullptr, {work_estimation_stage});
        // finalize, get nnz.
        std::int64_t *c_nnz = malloc_shared<std::int64_t>(1, q);
        auto finalize_nnz_stage    = sparse::matmat(q, A, B, C, matmat_request::get_nnz, descr,
                                                    c_nnz, nullptr, {compute_stage});
        // finalize, allocate C matrix array
        c_col_ind = malloc_device<int>(*c_nnz, q);
        c_val = malloc_device<kernel_type>(*c_nnz, q);
        sparse::set_csr_data(C, c_nrows, c_ncols, c_index, c_row_ptr, c_col_ind, c_val);

        // finalize, into C matrix
        auto finalize_into_stage = sparse::matmat(q, A, B, C, matmat_request::finalize, descr,
                                                  nullptr, nullptr, {finalize_nnz_stage});
        
        // fill into result syncarray.
        kernel_type *result_data = result.device_data();
        constexpr size_t work_group_size = 256;
        size_t global_group_size = round_up<work_group_size>(m);
        q.submit([&](handler &h){
            constexpr size_t sub_group_size = 8U;
            h.parallel_for(sycl::nd_range<1>(global_group_size, work_group_size),
                [=](nd_item<1> item)[[intel::reqd_sub_group_size(sub_group_size)]]
            {
                int i = item.get_global_linear_id();
                if (i >= m)
                    return;
                
                int col_data_start = c_row_ptr[i];
                int col_data_end   = c_row_ptr[i + 1];
            #pragma unroll
                for (int j = col_data_start; j < col_data_end; ++j) {
                    result_data[i * n + c_col_ind[j]] = c_val[j];
                }     
            });
        });

        q.wait_and_throw();
        // clean up
        free(c_nnz, q);
        free(c_col_ind, q);
        free(c_row_ptr, q);
        free(c_val, q);
        sparse::release_matmat_descr(&descr);
        sparse::release_matrix_handle(&A);
        sparse::release_matrix_handle(&B);
        sparse::release_matrix_handle(&C);
    }

    void dns_dns_mul(int m, int n, int k, const SyncArray<kernel_type> &dense_mat,
                     const SyncArray<kernel_type> &origin_dense, SyncArray<kernel_type> &result){
        // Eigen::Map<const Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> denseMat(dense_mat.host_data(), k, n);
        // Eigen::Map<const Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        //         originDenseMat(origin_dense.host_data(), m, k);
        // Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> retMat = originDenseMat * denseMat;
        // Eigen::Map<Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(result.host_data(),
        //                                                                                          retMat.rows(),
        //                                                                                          retMat.cols()) = retMat;
        kernel_type one(1.0);
        kernel_type zero(0.0);
        auto &q = thunder::get_sycl_queue();
        
        auto gemm_event = blas::column_major::gemm(q, transpose::trans, transpose::nontrans,
                                                   m, n, k,
                                                   one, const_cast<kernel_type *>(origin_dense.device_data()), k,
                                                   const_cast<kernel_type *>(dense_mat.device_data()), k, zero,
                                                   result.device_data(), m, {});
        q.wait_and_throw();
    }
} // end namespce svm_kernel
