//
// Created by jiashuai on 17-9-20.
//
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <thundersvm/syncarray.h>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>

#include "thundersvm/kernel/kernelmatrix_kernel.h"
#include <thundersvm/config.h>

namespace svm_kernel {
    void
    kernel_get_working_set_ins(const kernel_type *val, const int *col_ind, const int *row_ptr, const int *data_row_idx,
                               kernel_type *data_rows,
                               int m, int n, sycl::nd_item<3> item_ct1) {
        KERNEL_LOOP(i, m) {
            int row = data_row_idx[i];
            for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
                int col = col_ind[j];
                data_rows[col * m + i] = val[j]; // col-major for cuSPARSE
            }
        }
    }

    void
    kernel_RBF_kernel(const kernel_type *self_dot0, const kernel_type *self_dot1, kernel_type *dot_product, int m, int n,
                      kernel_type gamma, sycl::nd_item<3> item_ct1) {
        //m rows of kernel matrix, where m is the working set size; n is the number of training instances
        KERNEL_LOOP(idx, m * n) {
            int i = idx / n;//i is row id
            int j = idx % n;//j is column id
            dot_product[idx] = sycl::exp(
                -(self_dot0[i] + self_dot1[j] - dot_product[idx] * 2) * gamma);
        }
    }

    void
    kernel_RBF_kernel(const int *self_dot0_idx, const kernel_type *self_dot1, kernel_type *dot_product, int m, int n,
                      kernel_type gamma, sycl::nd_item<3> item_ct1) {
        //compute m rows of kernel matrix, where m is the working set size and n is the number of training instances, according to idx
        KERNEL_LOOP(idx, m * n) {
            int i = idx / n;//i is row id
            int j = idx % n;//j is column id
            dot_product[idx] =
                sycl::exp(-(self_dot1[self_dot0_idx[i]] + self_dot1[j] -
                            dot_product[idx] * 2) *
                          gamma);
        }
    }

    SYCL_EXTERNAL void
    kernel_sum_kernel_values(const float_type *coef, int total_sv,
                             const int *sv_start, const int *sv_count,
                             const float_type *rho, const kernel_type *k_mat,
                             float_type *dec_values, int n_classes,
                             int n_instances, sycl::nd_item<3> item_ct1) {
        KERNEL_LOOP(idx, n_instances) {
            int k = 0;
            int n_binary_models = n_classes * (n_classes - 1) / 2;
            for (int i = 0; i < n_classes; ++i) {
                for (int j = i + 1; j < n_classes; ++j) {
                    int si = sv_start[i];
                    int sj = sv_start[j];
                    int ci = sv_count[i];
                    int cj = sv_count[j];
                    const float_type *coef1 = &coef[(j - 1) * total_sv];
                    const float_type *coef2 = &coef[i * total_sv];
                    const kernel_type *k_values = &k_mat[idx * total_sv];
                    double sum = 0;
                    for (int l = 0; l < ci; ++l) {
                        sum += coef1[si + l] * k_values[si + l];
                    }
                    for (int l = 0; l < cj; ++l) {
                        sum += coef2[sj + l] * k_values[sj + l];
                    }
                    dec_values[idx * n_binary_models + k] = sum - rho[k];
                    k++;
                }
            }
        }
    }

    void
    kernel_poly_kernel(kernel_type *dot_product, kernel_type gamma, kernel_type coef0, int degree, int mn,
                       sycl::nd_item<3> item_ct1) {
        KERNEL_LOOP(idx, mn) {
            dot_product[idx] = sycl::pown(gamma * dot_product[idx] + coef0, degree);
        }
    }

    void kernel_sigmoid_kernel(kernel_type *dot_product, kernel_type gamma, kernel_type coef0, int mn,
                               sycl::nd_item<3> item_ct1) {
        KERNEL_LOOP(idx, mn) {
            dot_product[idx] = sycl::tanh(gamma * dot_product[idx] + coef0);
        }
    }

    void sum_kernel_values(const SyncArray<float_type> &coef, int total_sv,
                           const SyncArray<int> &sv_start,
                           const SyncArray<int> &sv_count,
                           const SyncArray<float_type> &rho,
                           const SyncArray<kernel_type> &k_mat,
                           SyncArray<float_type> &dec_values, int n_classes,
                           int n_instances) try {
        /*
        DPCT1038:58: When the kernel function name is used as a macro argument,
        the migration result may be incorrect. You need to verify the definition
        of the macro.
        */
        /*
        DPCT1001:60: The statement could not be removed.
        */
        /*
        DPCT1002:61: Special case error handling if-stmt was detected. You may
        need to rewrite this code.
        */
        /*
        DPCT1010:62: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        0;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    void get_working_set_ins(const SyncArray<kernel_type> &val,
                             const SyncArray<int> &col_ind,
                             const SyncArray<int> &row_ptr,
                             const SyncArray<int> &data_row_idx,
                             SyncArray<kernel_type> &data_rows, int m,
                             int n) try {
        /*
        DPCT1038:63: When the kernel function name is used as a macro argument,
        the migration result may be incorrect. You need to verify the definition
        of the macro.
        */
        /*
        DPCT1001:64: The statement could not be removed.
        */
        /*
        DPCT1002:65: Special case error handling if-stmt was detected. You may
        need to rewrite this code.
        */
        /*
        DPCT1010:66: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        0;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    void RBF_kernel(const SyncArray<kernel_type> &self_dot0,
                    const SyncArray<kernel_type> &self_dot1,
                    SyncArray<kernel_type> &dot_product, int m, int n,
                    kernel_type gamma) try {
        /*
        DPCT1038:67: When the kernel function name is used as a macro argument,
        the migration result may be incorrect. You need to verify the definition
        of the macro.
        */
        /*
        DPCT1001:68: The statement could not be removed.
        */
        /*
        DPCT1002:69: Special case error handling if-stmt was detected. You may
        need to rewrite this code.
        */
        /*
        DPCT1010:70: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        0;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    void RBF_kernel(const SyncArray<int> &self_dot0_idx,
                    const SyncArray<kernel_type> &self_dot1,
                    SyncArray<kernel_type> &dot_product, int m, int n,
                    kernel_type gamma) try {
        /*
        DPCT1038:71: When the kernel function name is used as a macro argument,
        the migration result may be incorrect. You need to verify the definition
        of the macro.
        */
        /*
        DPCT1001:72: The statement could not be removed.
        */
        /*
        DPCT1002:73: Special case error handling if-stmt was detected. You may
        need to rewrite this code.
        */
        /*
        DPCT1010:74: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        0;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    void poly_kernel(SyncArray<kernel_type> &dot_product, kernel_type gamma,
                     kernel_type coef0, int degree, int mn) try {
        /*
        DPCT1038:75: When the kernel function name is used as a macro argument,
        the migration result may be incorrect. You need to verify the definition
        of the macro.
        */
        /*
        DPCT1001:76: The statement could not be removed.
        */
        /*
        DPCT1002:77: Special case error handling if-stmt was detected. You may
        need to rewrite this code.
        */
        /*
        DPCT1010:78: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        0;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    void sigmoid_kernel(SyncArray<kernel_type> &dot_product, kernel_type gamma,
                        kernel_type coef0, int mn) try {
        /*
        DPCT1038:79: When the kernel function name is used as a macro argument,
        the migration result may be incorrect. You need to verify the definition
        of the macro.
        */
        /*
        DPCT1001:80: The statement could not be removed.
        */
        /*
        DPCT1002:81: Special case error handling if-stmt was detected. You may
        need to rewrite this code.
        */
        /*
        DPCT1010:82: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        0;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    sycl::queue *handle;
    oneapi::mkl::index_base descr;
    bool cusparse_init;

    void dns_csr_mul(int m, int n, int k, const SyncArray<kernel_type> &dense_mat, const SyncArray<kernel_type> &csr_val,
                     const SyncArray<int> &csr_row_ptr, const SyncArray<int> &csr_col_ind, int nnz,
                     SyncArray<kernel_type> &result) {
        if (!cusparse_init) {
            handle = &dpct::get_default_queue();
            descr = oneapi::mkl::index_base::zero;
            descr = oneapi::mkl::index_base::zero;
            /*
            DPCT1026:91: The call to cusparseSetMatType was removed because the
            function call is redundant in DPC++.
            */
            cusparse_init = true;
        }
        kernel_type one(1);
        kernel_type zero(0);

#if (SYCL_LANGUAGE_VERSION >= 11000)

        cusparseSpMatDescr_t matA;
        cusparseDnMatDescr_t matB, matC;
#ifdef USE_DOUBLE
        cudaDataType data_type = CUDA_R_64F;
#else//kernel type is float
        int data_type = 0;
#endif
        /*
        DPCT1007:83: Migration of cusparseCreateCsr is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        cusparseCreateCsr(&matA, m, k, nnz, (void *)csr_row_ptr.device_data(),
                          (void *)csr_col_ind.device_data(),
                          (void *)csr_val.device_data(), CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I, oneapi::mkl::index_base::zero,
                          data_type);
        /*
        DPCT1007:84: Migration of cusparseCreateDnMat is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        cusparseCreateDnMat(&matB, n, k, n, (void *)dense_mat.device_data(),
                            data_type, CUSPARSE_ORDER_COL);
        /*
        DPCT1007:85: Migration of cusparseCreateDnMat is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        cusparseCreateDnMat(&matC, m, n, m, (void *)result.device_data(),
                            data_type, CUSPARSE_ORDER_COL);

        size_t buffer_size = 0;
        /*
        DPCT1007:86: Migration of cusparseSpMM_bufferSize is not supported by
        the Intel(R) DPC++ Compatibility Tool.
        */
        cusparseSpMM_bufferSize(handle, oneapi::mkl::transpose::nontrans,
                                oneapi::mkl::transpose::trans, &one, matA, matB,
                                &zero, matC, data_type, CUSPARSE_CSRMM_ALG1,
                                &buffer_size);

        void *p_buffer = nullptr;
        p_buffer =
            (void *)sycl::malloc_device(buffer_size, dpct::get_default_queue());

        /*
        DPCT1007:87: Migration of cusparseSpMM is not supported by the Intel(R)
        DPC++ Compatibility Tool.
        */
        cusparseSpMM(handle, oneapi::mkl::transpose::nontrans,
                     oneapi::mkl::transpose::trans, &one, matA, matB, &zero,
                     matC, data_type, CUSPARSE_CSRMM_ALG1, p_buffer);

        sycl::free(p_buffer, dpct::get_default_queue());
        /*
        DPCT1007:88: Migration of cusparseDestroySpMat is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        cusparseDestroySpMat(matA);
        /*
        DPCT1007:89: Migration of cusparseDestroyDnMat is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        cusparseDestroyDnMat(matB);
        /*
        DPCT1007:90: Migration of cusparseDestroyDnMat is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        cusparseDestroyDnMat(matC);

#else

#ifdef USE_DOUBLE
        cusparseDcsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                        m, n, k, nnz, &one, descr, csr_val.device_data(), csr_row_ptr.device_data(),
                        csr_col_ind.device_data(),
                        dense_mat.device_data(), n, &zero, result.device_data(), m);
#else//kernel type is float
        cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                        m, n, k, nnz, &one, descr, csr_val.device_data(), csr_row_ptr.device_data(),
                        csr_col_ind.device_data(),
                        dense_mat.device_data(), n, &zero, result.device_data(), m);

        //cusparseScsrmm return row-major matrix, so no transpose is needed
#endif // ifdef USE_DOUBLE

#endif // if CUDART_VERSION >= 11000
    }
}
