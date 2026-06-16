#ifndef _RPE_CUDA_KERNEL
#define _RPE_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void dot_prod_with_idx_forward_cuda(int N, int M, int h, int hdim, int n_max, const int L, at::Tensor q_tensor, at::Tensor index_q_tensor, at::Tensor index_q_offsets_tensor, at::Tensor k_tensor, at::Tensor index_k_tensor, at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor, at::Tensor output_tensor);
void dot_prod_with_idx_backward_cuda(int N, int M, int h, int hdim, int n_max, const int L, at::Tensor grad_out_tensor, at::Tensor q_tensor, at::Tensor index_q_offsets_tensor, at::Tensor k_tensor, at::Tensor index_k_offsets_tensor, at::Tensor index_k_tensor, at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor, at::Tensor grad_q_tensor, at::Tensor grad_k_tensor, at::Tensor grad_table_q_tensor, at::Tensor grad_table_k_tensor);

void dot_prod_with_idx_all_forward_cuda(int N, int M, int h, int hdim, int n_max, const int L, at::Tensor q_tensor, at::Tensor index_q_tensor, at::Tensor index_q_offsets_tensor, at::Tensor k_tensor, at::Tensor index_k_tensor, at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor, at::Tensor output_tensor);

void attention_step2_with_rel_pos_value_forward_cuda(int N, int M, int h, int hdim, int n_max, at::Tensor attn_tensor, at::Tensor v_tensor, at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor table_tensor, at::Tensor rel_idx_tensor, at::Tensor output_tensor);
void attention_step2_with_rel_pos_value_backward_cuda(int N, int M, int h, int hdim, int L, int n_max, at::Tensor grad_out_tensor, at::Tensor index0_tensor, at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor index1_offsets_tensor, at::Tensor attn_tensor, at::Tensor v_tensor, at::Tensor table_tensor, at::Tensor rel_idx_tensor, at::Tensor grad_attn_tensor, at::Tensor grad_v_tensor, at::Tensor grad_table_tensor);

// Templated launchers (scalar_t = float/at::Half/at::BFloat16). q/k/v/attn/table/
// output/grad_q/grad_k/grad_v/grad_out are scalar_t; the atomic-accumulated rel-pos
// table gradients (grad_table*) stay float* so the cross-block sum is f32-precise
// even under bf16. Index buffers stay int32. Explicit instantiations live in
// relative_pos_encoding_cuda_kernel.cu. Not extern "C" — templates have C++ linkage.
template <typename scalar_t>
void dot_prod_with_idx_forward_cuda_launcher(int N, int M, int h, int hdim, int n_max, const int L, const scalar_t *q, const int *index_q, const int *index_q_offsets, const scalar_t *k, const int *index_k, const scalar_t *table_q, const scalar_t *table_k, const int *rel_idx, scalar_t *output);
template <typename scalar_t>
void dot_prod_with_idx_backward_cuda_launcher(int N, int M, int h, int hdim, int n_max, const int L, const scalar_t *grad_out, const scalar_t *q, const int *index_q_offsets, const scalar_t *k, const int *index_k_offsets, const int *index_k, const scalar_t *table_q, const scalar_t *table_k, const int *rel_idx, scalar_t *grad_q, scalar_t *grad_k, float *grad_table_q, float *grad_table_k);

template <typename scalar_t>
void dot_prod_with_idx_all_forward_cuda_launcher(int N, int M, int h, int hdim, int n_max, const int L, const scalar_t *q, const int *index_q, const int *index_q_offsets, const scalar_t *k, const int *index_k, const scalar_t *table_q, const scalar_t *table_k, const int *rel_idx, scalar_t *output);

template <typename scalar_t>
void attention_step2_with_rel_pos_value_forward_cuda_launcher(int N, int M, const int h, int hdim, int n_max, const scalar_t *attn, const scalar_t *v, const int *index0_offsets, const int *index1, const scalar_t *table, const int *rel_idx, scalar_t *output);
template <typename scalar_t>
void attention_step2_with_rel_pos_value_backward_cuda_launcher(int N, int M, int h, int hdim, int L, int n_max, const scalar_t *grad_out, const int *index0, const int *index0_offsets, const int *index1, const int *index1_offsets, const scalar_t *attn, const scalar_t *v, const scalar_t *table, const int *rel_idx, scalar_t *grad_attn, scalar_t *grad_v, float *grad_table);

#endif
