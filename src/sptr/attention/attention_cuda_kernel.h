#ifndef _ATTENTION_CUDA_KERNEL
#define _ATTENTION_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void attention_step1_forward_cuda(int N_q, int N_k, int M, int h, int hdim, const unsigned int n_max, at::Tensor q_tensor, at::Tensor k_tensor, at::Tensor index0_tensor, at::Tensor index1_tensor, at::Tensor attn_tensor);
void attention_step1_backward_cuda(int N, int M, int h, int hdim, const unsigned int n_max, at::Tensor grad_out_tensor, at::Tensor index0_tensor, at::Tensor index0_tensor_offsets, at::Tensor index1_tensor, at::Tensor index1_tensor_offsets, at::Tensor q_tensor, at::Tensor k_tensor, at::Tensor grad_q_tensor, at::Tensor grad_k_tensor);

void attention_step2_forward_cuda(int N, int M, int h, int hdim, int n_max, at::Tensor attn_tensor, at::Tensor v_tensor, at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor output_tensor);
void attention_step2_backward_cuda(int N, int M, int h, int hdim, int n_max, at::Tensor grad_out_tensor, at::Tensor index0_tensor, at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor index1_offsets_tensor, at::Tensor attn_tensor, at::Tensor v_tensor, at::Tensor grad_attn_tensor, at::Tensor grad_v_tensor);

// Templated launchers (scalar_t = float/at::Half/at::BFloat16). The kernels load
// scalar_t and accumulate in float (f32-accumulate bf16 pattern); index buffers
// stay int32. Explicit instantiations live in attention_cuda_kernel.cu. Not
// extern "C" — templates have C++ linkage.
template <typename scalar_t>
void attention_step1_forward_cuda_launcher(int N_q, int N_k, int M, int h, int hdim, const unsigned int n_max, const scalar_t *q, const scalar_t *k, const int *index0, const int *index1, scalar_t *attn);
template <typename scalar_t>
void attention_step1_backward_cuda_launcher(int N, int M, int h, int hdim, const unsigned int n_max, const scalar_t *grad_out, const int *index0, const int *index0_offsets, const int *index1, const int *index1_offsets, const scalar_t *q, const scalar_t *k, scalar_t *grad_q, scalar_t *grad_k);

template <typename scalar_t>
void attention_step2_forward_cuda_launcher(int N, int M, const int h, int hdim, int n_max, const scalar_t *attn, const scalar_t *v, const int *index0_offsets, const int *index1, scalar_t *output);
template <typename scalar_t>
void attention_step2_backward_cuda_launcher(int N, int M, int h, int hdim, int n_max, const scalar_t *grad_out, const int *index0, const int *index0_offsets, const int *index1, const int *index1_offsets, const scalar_t *attn, const scalar_t *v, scalar_t *grad_attn, scalar_t *grad_v);

#endif
