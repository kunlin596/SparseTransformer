#include <vector>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "attention_cuda_kernel.h"

// Dispatch over fp32 + half + bf16. The macro binds scalar_t for the lambda;
// feature tensors are read/written as scalar_t (the launcher accumulates in f32).


void attention_step1_forward_cuda(int N_q, int N_k, int M, int h, int hdim, const unsigned int n_max, at::Tensor q_tensor, at::Tensor k_tensor,
    at::Tensor index0_tensor, at::Tensor index1_tensor, at::Tensor attn_tensor)
{
    const int *index0 = index0_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q_tensor.scalar_type(), "attention_step1_forward_cuda", [&] {
        const scalar_t *q = q_tensor.data_ptr<scalar_t>();
        const scalar_t *k = k_tensor.data_ptr<scalar_t>();
        scalar_t *attn = attn_tensor.data_ptr<scalar_t>();
        attention_step1_forward_cuda_launcher<scalar_t>(N_q, N_k, M, h, hdim, n_max, q, k, index0, index1, attn);
    });
}

void attention_step1_backward_cuda(int N, int M, int h, int hdim, const unsigned int n_max, at::Tensor grad_out_tensor,
    at::Tensor index0_tensor, at::Tensor index0_tensor_offsets, at::Tensor index1_tensor, at::Tensor index1_tensor_offsets, at::Tensor q_tensor, at::Tensor k_tensor,
    at::Tensor grad_q_tensor, at::Tensor grad_k_tensor)
{
    const int *index0 = index0_tensor.data_ptr<int>();
    const int *index0_offsets = index0_tensor_offsets.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    const int *index1_offsets = index1_tensor_offsets.data_ptr<int>();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q_tensor.scalar_type(), "attention_step1_backward_cuda", [&] {
        const scalar_t *grad_out = grad_out_tensor.data_ptr<scalar_t>();
        const scalar_t *q = q_tensor.data_ptr<scalar_t>();
        const scalar_t *k = k_tensor.data_ptr<scalar_t>();
        scalar_t *grad_q = grad_q_tensor.data_ptr<scalar_t>();
        scalar_t *grad_k = grad_k_tensor.data_ptr<scalar_t>();
        attention_step1_backward_cuda_launcher<scalar_t>(N, M, h, hdim, n_max, grad_out, index0, index0_offsets, index1, index1_offsets, q, k, grad_q, grad_k);
    });
}

void attention_step2_forward_cuda(int N, int M, int h, int hdim, int n_max, at::Tensor attn_tensor, at::Tensor v_tensor,
    at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor output_tensor)
{
    const int *index0_offsets = index0_offsets_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, attn_tensor.scalar_type(), "attention_step2_forward_cuda", [&] {
        const scalar_t *attn = attn_tensor.data_ptr<scalar_t>();
        const scalar_t *v = v_tensor.data_ptr<scalar_t>();
        scalar_t *output = output_tensor.data_ptr<scalar_t>();
        attention_step2_forward_cuda_launcher<scalar_t>(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, output);
    });
}

void attention_step2_backward_cuda(int N, int M, int h, int hdim, int n_max, at::Tensor grad_out_tensor, at::Tensor index0_tensor,
    at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor index1_offsets_tensor, at::Tensor attn_tensor, at::Tensor v_tensor,
    at::Tensor grad_attn_tensor, at::Tensor grad_v_tensor)
{
    const int *index0 = index0_tensor.data_ptr<int>();
    const int *index0_offsets = index0_offsets_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    const int *index1_offsets = index1_offsets_tensor.data_ptr<int>();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_out_tensor.scalar_type(), "attention_step2_backward_cuda", [&] {
        const scalar_t *grad_out = grad_out_tensor.data_ptr<scalar_t>();
        const scalar_t *attn = attn_tensor.data_ptr<scalar_t>();
        const scalar_t *v = v_tensor.data_ptr<scalar_t>();
        scalar_t *grad_attn = grad_attn_tensor.data_ptr<scalar_t>();
        scalar_t *grad_v = grad_v_tensor.data_ptr<scalar_t>();
        attention_step2_backward_cuda_launcher<scalar_t>(N, M, h, hdim, n_max, grad_out, index0, index0_offsets, index1, index1_offsets, attn, v, grad_attn, grad_v);
    });
}
