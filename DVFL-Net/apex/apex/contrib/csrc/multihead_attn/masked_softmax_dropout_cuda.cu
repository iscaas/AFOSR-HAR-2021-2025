#include <iostream>
#include <math.h>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "dropout.cuh"
#include "softmax.cuh"

namespace multihead_attn {
namespace fused_softmax {
namespace mask_softmax_dropout {

std::vector<torch::Tensor> fwd_cuda(bool is_training, int heads,
                                    torch::Tensor const &input,
                                    const uint8_t *pad_mask,
                                    float dropout_prob) {
  const int attn_batches = input.size(0);
  const int sequences = attn_batches / heads;
  const int q_seq_len = input.size(1);
  const int k_seq_len = q_seq_len;
  const int dropout_elems = attn_batches * q_seq_len * k_seq_len;

  // There is no reason to use more than one stream as every kernel is
  // sequentially dependent
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  // 3 Intermediate Results + Output (Note: dropout intermediates are generated
  // by ATen library code)
  auto act_options = input.options().requires_grad(false);
  auto mask_options = act_options.dtype(torch::kUInt8);

  torch::Tensor softmax_results =
      torch::empty({attn_batches, q_seq_len, k_seq_len}, act_options);
  torch::Tensor dropout_results =
      torch::empty({attn_batches, q_seq_len, k_seq_len}, act_options);
  torch::Tensor dropout_mask =
      torch::empty({attn_batches, q_seq_len, k_seq_len}, mask_options);

  // Softmax Intermediate Result Ptr (used by Matmul1 -> Softmax)
  void *input_ptr = static_cast<void *>(input.data_ptr());
  void *softmax_results_ptr = static_cast<void *>(softmax_results.data_ptr());

  // Padded Softmax
  bool softmax_success = false;
  if (pad_mask == nullptr) {
    softmax_success = dispatch_softmax<half, half, float>(
        reinterpret_cast<half *>(softmax_results_ptr),
        reinterpret_cast<const half *>(input_ptr), k_seq_len, k_seq_len,
        attn_batches * q_seq_len);
  } else {
    softmax_success = dispatch_masked_softmax<half, half, float>(
        reinterpret_cast<half *>(softmax_results_ptr),
        reinterpret_cast<const half *>(input_ptr), pad_mask, k_seq_len,
        k_seq_len, attn_batches * q_seq_len,
        attn_batches * q_seq_len / sequences);
  }

  if (is_training) {
    // use at:: function so that C++ version generates the same random mask as
    // python version
    auto dropout_tuple =
        at::_fused_dropout(softmax_results, 1.0f - dropout_prob);
    dropout_results = std::get<0>(dropout_tuple);
    dropout_mask = std::get<1>(dropout_tuple);
  }

  // Matmul2

  return {dropout_results, dropout_mask, softmax_results};
}

torch::Tensor bwd_cuda(int heads, torch::Tensor const &output_grads,
                       torch::Tensor const &softmax_results,
                       torch::Tensor const &dropout_mask,
                       const uint8_t *padding_mask, float dropout_prob) {
  const int attn_batches = output_grads.size(0);
  const int q_seq_len = output_grads.size(1);
  const int k_seq_len = q_seq_len;
  const int dropout_elems = attn_batches * q_seq_len * k_seq_len;
  // TODO: Streams can be used in Backprop but I haven't added more than one
  // in my first attempt to create the code
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  // Output Tensor Allocations
  //  torch::Tensor input_grads         = torch::empty_like(output_grads);

  // Apply Dropout Mask and Scale by Dropout Probability
  // Softmax Grad
  if (padding_mask == nullptr) {
    dispatch_masked_scale_softmax_backward_stream<half, half, float, false>(
        static_cast<half *>(output_grads.data_ptr()),
        static_cast<half *>(output_grads.data_ptr()),
        reinterpret_cast<half const *>(softmax_results.data_ptr()),
        static_cast<uint8_t const *>(dropout_mask.data_ptr()),
        1.0 / (1.0 - dropout_prob), k_seq_len, k_seq_len,
        attn_batches * q_seq_len, stream);
  } else {
    dispatch_masked_scale_softmax_backward_masked_out_stream<half, half, float,
                                                             false>(
        static_cast<half *>(output_grads.data_ptr()),
        static_cast<half *>(output_grads.data_ptr()),
        reinterpret_cast<half const *>(softmax_results.data_ptr()),
        static_cast<uint8_t const *>(dropout_mask.data_ptr()),
        static_cast<uint8_t const *>(padding_mask), 1.0 / (1.0 - dropout_prob),
        k_seq_len, k_seq_len, attn_batches * q_seq_len, heads, stream);
  }
  // backward pass is completely in-place
  return output_grads;
}
} // namespace mask_softmax_dropout
} // namespace fused_softmax
} // namespace multihead_attn
