#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>

#include "type_shim.h"
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

template<typename T>
__device__ __forceinline__ bool is_aligned(T* p){
  return ((uint64_t)p) % (ILP*sizeof(T)) == 0;
}

template<typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset){
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

template<typename in_t, typename out_t>
struct L2NormScaleFunctor
{
  __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<2>& tl,
    float* output,
    float* output_per_tensor,
    float scale,
    bool per_tensor,
    int max_chunks_per_tensor)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    in_t* in = (in_t*)tl.addresses[0][tensor_loc];
    in += chunk_idx*chunk_size;

    out_t* out = (out_t*)tl.addresses[1][tensor_loc];
    out += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    __shared__ float s_vals[512];

    float vals[ILP]; // = {0}; // this probably works too but I want to be sure...
    in_t r_in[ILP];
    for(int i = 0; i < ILP; i++)
    {
      vals[i] = 0.f;
      r_in[i] = 0;
    }
    //bool finite = true;
    out_t r_out[ILP];

    // to make things simple, we put aligned case in a different code path
    if(n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(in) && is_aligned(out))
    {
      for(int i_start = threadIdx.x; i_start*ILP < n && i_start*ILP < chunk_size; i_start += blockDim.x)
      {
        // load
        load_store(r_in, in, 0 , i_start);
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          float next = static_cast<float>(r_in[ii]);
          r_out[ii] = next*scale;
          vals[ii] += next*next;
          //finite = finite && isfinite(r_in[ii]);
        }
        load_store(out, r_out, i_start, 0);
      }
    }
    else
    {
      for(int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x*ILP)
      {
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          r_in[ii] = 0;
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            r_in[ii] = in[i];
            float next = static_cast<float>(in[i]);
            vals[ii] += next*next;
          }
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          r_out[ii] = static_cast<float>(r_in[ii]) * scale;
         // finite = finite && isfinite(r_in[ii]);
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
            out[i] = r_out[ii];
        }
      }
    }

    float val = 0.f;
    for(int i = 0; i < ILP; i++)
        val += vals[i];

    float final = reduce_block_into_lanes(s_vals, val);

    if(threadIdx.x == 0)
    {
      if(!isfinite(final))
        *noop_gmem = 1; // Blindly fire off a write.  These will race but that's ok.
      output[blockIdx.x] += final;
      if(per_tensor)
        output_per_tensor[(tl.start_tensor_this_launch + tensor_loc)*max_chunks_per_tensor + chunk_idx] = final;
    }
  }
};
// Probably better to template, but since we are not likely to support other norm
template<typename x_t>
struct MaxNormFunctor
{
  __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<1>& tl,
    float* output,
    float* output_per_tensor,
    bool per_tensor,
    int max_chunks_per_tensor)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    x_t* x = (x_t*)tl.addresses[0][tensor_loc];
    x += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    __shared__ float s_vals[512];

    float vals[ILP]; // = {0}; // this probably works too but I want to be sure...
    x_t r_x[ILP];
    for(int i = 0; i < ILP; i++)
    {
      vals[i] = 0.f;
      r_x[i] = 0;
    }

    // to make things simple, we put aligned case in a different code path
    if(n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x))
    {
      for(int i_start = threadIdx.x; i_start*ILP < n && i_start*ILP < chunk_size; i_start += blockDim.x)
      {
        // load
        load_store(r_x, x, 0 , i_start);
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          float next = static_cast<float>(r_x[ii]);
          vals[ii] = fmaxf(fabsf(vals[ii]), fabsf(next));
        }
      }
    }
    else
    {
      for(int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x*ILP)
      {
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            float next = static_cast<float>(x[i]);
            vals[ii] = fmaxf(fabsf(vals[ii]), fabsf(next));
          }
        }
      }
    }

    float val = 0.f;
    for(int i = 0; i < ILP; i++)
        val = fmaxf(fabsf(val), fabsf(vals[i]));

    float final = reduce_block_into_lanes_max_op(s_vals, val);

    if(threadIdx.x == 0)
    {
      if(!isfinite(final))
        *noop_gmem = 1; // Blindly fire off a write.  These will race but that's ok.
      output[blockIdx.x] = fmaxf(fabsf(output[blockIdx.x]), fabsf(final));
      if(per_tensor)
        output_per_tensor[(tl.start_tensor_this_launch + tensor_loc)*max_chunks_per_tensor + chunk_idx] = final;
    }
  }
};

__global__ void cleanup_v3(
  float* output,
  float* output_per_tensor,
  float* ret,
  float* ret_per_tensor,
  bool per_tensor,
  int max_chunks_per_tensor)
{
  __shared__ float vals[512];

  if(blockIdx.x == 0)
  {
    float val = 0;
    if(threadIdx.x < 320)
      val = output[threadIdx.x];

    float final = reduce_block_into_lanes(vals, val);

    if(threadIdx.x == 0)
      *ret = sqrt(final);
  }

  if(per_tensor)
  {
    float* output_this_tensor = output_per_tensor + blockIdx.x*max_chunks_per_tensor;

    float val = 0;
    for(int i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x)
      val += output_this_tensor[i];

    float final = reduce_block_into_lanes(vals, val);

    if(threadIdx.x == 0)
      ret_per_tensor[blockIdx.x] = sqrt(final);
  }
}


std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_scale_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float scale,
  at::optional<bool> per_tensor_python)
{
  bool per_tensor = per_tensor_python.has_value() ? per_tensor_python.value() : false;

  auto float_options = tensor_lists[0][0].options().dtype(at::kFloat);
  auto output = at::zeros({320}, float_options);

  at::Tensor output_per_tensor;
  at::Tensor ret_per_tensor;

  int ntensors = tensor_lists[0].size();
  int max_chunks_per_tensor = -1;

  if(per_tensor)
  {
    for(int t = 0; t < ntensors; t++)
    {
      int max_chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1)/chunk_size;
      if(max_chunks_this_tensor > max_chunks_per_tensor)
        max_chunks_per_tensor = max_chunks_this_tensor;
    }
    output_per_tensor = at::zeros({ntensors*max_chunks_per_tensor}, float_options);
    ret_per_tensor = at::empty({ntensors}, float_options);
  }
  else
  {
    ret_per_tensor = at::empty({0}, float_options);
  }

  DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "multi_tensor_l2norm_scale_cuda",
  DISPATCH_FLOAT_AND_HALF(tensor_lists[1][0].scalar_type(), 1, "multi_tensor_l2norm_scale_cuda",
    multi_tensor_apply<2>(
      BLOCK_SIZE,
      chunk_size,
      noop_flag,
      tensor_lists,
      L2NormScaleFunctor<scalar_t_0, scalar_t_1>(),
      output.DATA_PTR<float>(),
      per_tensor ? output_per_tensor.DATA_PTR<float>() : nullptr,
      scale,
      per_tensor,
      max_chunks_per_tensor);))

  AT_CUDA_CHECK(cudaGetLastError());
  // AT_CUDA_CHECK(cudaDeviceSynchronize());

  // This involves one more small kernel launches, but will be negligible end to end.
  // I could get rid of these by hacking the functor + multi tensor harness with persistence
  // logic, but keeping it simple for now
  auto ret = at::empty({1}, output.options());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  auto stream = at::cuda::getCurrentCUDAStream();
  cleanup_v3<<<per_tensor ? ntensors : 1, 512, 0, stream>>>(
    output.DATA_PTR<float>(),
    per_tensor ? output_per_tensor.DATA_PTR<float>() : nullptr,
    ret.DATA_PTR<float>(),
    per_tensor ? ret_per_tensor.DATA_PTR<float>() : nullptr,
    per_tensor,
    max_chunks_per_tensor);

  return std::tuple<at::Tensor, at::Tensor>(ret, ret_per_tensor);
}


