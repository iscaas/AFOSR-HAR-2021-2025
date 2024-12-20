/*
* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#include "norm_sample.h"
#include <cudnn_frontend.h>
#include "cudnn_backend.h"
#include <ATen/cudnn/Handle.h>  // for getcudnnhandle
#include <torch/extension.h>
#include <torch/torch.h>

// some helpers
int64_t checkCudaError(cudaError_t code, const char* expr, const char* file, int line) {
    if (code) {
        printf("CUDA error at %s:%d, code=%d (%s) in '%s'", file, line, (int)code, cudaGetErrorString(code), expr);
	return 1;
    }
    return 0;
}

int64_t checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line) {
    if (code) {
        printf("CUDNN error at %s:%d, code=%d (%s) in '%s'\n", file, line, (int)code, cudnnGetErrorString(code), expr);
        return 1;
    }
    return 0;
}

bool
AllowAll(cudnnBackendDescriptor_t engine_config) {
  (void)engine_config;
  return false;
}

void generateStrides(const int64_t* dimA, int64_t* strideA, int64_t nbDims, cudnnTensorFormat_t filterFormat) {
  // For INT8x4 and INT8x32 we still compute standard strides here to input
  // into the cuDNN functions. We will manually scale by resizeFactor in the cpu ref.
  if (filterFormat == CUDNN_TENSOR_NCHW) {
    strideA[nbDims - 1] = 1;
    for (int64_t d = nbDims - 2; d >= 0; d--) {
      strideA[d] = strideA[d + 1] * dimA[d + 1];
    }
  } else {
    // Here we assume that the format is CUDNN_TENSOR_NHWC
    strideA[1]          = 1;
    strideA[nbDims - 1] = strideA[1] * dimA[1];
    for (int64_t d = nbDims - 2; d >= 2; d--) {
      strideA[d] = strideA[d + 1] * dimA[d + 1];
    }
    strideA[0] = strideA[2] * dimA[2];
  }
}


// runtime
cudnn_frontend::ExecutionPlan run_batch_norm_forward(int64_t *tensorDims,
						     int64_t *perChannelSum,
						     int64_t *epsilon,
						     int64_t *peerDims,
						     cudnnDataType_t data_type) {

  // get the cudnn handle
  cudnnHandle_t handle = torch::native::getCudnnHandle();

  // Creates the necessary tensor descriptors
  int64_t tensor_stride[4];
  int64_t stride[4];
  int64_t peer_stride[4];

  // NHWC format. GenerateStrides() takes care of this. Howeever, tensor dims should still be NCHW
  generateStrides(tensorDims, tensor_stride, (int64_t)4, CUDNN_TENSOR_NHWC);
  generateStrides(peerDims, peer_stride, (int64_t)4, CUDNN_TENSOR_NHWC);

  auto tensor_create = [&tensor_stride, &tensorDims](cudnnDataType_t type,
  int64_t id) {
    return cudnn_frontend::TensorBuilder()
      .setDim(4, tensorDims)
        .setStrides(4, tensor_stride)
          .setId(id)
            .setAlignment(16)
              .setDataType(type)
                .build();
  };

  auto peer_tensor_create = [&peer_stride, &tensorDims](cudnnDataType_t type,
  int64_t id) {
    return cudnn_frontend::TensorBuilder()
      .setDim(4, tensorDims)
        .setStrides(4, peer_stride)
          .setId(id)
            .setAlignment(16)
              .setDataType(type)
                .build();
  };


  generateStrides(perChannelSum, stride, (int64_t)4, CUDNN_TENSOR_NHWC);

  auto per_channel_tensor_create = [&stride, &perChannelSum](cudnnDataType_t type, int64_t id) {
    return cudnn_frontend::TensorBuilder()
      .setDim(4, perChannelSum)
        .setStrides(4, stride)
          .setId(id)
            .setAlignment(16)
              .setDataType(type)
                .build();
  };

  auto xTensor             = tensor_create(data_type, 100);
  auto yTensor             = tensor_create(data_type, 101);
  auto scaleTensor         = per_channel_tensor_create(CUDNN_DATA_FLOAT, 102);
  auto biasTensor          = per_channel_tensor_create(CUDNN_DATA_FLOAT, 103);
  auto inMeanTensor        = per_channel_tensor_create(CUDNN_DATA_FLOAT, 104);
  auto inVarTensor         = per_channel_tensor_create(CUDNN_DATA_FLOAT, 105);
  auto outMeanTensor       = per_channel_tensor_create(CUDNN_DATA_FLOAT, 106);
  auto outVarTensor        = per_channel_tensor_create(CUDNN_DATA_FLOAT, 107);
  auto savedMeanTensor     = per_channel_tensor_create(CUDNN_DATA_FLOAT, 108);
  auto savedInvVarTensor   = per_channel_tensor_create(CUDNN_DATA_FLOAT, 109);

  
  int64_t epsilon_stride[4];
  generateStrides(epsilon, epsilon_stride, (int64_t)4, CUDNN_TENSOR_NHWC);
  auto scalar_tensor_create = [&epsilon_stride, &epsilon](cudnnDataType_t type, int64_t id) {
    return cudnn_frontend::TensorBuilder()
      .setDim(4, epsilon)
        .setStrides(4, epsilon_stride)
          .setId(id)
            .setAlignment(16)
              .setDataType(type)
                .setByValue(true)
                  .build();
  };

  auto epsilonTensor       = scalar_tensor_create(CUDNN_DATA_DOUBLE, 110);
  auto expDecayTensor      = scalar_tensor_create(CUDNN_DATA_DOUBLE, 111);

  // Create the two peer stat tensors. Jump IDs in case we need to add more tensors with UIDs
  std::vector<cudnn_frontend::Tensor_v8> peerStatTensors;
  for (size_t i = 112; i < 112 + peerDims[0]; ++i) {
    peerStatTensors.push_back(peer_tensor_create(CUDNN_DATA_FLOAT, i));
  }

#if (CUDNN_VERSION >= 8500)
  // Batch normalization
  cudnnBackendNormMode_t normalizationMode = CUDNN_BATCH_NORM;

  // Forward training
  cudnnBackendNormFwdPhase_t phase = CUDNN_NORM_FWD_TRAINING;

  //Create a Finalize node
  auto batch_norm_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR)
    .setNormalizationMode(normalizationMode)
    .setNormFwdPhase(phase)
    .setxDesc(xTensor)
    .setScaleAndBias(scaleTensor, biasTensor)
    .setPrevRunningMeanAndVar(inMeanTensor, inVarTensor)
    .setNextRunningMeanAndVar(outMeanTensor, outVarTensor)
    .setSavedMeanAndInvVar(savedMeanTensor, savedInvVarTensor)
    .setEpsilonTensor(epsilonTensor)
    .setExpDecayFactorTensor(expDecayTensor)
    .setPeerStatTensor(peerStatTensors)
    .setyDesc(yTensor)
    .build();

  std::array<cudnn_frontend::Operation const*, 1> ops = {&batch_norm_op};
#else
  std::array<cudnn_frontend::Operation const*, 0> ops = {};
#endif
  auto opGraph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(ops.size(), ops.data()).build();
  //std::cout << opGraph.describe() << std::endl;

  cudnn_frontend::EngineConfigList filtered_configs;
  auto statuses =
    cudnn_frontend::get_heuristics_list<2>({"heuristics_instant"
      , "heuristics_fallback"
    }, opGraph,::AllowAll, filtered_configs, true);

  //std::cout << "get_heuristics_list Statuses: ";
  //for (auto i = 0u ; i < statuses.size(); i++) {
  //  std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
  //}
  //std::cout << std::endl;
  //std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

  // some verbose printing:
  //std::cout << "Tensor shape: (" << tensorDims[0] << ", " << tensorDims[1] << ", " << tensorDims[2] << ", " << tensorDims[3] << ")" << std::endl;
  
  auto plan_builder = [&filtered_configs, &opGraph, &handle]() {
    for (auto i = 0u; i < filtered_configs.size(); i++) {
      try {
        auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle).setEngineConfig(filtered_configs[i], opGraph.getTag()).build();
        return plan;
      } catch (cudnn_frontend::cudnnException &e) {
        continue;
      }
    }
    return cudnn_frontend::ExecutionPlanBuilder().setHandle(handle).setEngineConfig(filtered_configs[0], opGraph.getTag()).build();
  };

  assert(filtered_configs.size() > 0);
  auto plan = plan_builder();

  return plan;

}

void execute_batch_norm_forward(cudnn_frontend::ExecutionPlan plan,
				void *xDevPtr,
				void *yDevPtr,
				void *scaledevPtr,
				void *biasdevPtr,
				void *in_meandevPtr,
				void *in_vardevPtr,
				void *out_meandevPtr,
				void *out_vardevPtr,
				void *saved_meandevPtr,
				void *saved_inv_vardevPtr,
				const std::vector<void*> &peer_devPtrs,
				double epsilon_val,
				double exponential_decay_factor,
				size_t peer_size, 
				int rank_id) {

  // get handle
  cudnnHandle_t handle_ = torch::native::getCudnnHandle();
  
  // get stream                                                                                                                                                          
  cudaStream_t stream;
  cudnnGetStream(handle_, &stream);
  
  try {
    // allocate workspace
    auto workspace_size = plan.getWorkspaceSize();
    auto workspace_tensor = at::empty({(workspace_size+3)/4}, at::TensorOptions(at::kCUDA).dtype(at::kFloat));
    void* workPtr = nullptr;
    if (workspace_size > 0) {
      workPtr = workspace_tensor.data_ptr<float>();
    }
    
    // first the data pointers
    std::vector<void*> data_ptrs {xDevPtr, yDevPtr, scaledevPtr, biasdevPtr,
	in_meandevPtr, in_vardevPtr, out_meandevPtr, out_vardevPtr,
	saved_meandevPtr, saved_inv_vardevPtr, 
	&epsilon_val, &exponential_decay_factor};
    data_ptrs.insert(data_ptrs.end(), peer_devPtrs.begin(), peer_devPtrs.end());
    // then the uids
    std::vector<int64_t> uids;
    for (size_t i = 100; i < 100 + data_ptrs.size(); ++i) {
      uids.push_back(i);
    }
    auto variantPack  = cudnn_frontend::VariantPackBuilder()
      .setWorkspacePointer(workPtr)
      .setDataPointers(data_ptrs.size(), data_ptrs.data())
      .setUids(uids.size(), uids.data())
      .build();
    //std::cout << "variantPack " << variantPack.describe() << std::endl;
    cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());    
    cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    // Reset local communication buffer
    cudaMemsetAsync(peer_devPtrs[rank_id], 0, peer_size*4, stream);
    
  } catch (cudnn_frontend::cudnnException &e) {
    struct cudaDeviceProp prop;
    checkCudaErr(cudaGetDeviceProperties(&prop, 0));
    if (prop.major == 8) {
      std::cout << "[ERROR] Exception " << e.what() << std::endl;
      assert(false);
    }
  }
}

cudnn_frontend::ExecutionPlan run_batch_norm_backward(int64_t *tensorDims,
						      int64_t *perChannelSum,
						      int64_t *epsilon,
						      int64_t *peerDims,
						      cudnnDataType_t data_type) {

  // get cudnn handle
  cudnnHandle_t handle = torch::native::getCudnnHandle();

  // Creates the necessary tensor descriptors
  int64_t tensor_stride[4];
  int64_t stride[4];
  int64_t peer_stride[4];

  // NHWC format. GenerateStrides() takes care of this. Howeever, tensor dims should still be NCHW
  generateStrides(tensorDims, tensor_stride, (int64_t)4, CUDNN_TENSOR_NHWC);
  generateStrides(peerDims, peer_stride, (int64_t)4, CUDNN_TENSOR_NHWC);

  auto tensor_create = [&tensor_stride, &tensorDims](cudnnDataType_t type, int64_t id) {
      return cudnn_frontend::TensorBuilder()
        .setDim(4, tensorDims)
          .setStrides(4, tensor_stride)
            .setId(id)
              .setAlignment(16)
                .setDataType(type)
                  .build();
  };

  auto peer_tensor_create = [&peer_stride, &peerDims](cudnnDataType_t type, int64_t id) {
      return cudnn_frontend::TensorBuilder()
        .setDim(4, peerDims)
          .setStrides(4, peer_stride)
            .setId(id)
              .setAlignment(16)
                .setDataType(type)
                  .build();
  };

  generateStrides(perChannelSum, stride, (int64_t)4, CUDNN_TENSOR_NHWC);

  auto per_channel_tensor_create = [&stride, &perChannelSum](cudnnDataType_t type, int64_t id) {
      return cudnn_frontend::TensorBuilder()
        .setDim(4, perChannelSum)
          .setStrides(4, stride)
            .setId(id)
              .setAlignment(16)
                .setDataType(type)
                  .build();
  };

  auto xTensor             = tensor_create(data_type, 100);
  auto dyTensor            = tensor_create(data_type, 101);
  auto scaleTensor         = per_channel_tensor_create(CUDNN_DATA_FLOAT, 102);
  auto savedMeanTensor     = per_channel_tensor_create(CUDNN_DATA_FLOAT, 103);
  auto savedInvVarTensor   = per_channel_tensor_create(CUDNN_DATA_FLOAT, 104);
  auto dxTensor            = tensor_create(data_type, 105);
  auto dScaleTensor        = per_channel_tensor_create(CUDNN_DATA_FLOAT, 106);
  auto dBiasTensor         = per_channel_tensor_create(CUDNN_DATA_FLOAT, 107);

  int64_t epsilon_stride[4];
  generateStrides(epsilon, epsilon_stride, (int64_t)4, CUDNN_TENSOR_NHWC);
  auto scalar_tensor_create = [&epsilon_stride, &epsilon](cudnnDataType_t type, int64_t id) {
      return cudnn_frontend::TensorBuilder()
        .setDim(4, epsilon)
          .setStrides(4, epsilon_stride)
            .setId(id)
              .setAlignment(16)
                .setDataType(type)
                  .setByValue(true)
                    .build();
  };

  auto epsilonTensor       = scalar_tensor_create(CUDNN_DATA_DOUBLE, 108);

  std::vector<cudnn_frontend::Tensor_v8> peerStatTensors;
  for (size_t i = 109; i < 109 + peerDims[0]; ++i) {
    peerStatTensors.push_back(peer_tensor_create(CUDNN_DATA_FLOAT, i));
  }

#if (CUDNN_VERSION >= 8500)
  // Batch normalization
  cudnnBackendNormMode_t normalizationMode = CUDNN_BATCH_NORM;

  //Create a Finalize node
  auto batch_norm_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR)
    .setNormalizationMode(normalizationMode)
    .setxDesc(xTensor)
    .setSavedMeanAndInvVar(savedMeanTensor, savedInvVarTensor)
    .setdyDesc(dyTensor)
    .setScale(scaleTensor)
    .setEpsilonTensor(epsilonTensor)
    .setDScaleAndDBias(dScaleTensor, dBiasTensor)
    .setdxDesc(dxTensor)
    .setPeerStatTensor(peerStatTensors)
    .build();

  std::array<cudnn_frontend::Operation const*, 1> ops = {&batch_norm_op};
#else
  std::array<cudnn_frontend::Operation const*, 0> ops = {};
#endif
    
  auto opGraph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(ops.size(), ops.data()).build();
  //std::cout << opGraph.describe() << std::endl;

  cudnn_frontend::EngineConfigList filtered_configs;
  auto statuses =
    cudnn_frontend::get_heuristics_list<2>({"heuristics_instant"
					    , "heuristics_fallback"
      }, opGraph,::AllowAll, filtered_configs, true);
    
  auto plan_builder = [&filtered_configs, &opGraph, &handle]() {
    for (auto i = 0u; i < filtered_configs.size(); i++) {
      try {
        auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle).setEngineConfig(filtered_configs[i], opGraph.getTag()).build();
        return plan;
      } catch (cudnn_frontend::cudnnException &e) {
        continue;
      }
    }
    return cudnn_frontend::ExecutionPlanBuilder().setHandle(handle).setEngineConfig(filtered_configs[0], opGraph.getTag()).build();
  };

  assert(filtered_configs.size() > 0);
  auto plan = plan_builder();

  return plan;
}

void execute_batch_norm_backward(cudnn_frontend::ExecutionPlan plan,
				 void *xDevPtr,
				 void *dyDevPtr,
				 void *scaledevPtr,
				 void *saved_meandevPtr,
				 void *saved_inv_vardevPtr,
				 const std::vector<void*> &peer_devPtrs,
				 void *dxDevPtr,
				 void *dscaledevPtr,
				 void *dbiasdevPtr,
				 double epsilon_val,
				 size_t peer_size,
				 int rank_id) {

  // get handle
  cudnnHandle_t handle_ = torch::native::getCudnnHandle();
  
  // get stream
  cudaStream_t stream;
  cudnnGetStream(handle_, &stream);
      
  try {
    // allocate workspace
    auto workspace_size = plan.getWorkspaceSize();
    auto workspace_tensor = at::empty({(workspace_size+3)/4}, at::TensorOptions(at::kCUDA).dtype(at::kFloat));
    void* workPtr = nullptr;
    if (workspace_size > 0) {
      workPtr = workspace_tensor.data_ptr<float>();
    }
    
    // create helper arrays
    std::vector<void*> data_ptrs {xDevPtr, dyDevPtr, scaledevPtr,
	saved_meandevPtr, saved_inv_vardevPtr,
	dxDevPtr, dscaledevPtr, dbiasdevPtr, &epsilon_val};
    data_ptrs.insert(data_ptrs.end(), peer_devPtrs.begin(), peer_devPtrs.end());
    std::vector<int64_t> uids;
    for (size_t i = 100; i < 100 + data_ptrs.size(); ++i) {
      uids.push_back(i);
    }
    
    auto variantPack  = cudnn_frontend::VariantPackBuilder()
      .setWorkspacePointer(workPtr)
      .setDataPointers(data_ptrs.size(), data_ptrs.data())
      .setUids(uids.size(), uids.data())
      .build();
    cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());

    cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    // Reset local communication buffer
    cudaMemsetAsync(peer_devPtrs[rank_id], 0, peer_size*4, stream);
    
  } catch (cudnn_frontend::cudnnException &e) {
    struct cudaDeviceProp prop;
    checkCudaErr(cudaGetDeviceProperties(&prop, 0));
    if (prop.major == 8) {
      std::cout << "[ERROR] Exception " << e.what() << std::endl;
      assert(false);
    }
  }
}
