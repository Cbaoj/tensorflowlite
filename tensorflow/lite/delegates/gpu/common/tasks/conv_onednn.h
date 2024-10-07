/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifdef TFLITE_ENABLE_ONEDNN

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_ONEDNN_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_ONEDNN_H_

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_conversion.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_operation.h"
#include "oneapi/dnnl/dnnl.hpp"

namespace tflite {
namespace gpu {


class ConvOneDNN : public GPUOperation {
 public:
   enum class onednn_op_type {
     op_invalid,
     op_conv_forward,
     op_relu,
   };

  struct onednn_primitive_desc {
    onednn_op_type op_type;
    std::vector<dnnl::post_ops> post_ops;
    std::vector<dnnl::memory::desc> mem_descs;
    std::vector<dnnl::memory::dim> mem_dims;
    dnnl::primitive_desc primitive_desc;
    dnnl::primitive_attr pritimive_attr;
};

  ConvOneDNN() = default;

  ConvOneDNN(ConvOneDNN&& operation);
  ConvOneDNN& operator=(ConvOneDNN&& operation);
  ConvOneDNN(const ConvOneDNN&) = delete;
  ConvOneDNN& operator=(const ConvOneDNN&) = delete;

  absl::Status AddOperation(const GpuInfo& gpu_info, GPUOperation* operation);
  absl::Status AssembleCode(const GpuInfo& gpu_info);
  absl::Status prepare_dnn_kernel(const cl::CreationContext& creation_context, cl::CLArguments &cl_args);
  absl::Status AddToQueue(cl::CLCommandQueue* queue);
  //std::unordered_map<int, dnnl::memory> get_arguments();
  //dnnl::primitive*  get_primitive_descriptor();
  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights);
  template <DataType T>
  void UploadBias(const tflite::gpu::Tensor<Linear, T>& bias);

 private:
  ConvOneDNN( const OperationDef& definition,
              const Convolution2DAttributes& attr, const GpuInfo& gpu_info,
              const BHWC* src_shape = nullptr,
              const BHWC* dst_shape = nullptr);

  explicit ConvOneDNN(const OperationDef& definition);

  friend ConvOneDNN CreateConvOneDNN( const GpuInfo& gpu_info,
                                       const OperationDef& definition,
                                       const Convolution2DAttributes& attr,
                                       const BHWC* src_shape,
                                       const BHWC* dst_shape);
  onednn_op_type op_type;
  dnnl::post_ops post_ops;
  dnnl::memory::dims src_dim;
  dnnl::memory::dims dst_dim;
  dnnl::memory::dims weight_dim;
  dnnl::memory::dims stride_dim;
  dnnl::memory::dims paddingl_dim;
  dnnl::memory::dims paddingr_dim;
  dnnl::memory::dims dialate_dim;
  dnnl::memory::dims bias_dim;
  dnnl::convolution_forward::primitive_desc primitive_desc;
  dnnl::primitive_attr pritimive_attr;
  dnnl::convolution_forward conv_primitive;
  std::unordered_map<int, dnnl::memory> arguments_;
}; 

ConvOneDNN CreateConvOneDNN(  const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              const Convolution2DAttributes& attr,
                              const BHWC* src_shape = nullptr,
                              const BHWC* dst_shape = nullptr);

template <DataType T>
void ConvOneDNN::UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights) {
  BufferDescriptor desc;
  desc.element_type = DataType::FLOAT32;
  desc.element_size = 4;
  desc.memory_type =  MemoryType::GLOBAL;
  desc.size = weights.shape.i * weights.shape.o * weights.shape.h * weights.shape.w * sizeof(desc.element_type);
  desc.data.resize(desc.size);
  memcpy(reinterpret_cast<unsigned char*>(desc.data.data()), reinterpret_cast<const unsigned char*>(weights.data.data()), desc.size);
  args_.AddObject("weights", std::make_unique<BufferDescriptor>(std::move(desc)));
}

template <DataType T>
void ConvOneDNN::UploadBias(const tflite::gpu::Tensor<Linear, T>& bias) {
  BufferDescriptor desc;
  desc.element_type = DataType::FLOAT32;
  desc.element_size = 4;
  desc.memory_type = MemoryType::GLOBAL;
  desc.size = bias.shape.v * sizeof(desc.element_type);
  desc.data.resize(desc.size);
  memcpy(reinterpret_cast<unsigned char*>(desc.data.data()), reinterpret_cast<const unsigned char*>(bias.data.data()), desc.size);
  args_.AddObject("biases", std::make_unique<BufferDescriptor>(std::move(desc)));
}



}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_ONEDNN_H_
#endif //TFLITE_ENABLE_ONEDNN
