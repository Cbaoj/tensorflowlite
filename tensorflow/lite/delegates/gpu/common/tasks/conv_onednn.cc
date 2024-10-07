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

#include "tensorflow/lite/delegates/gpu/common/tasks/conv_onednn.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace gpu {

ConvOneDNN::ConvOneDNN(  const OperationDef& definition,
                         const Convolution2DAttributes& attr,
                         const GpuInfo& gpu_info, const BHWC* src_shape, const BHWC* dst_shape)
    : GPUOperation(definition),
      weight_dim({attr.weights.shape.o, attr.weights.shape.i, attr.weights.shape.w, attr.weights.shape.h}),
      stride_dim({attr.strides.w, attr.strides.h}),
      dialate_dim({attr.dilations.w, attr.dilations.h}),
      paddingl_dim({attr.padding.prepended.w, attr.padding.prepended.h}),
      paddingr_dim({attr.padding.appended.w, attr.padding.appended.h}),
      bias_dim({attr.bias.shape.v})
{
  if(src_shape) {
    src_dim = {src_shape->b, src_shape->c, src_shape->h, src_shape->w};  //BCHW
  }
  if(dst_shape) {
    dst_dim = {dst_shape->b, dst_shape->c, dst_shape->h, dst_shape->w};  //BCHW
  }
  AddSrcTensor("src_tensor", definition_.src_tensors[0]);
  AddDstTensor("dst_tensor", definition_.dst_tensors[0]);
  UploadWeights(attr.weights);
  UploadBias(attr.bias);
}

ConvOneDNN::ConvOneDNN(ConvOneDNN&& operation)
  : GPUOperation(std::move(operation)),
    src_dim(operation.src_dim),
    dst_dim(operation.dst_dim),
    weight_dim(operation.weight_dim),
    stride_dim(operation.stride_dim),
    dialate_dim(operation.dialate_dim),
    paddingl_dim(operation.paddingl_dim),
    paddingr_dim(operation.paddingr_dim),
    bias_dim(operation.bias_dim) {
}

ConvOneDNN& ConvOneDNN::operator=(ConvOneDNN&& operation) {
  if (this != &operation) {
    std::swap(weight_dim, operation.weight_dim);
    std::swap(stride_dim, operation.stride_dim);
    std::swap(dialate_dim, operation.dialate_dim);
    std::swap(bias_dim, operation.bias_dim);
    std::swap(paddingl_dim, operation.paddingl_dim);
    std::swap(paddingr_dim, operation.paddingr_dim);
    std::swap(src_dim, operation.src_dim);
    std::swap(dst_dim, operation.dst_dim);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status ConvOneDNN::AddOperation(const GpuInfo& gpu_info, GPUOperation* operation)
{
  if (operation->get_dnn_algorithm() == dnnl::algorithm::eltwise_relu) {
    //need alpha, beta
    post_ops.append_eltwise(dnnl::algorithm::eltwise_relu, 0.1, 0.1);
  }
  return absl::OkStatus();
}

absl::Status ConvOneDNN::AssembleCode(const GpuInfo& gpu_info)
{
  return absl::OkStatus();  
}

absl::Status ConvOneDNN::AddToQueue(cl::CLCommandQueue* queue)
{
  conv_primitive.execute(*queue->get_dnn_stream(), arguments_);
  return absl::OkStatus();  
}

absl::Status ConvOneDNN::prepare_dnn_kernel(const cl::CreationContext& creation_context, cl::CLArguments &cl_args)
{
/* TODO: data_type doesn't have option of i8 yet */
  //auto data_type = GetDefinition().GetPrimaryDataType();
  auto engine = *creation_context.dnn_engine;
  auto data_type = GetPrecision();
  auto dnn_data_type   = data_type == CalculationsPrecision::F16? dnnl::memory::data_type::f16 : dnnl::memory::data_type::f32;  //todo: this is calculation precision, not input/output data type
  dnnl::memory::desc src_desc({src_dim}, dnn_data_type, dnnl::memory::format_tag::nchw /*nhwc*/);
  dnnl::memory::desc dst_desc({dst_dim}, dnn_data_type, dnnl::memory::format_tag::nchw /*nchw*/);

//weights and bias data type are hard defined in struct Convolution2DAttributes {
//  Tensor<OHWI, DataType::FLOAT32> weights;
//  Tensor<Linear, DataType::FLOAT32> bias;
  dnnl::memory::desc weights_desc({weight_dim}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::oihw /*ohwi*/);
  dnnl::memory::desc bias_desc({bias_dim}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::x);

  cl_mem weights_cl_mem = cl_args.get_clmem_buffer("weights_buffer");
  dnnl::memory weights_mem(weights_desc, engine, DNNL_MEMORY_NONE);
  dnnl::ocl_interop::set_mem_object(weights_mem, weights_cl_mem);
  arguments_.insert({DNNL_ARG_WEIGHTS, weights_mem});

  cl_mem bias_cl_mem = cl_args.get_clmem_buffer("biases_buffer");
  dnnl::memory bias_mem(bias_desc, engine, DNNL_MEMORY_NONE);
  dnnl::ocl_interop::set_mem_object(bias_mem, bias_cl_mem);
  arguments_.insert({DNNL_ARG_BIAS, bias_mem});

  cl::Tensor *src_tensor = dynamic_cast<cl::Tensor*>(src_[0]);
  dnnl::memory src_mem(src_desc, engine, DNNL_MEMORY_NONE);
  dnnl::ocl_interop::set_mem_object(src_mem, src_tensor->GetMemoryPtr());
  arguments_.insert({DNNL_ARG_SRC, src_mem});

  cl::Tensor *dst_tensor = dynamic_cast<cl::Tensor*>(dst_[0]);
  dnnl::memory dst_mem(dst_desc, engine, DNNL_MEMORY_NONE);
  dnnl::ocl_interop::set_mem_object(dst_mem, dst_tensor->GetMemoryPtr());
  arguments_.insert({DNNL_ARG_DST, dst_mem});

  dnnl::primitive_attr attr;
  attr.set_post_ops(post_ops);
  try {
    primitive_desc = dnnl::convolution_forward::primitive_desc(engine, dnnl::prop_kind::forward_inference,
                dnnl::algorithm::convolution_direct, src_desc, weights_desc, bias_desc, dst_desc, stride_dim, paddingl_dim, paddingr_dim, attr);
    conv_primitive = dnnl::convolution_forward(primitive_desc);
  } catch (error_t &e) {
    return absl::UnimplementedError("oneDNN error");
  }
  return absl::OkStatus();
}


ConvOneDNN CreateConvOneDNN(  const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              const Convolution2DAttributes& attr,
                              const BHWC* src_shape,
                              const BHWC* dst_shape)
{
  ConvOneDNN conv(definition, attr, gpu_info, src_shape, dst_shape);
  return conv;
}

}  // namespace gpu
}  // namespace tflite
#endif //TFLITE_ENABLE_ONEDNN
