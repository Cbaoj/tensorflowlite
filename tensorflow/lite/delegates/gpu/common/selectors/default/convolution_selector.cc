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

#include <memory>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/model_hints.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_constants.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_generic.h"
#ifdef TFLITE_ENABLE_ONEDNN
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_onednn.h"
#endif
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_metal_simd.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter.h"

namespace tflite {
namespace gpu {
namespace {

std::unique_ptr<GPUOperation> SelectConvolutionAdreno(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def,
    ModelHints hints) {
  if (IsConvConstantsSupported(gpu_info, op_def, attr)) {
    GPUOperation conv = CreateConvConstants(gpu_info, op_def, attr);
    return std::make_unique<GPUOperation>(std::move(conv));
  } else {
    ConvGeneric conv = CreateConvGeneric(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvGeneric>(std::move(conv));
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionNVidia(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def) {
  if (IsConvConstantsSupported(gpu_info, op_def, attr)) {
    GPUOperation conv = CreateConvConstants(gpu_info, op_def, attr);
    return std::make_unique<GPUOperation>(std::move(conv));
  } else {
    ConvGeneric conv = CreateConvGeneric(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvGeneric>(std::move(conv));
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionApple(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def) {
  if (IsConvolutionMetalSimdSupported(gpu_info, op_def, attr) &&
      op_def.precision == CalculationsPrecision::F32 && gpu_info.IsApple() &&
      gpu_info.apple_info.IsSIMDMatMulFp32Perf2x() &&
      IsGoodTaskSizeForAppleConvSimd(dst_shape, gpu_info)) {
    ConvolutionMetalSimd conv =
        CreateConvolutionMetalSimd(op_def, dst_shape, attr, gpu_info);
    return std::make_unique<ConvolutionMetalSimd>(std::move(conv));
  } else {
    ConvGeneric conv = CreateConvGeneric(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvGeneric>(std::move(conv));
  }
}

#ifdef TFLITE_ENABLE_ONEDNN
std::unique_ptr<GPUOperation> SelectConvolutionIntel(
    const Convolution2DAttributes& attr, const BHWC& src_shape, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def) {
  //skip unsupported cases, this is the place where it can switch back to default path by if(true)
  //however there's still one difference, onednn path has ModelHints::kNoWinogradOptimizations set (winograd turns off), see api.cc
  //if(true) {
  if(op_def.src_tensors.size() > 1) {
    ConvGeneric conv = CreateConvGeneric(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvGeneric>(std::move(conv));
  }
  OperationDef conv_temp_def = op_def;
  conv_temp_def.dnn_op_alg = dnnl::algorithm::convolution_direct;
  ConvOneDNN conv = CreateConvOneDNN(gpu_info, conv_temp_def, attr, &src_shape, &dst_shape);
  return std::make_unique<ConvOneDNN>(std::move(conv));
}
#endif
}  // namespace

std::unique_ptr<GPUOperation> SelectConvolution(
    const Convolution2DAttributes& attr, const BHWC& src_shape, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def,
    ModelHints hints) {
  if (gpu_info.IsApple()) {
    return SelectConvolutionApple(attr, dst_shape, gpu_info, op_def);
  } else if (gpu_info.IsAdreno()) {
    return SelectConvolutionAdreno(attr, dst_shape, gpu_info, op_def, hints);
#ifdef TFLITE_ENABLE_ONEDNN
  } else if (gpu_info.IsIntel()) {
    return SelectConvolutionIntel(attr, src_shape, dst_shape, gpu_info, op_def);
#endif
  } else if (gpu_info.IsPowerVR() || gpu_info.IsAMD() || gpu_info.IsIntel() ||
             gpu_info.IsApple() || gpu_info.IsMali()) {
    ConvGeneric conv = CreateConvGeneric(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvGeneric>(std::move(conv));
  } else if (gpu_info.IsNvidia()) {
    return SelectConvolutionNVidia(attr, dst_shape, gpu_info, op_def);
  } else {
    return SelectConvolutionAdreno(attr, dst_shape, gpu_info, op_def, hints);
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionForWinograd(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def,
    ModelHints hints) {
  ConvGeneric conv =
      CreateConvGenericWino4x4To6x6(gpu_info, op_def, attr, &dst_shape);
  return std::make_unique<ConvGeneric>(std::move(conv));
}

std::unique_ptr<GPUOperation> SelectConvolutionWithDynamicWeights(
    const Convolution2DAttributes& attr, const BHWC& weights_shape,
    const BHWC& dst_shape, const GpuInfo& gpu_info,
    const OperationDef& op_def, ModelHints hints,
    WeightsDescription* weights_desc) {
    ConvGeneric conv = CreateConvGenericDynamicWeights(
        gpu_info, op_def, attr, weights_shape, &dst_shape);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvGeneric>(std::move(conv));
}

std::unique_ptr<GPUOperation> SelectConvolutionBatchedMatMul(
    const OHWI& weights_shape, const BHWC& dst_shape, const GpuInfo& gpu_info,
    const OperationDef& op_def, ModelHints hints,
    WeightsDescription* weights_desc) {
    ConvGeneric conv = CreateConvGenericBatchedMatMul(
        gpu_info, op_def, weights_shape, &dst_shape);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvGeneric>(std::move(conv));
}

std::unique_ptr<GPUOperation> SelectConverterToConvWeights(
    const WeightsDescription& weights_desc, const OperationDef& op_def,
    ModelHints hints, Layout input_layout) {
  ConverterToConvWeights converter =
      ConverterToConvWeights(op_def, weights_desc, input_layout);
  return std::make_unique<ConverterToConvWeights>(std::move(converter));
}

}  // namespace gpu
}  // namespace tflite
