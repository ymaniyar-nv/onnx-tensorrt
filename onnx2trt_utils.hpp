/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include "ShapedWeights.hpp"
#include "trt_utils.hpp"
#include "OnnxAttrs.hpp"

#include <onnx/onnx.pb.h>
#include <onnx/onnxifi.h>
#include <NvInfer.h>

#include <iostream>
#include <sstream>  // For std::stringstream
#include <cstring>  // For std::memcpy
#include <numeric>

#define LOG_VERBOSE(msg)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        std::stringstream ss{};                                                                                        \
        ss << __FILE__ << ":" << __LINE__ << ": " << msg;                                                              \
        ctx->logger().log(nvinfer1::ILogger::Severity::kVERBOSE, ss.str().c_str());                                    \
    } while (0)

inline nvinfer1::Dims makeDims(int nbDims, int val)
{
    nvinfer1::Dims dims;
    dims.nbDims = nbDims;
    std::fill_n(dims.d, nbDims, val);
    return dims;
}

class CeilingPoolDim:public nvinfer1::IOutputDimensionsFormula{
public:
    nvinfer1::DimsHW compute(nvinfer1::DimsHW inputDims, nvinfer1::DimsHW kernelSize,
        nvinfer1::DimsHW stride, nvinfer1::DimsHW padding, nvinfer1::DimsHW dilation, const char* layerName) const
    {
        nvinfer1::DimsHW outputDims;
        for (int dimension = 0; dimension < inputDims.nbDims; ++dimension)
        {
            outputDims.d[dimension] = static_cast<int>(ceil((inputDims.d[dimension] + padding.d[dimension] * 2.0 - kernelSize.d[dimension]) / stride.d[dimension] + 1.0));
        }
        return outputDims;
    }
};

inline int64_t volume(const nvinfer1::Dims& dims)
{
    std::for_each(dims.d, dims.d + dims.nbDims, [](int d){ assert(d >= 0 && "volume makes no sense for dynamic shapes");});
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>{});
}

inline std::ostream& operator<<(std::ostream& stream, nvinfer1::Dims const& shape) {
    stream << "(";
    for (int i = 0; i < shape.nbDims; ++i)
    {
        stream << (i ? ", " : "") << shape.d[i];
    }
    return stream << ")";
}

inline std::ostream& operator<<(std::ostream& stream, nvinfer1::DataType const& dtype) {
  switch( dtype ) {
  case nvinfer1::DataType::kFLOAT: return stream << "float32";
  case nvinfer1::DataType::kHALF:  return stream << "float16";
  case nvinfer1::DataType::kINT8:  return stream << "int8";
  case nvinfer1::DataType::kINT32: return stream << "int32";
  case nvinfer1::DataType::kBOOL: return stream << "bool";
  default: throw std::runtime_error("Unknown dtype");
  }
}

// TODO: Remove this when finished debugging
inline std::ostream& operator<<(std::ostream& stream, nvinfer1::Permutation const& perm) {
    int ndims = nvinfer1::Dims::MAX_DIMS;
    stream << "(" << perm.order[0];
    for (int i = 1; i < ndims; ++i)
    {
        stream << ", " << perm.order[i];
    }
    stream << ")";
    return stream;
}
/*
// TODO: Remove this when finished debugging
inline std::ostream& operator<<(std::ostream& stream, google::protobuf::Message const& message) {
  stream << print_onnx_to_string(message);
  return stream;
}
*/
namespace onnx2trt {

inline int getDtypeSize(int32_t onnxDtype) {
    switch (onnxDtype)
    {
        case ::ONNX_NAMESPACE::TensorProto::FLOAT16:    return 2;
        case ::ONNX_NAMESPACE::TensorProto::FLOAT:      return 4;
        case ::ONNX_NAMESPACE::TensorProto::DOUBLE:     return 8;
        case ::ONNX_NAMESPACE::TensorProto::COMPLEX64:  return 8;
        case ::ONNX_NAMESPACE::TensorProto::COMPLEX128: return 16;
        case ::ONNX_NAMESPACE::TensorProto::UINT8:      return 1;
        case ::ONNX_NAMESPACE::TensorProto::INT8:       return 1;
        case ::ONNX_NAMESPACE::TensorProto::UINT16:     return 2;
        case ::ONNX_NAMESPACE::TensorProto::INT16:      return 2;
        case ::ONNX_NAMESPACE::TensorProto::UINT32:     return 4;
        // Booleans are stored in int32 tensors in ONNX
        case ::ONNX_NAMESPACE::TensorProto::BOOL:       return 4;
        case ::ONNX_NAMESPACE::TensorProto::INT32:      return 4;
        case ::ONNX_NAMESPACE::TensorProto::UINT64:     return 8;
        case ::ONNX_NAMESPACE::TensorProto::INT64:      return 8;
        default: return -1;
    }
}

inline const char* getDtypeName(int32_t onnxDtype) {
    switch (onnxDtype)
    {
        case ::ONNX_NAMESPACE::TensorProto::FLOAT:      return "FLOAT";
        case ::ONNX_NAMESPACE::TensorProto::UINT8:      return "UINT8";
        case ::ONNX_NAMESPACE::TensorProto::INT8:       return "INT8";
        case ::ONNX_NAMESPACE::TensorProto::UINT16:     return "UINT16";
        case ::ONNX_NAMESPACE::TensorProto::INT16:      return "INT16";
        case ::ONNX_NAMESPACE::TensorProto::INT32:      return "INT32";
        case ::ONNX_NAMESPACE::TensorProto::INT64:      return "INT64";
        case ::ONNX_NAMESPACE::TensorProto::STRING:     return "STRING";
        case ::ONNX_NAMESPACE::TensorProto::BOOL:       return "BOOL";
        case ::ONNX_NAMESPACE::TensorProto::FLOAT16:    return "FLOAT16";
        case ::ONNX_NAMESPACE::TensorProto::DOUBLE:     return "DOUBLE";
        case ::ONNX_NAMESPACE::TensorProto::UINT32:     return "UINT32";
        case ::ONNX_NAMESPACE::TensorProto::UINT64:     return "UINT64";
        case ::ONNX_NAMESPACE::TensorProto::COMPLEX64:  return "COMPLEX64";
        case ::ONNX_NAMESPACE::TensorProto::COMPLEX128: return "COMPLEX128";
        default: return "<UNKNOWN>";
    }
}

inline nvinfer1::IPluginV2* importPluginFromRegistry(IImporterContext* ctx, const std::string& pluginName,
    const std::string& pluginVersion, const std::string& nodeName, const std::vector<nvinfer1::PluginField>& pluginFields)
{
    const auto mPluginRegistry = getPluginRegistry();
    const auto pluginCreator = mPluginRegistry->getPluginCreator(pluginName.c_str(), pluginVersion.c_str(), "");

    if (!pluginCreator)
    {
      return nullptr;
    }

    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = pluginFields.size();
    fc.fields = pluginFields.data();

    return pluginCreator->createPlugin(nodeName.c_str(), &fc);
}

inline nvinfer1::ITensor* reshape_tensor(IImporterContext* ctx,
               nvinfer1::ITensor& tensor,
               nvinfer1::Dims shape) {
  if( shape == tensor.getDimensions() ) {
    return &tensor;
  }
  nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
  if( !layer ) {
    return nullptr;
  }
  layer->setReshapeDimensions(shape);
  return layer->getOutput(0);
}

inline void broadcast_tensors(IImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2)
{
    if (t1->getDimensions().nbDims == t2->getDimensions().nbDims)
    {
        return;
    }
    nvinfer1::ITensor* largeTensor;
    nvinfer1::ITensor* smallTensor;
    if (t1->getDimensions().nbDims > t2->getDimensions().nbDims)
    {
        largeTensor = t1;
        smallTensor = t2;
    }
    else
    {
        largeTensor = t2;
        smallTensor = t1;
    }

    nvinfer1::Dims largeDims = largeTensor->getDimensions();
    nvinfer1::Dims smallDims = smallTensor->getDimensions();
    nvinfer1::Dims newDims = expand_dims(smallDims, largeDims.nbDims);
    LOG_VERBOSE("Broadcasting " << smallDims << " across " << largeDims << ". New shape: " << newDims);

    t1 == smallTensor ? t1 = reshape_tensor(ctx, *t1, newDims) : t2 = reshape_tensor(ctx, *t2, newDims);
}

inline bool convert_dtype(int32_t onnx_dtype,
                          nvinfer1::DataType* trt_dtype) {
  switch (onnx_dtype)
  {
    case ::ONNX_NAMESPACE::TensorProto::FLOAT:
        *trt_dtype = nvinfer1::DataType::kFLOAT;
        break;
    case ::ONNX_NAMESPACE::TensorProto::INT8:
        *trt_dtype = nvinfer1::DataType::kINT8;
        break;
    case ::ONNX_NAMESPACE::TensorProto::FLOAT16:
        *trt_dtype = nvinfer1::DataType::kHALF;
        break;
    case ::ONNX_NAMESPACE::TensorProto::BOOL:
    case ::ONNX_NAMESPACE::TensorProto::INT32:
        *trt_dtype = nvinfer1::DataType::kINT32;
        break;
    // See ShapedWeights.cpp for sanity check if all values can be safetly downcasted to INT32
    case ::ONNX_NAMESPACE::TensorProto::INT64:
        *trt_dtype = nvinfer1::DataType::kINT32;
        break;
    default:
        std::cerr << "Unsupported ONNX data type: " << getDtypeName(onnx_dtype) << " (" << std::to_string(onnx_dtype) << ")" << std::endl;
        return false;
    }
    return true;
}

template<typename OnnxDims>
inline nvinfer1::Dims convert_dims(OnnxDims const& onnx_dims) {
    std::vector<int> onnx_dims_vector;
    for (const auto& onnx_dim : onnx_dims)
    {
        onnx_dims_vector.push_back((onnx_dim.dim_param() == "" ? onnx_dim.dim_value() : -1));
    }
    nvinfer1::Dims trt_dims;
    trt_dims.nbDims = onnx_dims_vector.size();
    assert(trt_dims.nbDims <= nvinfer1::Dims::MAX_DIMS);
    std::copy(onnx_dims_vector.begin(), onnx_dims_vector.end(), trt_dims.d);
    return trt_dims;
}

inline bool convert_weight_descriptor(onnxTensorDescriptorV1 const &desc,
                                      onnx2trt::ShapedWeights *weights) {
    nvinfer1::Dims shape;
    shape.nbDims = desc.dimensions;
    // Special case for scalars
    if( shape.nbDims == 0 )
    {
        shape.nbDims = 1;
        shape.d[0] = 1;
    }
    else
    {
        std::copy(desc.shape, desc.shape + desc.dimensions, shape.d);
    }

    size_t element_count = 1;
    for (int i = 0; i < shape.nbDims; ++i)
    {
        element_count *= shape.d[i];
    }

    void* data_ptr;
    size_t nbytes;
    int32_t dtype;
    data_ptr = (void*)(desc. buffer);
    if (desc.dataType == ONNXIFI_DATATYPE_FLOAT32)
    {
        dtype = ::ONNX_NAMESPACE::TensorProto::FLOAT;
        nbytes = element_count * sizeof(float);
    }
    else if (desc.dataType == ONNXIFI_DATATYPE_FLOAT16)
    {
        dtype = ::ONNX_NAMESPACE::TensorProto::FLOAT16;
        nbytes = element_count * sizeof(float) / 2;
    }
    else if (desc.dataType == ONNXIFI_DATATYPE_INT32)
    {
        dtype = ::ONNX_NAMESPACE::TensorProto::INT32;
        nbytes = element_count * sizeof(int32_t);
    }
    else if (desc.dataType == ONNXIFI_DATATYPE_INT64)
    {
        dtype = ::ONNX_NAMESPACE::TensorProto::INT64;
        nbytes = element_count * sizeof(int64_t);
    } else
    {
        // Unsupported format
        return false;
    }

    onnx2trt::ShapedWeights trt_weights(dtype, data_ptr, shape);
    (void)nbytes;
    assert(trt_weights.size_bytes() == nbytes);
    *weights = trt_weights;
    return true;
}

inline bool convert_onnx_weights(::ONNX_NAMESPACE::TensorProto const& onnx_tensor, onnx2trt::ShapedWeights* weights)
{
    nvinfer1::Dims shape;
    shape.nbDims = onnx_tensor.dims().size();
    std::copy(onnx_tensor.dims().begin(), onnx_tensor.dims().end(), shape.d);

    auto const& onnx_tensor_type = onnx_tensor.data_type();

    void* data_ptr; // TODO: See if can make const*
    size_t nbytes;
    if( onnx_tensor.raw_data().size() > 0 )
    {
        data_ptr = (void*)onnx_tensor.raw_data().data();
        nbytes = onnx_tensor.raw_data().size();
    }
    else if( onnx_tensor.float_data().size() > 0 )
    {
        assert(onnx_tensor_type == ::ONNX_NAMESPACE::TensorProto::FLOAT);
        data_ptr = (void*)onnx_tensor.float_data().data();
        nbytes = onnx_tensor.float_data().size() * sizeof(float);
    }
    else if (onnx_tensor.int32_data().size() > 0)
    {
        assert(getDtypeSize(onnx_tensor_type) == 4);
        data_ptr = (void*)onnx_tensor.int32_data().data();
        nbytes = onnx_tensor.int32_data().size() * sizeof(int32_t);
    }
    else if( onnx_tensor.int64_data().size() > 0 )
    {
        assert(onnx_tensor_type == ::ONNX_NAMESPACE::TensorProto::INT64);
        data_ptr = (void*)onnx_tensor.int64_data().data();
        nbytes = onnx_tensor.int64_data().size() * sizeof(int64_t);
    }
    else
    {
        // Unsupported ONNX tensor format!
        return false;
    }

    onnx2trt::ShapedWeights trt_weights(onnx_tensor_type, data_ptr, shape);
    (void)nbytes;
    assert(trt_weights.size_bytes() == nbytes);
    *weights = trt_weights;
    return true;
}

// Returns the input if it is already a tensor. If it is of type ShapedWeights, adds a new
// constant layer to the TRT network and returns its output.
inline nvinfer1::ITensor& convertToTensor(TensorOrWeights& input, IImporterContext* ctx)
{
    if (input.is_tensor())
    {
        return input.tensor();
    }
    else
    {
        // Handle non-tensor indices input by adding a new constant layer to the network.
        const ShapedWeights& weights = input.weights();
        return *(ctx->network()->addConstant(weights.shape, weights)->getOutput(0));
    }

}

inline int div_ceil(int n, int d) {
  return (n - 1) / d + 1;
}

// Convert an ONNX axis into a TRT axis
inline Status convert_axis(int& axis, int nbDims)
{
  // Support negative indexing
  if (axis < 0)
  {
    axis += nbDims;
  }
  ASSERT(axis >= 0 && axis < nbDims, ErrorCode::kUNSUPPORTED_NODE);
  return Status::success();
}

inline int get_conv_output_size(int input_size, int filter_size,
                                int stride, int dilation_rate,
                                int total_padding) {
  // This is based on the CUDNN formula
  int effective_input_size  = input_size + total_padding;
  int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  return div_ceil(effective_input_size - (effective_filter_size - 1), stride);
}

// Adds a constant scalar to the network in the form of a constant layer.
// shape can be specified for broadcasting purposes, but it's volume should be 1.
template <typename ScalarType>
inline nvinfer1::IConstantLayer* addConstantScalar(IImporterContext* ctx, ScalarType scalar, ShapedWeights::DataType type, nvinfer1::Dims shape=nvinfer1::Dims{0})
{
    assert(volume(shape) == 1 && "Cannot add constant scalar with a shape that has volume > 1");
    ShapedWeights scalarWeights = ctx->createTempWeights(type, shape);
    static_cast<ScalarType*>(scalarWeights.values)[0] = scalar;
    return ctx->network()->addConstant(scalarWeights.shape, scalarWeights);
}

template <typename ScalarType>
inline nvinfer1::IConstantLayer* addConstant(IImporterContext* ctx, const std::vector<ScalarType>& values, ShapedWeights::DataType type, nvinfer1::Dims shape)
{
    assert(volume(shape) == static_cast<int64_t>(values.size()) && "Shape does not match number of values provided");
    assert(sizeof(ScalarType) == getDtypeSize(type) && "ONNX dtype does not have the same size as the value type");
    ShapedWeights weights = ctx->createTempWeights(type, shape);
    std::memcpy(weights.values, values.data(), values.size() * sizeof(ScalarType));
    return ctx->network()->addConstant(weights.shape, weights);
}

// Get the length of the specified axis. e.g. for a tensor of shape (4, 5, 6) with axis=0, this would return 4.
inline nvinfer1::ITensor* getAxisLength(IImporterContext* ctx, nvinfer1::ITensor* inpTensor, int axis, nvinfer1::Dims shape=nvinfer1::Dims{0})
{
    // fast path for static dims
    auto dims = inpTensor->getDimensions();
    int d = dims.d[axis];
    if (d >= 0)
    {
        return addConstantScalar(ctx, d, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, shape)->getOutput(0);
    }
    else
    {
        auto& inpShape = *ctx->network()->addShape(*inpTensor)->getOutput(0);
        auto& axisValue = *addConstantScalar(ctx, axis, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, shape)->getOutput(0);
        return ctx->network()->addGather(inpShape, axisValue, 0)->getOutput(0);
    }
}

// If the input tensor has exactly one element, creates a new scalar tensor from it.
inline nvinfer1::ITensor* convertToScalar(IImporterContext* ctx, nvinfer1::ITensor* inpTensor)
{
    const auto tensorVolume = volume(inpTensor->getDimensions());
    if (tensorVolume != 1)
    {
        LOG_VERBOSE("Cannot convert tensor to scalar. Note: Tensor dimensions were: " << inpTensor->getDimensions() << ", with volume: " << tensorVolume);
        return nullptr;
    }
    nvinfer1::IShuffleLayer* reshape = ctx->network()->addShuffle(*inpTensor);
    reshape->setReshapeDimensions(nvinfer1::Dims{0});
    return reshape->getOutput(0);
}

// shape must be a shape tensor
inline nvinfer1::ITensor* constantOfShape(IImporterContext* ctx, nvinfer1::ITensor* constant, nvinfer1::ITensor* shape)
{
    int rank = shape->getDimensions().d[0];

    std::vector<int> starts(rank);
    std::fill(starts.begin(), starts.end(), 0);

    nvinfer1::Dims strides{rank};
    std::fill(strides.d, strides.d + strides.nbDims, 0);

    // Slice will not work if constant does not have the same rank as start/size/strides.
    nvinfer1::Dims unsqueezeDims{rank};
    std::fill(unsqueezeDims.d, unsqueezeDims.d + unsqueezeDims.nbDims, 1);
    nvinfer1::IShuffleLayer* unsqueeze = ctx->network()->addShuffle(*constant);
    unsqueeze->setReshapeDimensions(unsqueezeDims);
    constant = unsqueeze->getOutput(0);

    nvinfer1::ISliceLayer* broadcast = ctx->network()->addSlice(*constant, nvinfer1::Dims{}, nvinfer1::Dims{}, strides);
    broadcast->setInput(1, *addConstant(ctx, starts, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, rank})->getOutput(0));
    broadcast->setInput(2, *shape);
    return broadcast->getOutput(0);
}

void get_kernel_params(::ONNX_NAMESPACE::NodeProto const& onnx_node,
                       nvinfer1::Dims* kernel_size,
                       nvinfer1::Dims* strides,
                       nvinfer1::Dims* beg_padding,
                       nvinfer1::Dims* end_padding,
                       nvinfer1::PaddingMode& paddingMode,
                       bool& count_exclude_padding,
                       nvinfer1::Dims* dilations=nullptr,
                       nvinfer1::Dims* output_padding=nullptr);

inline nvinfer1::ScaleMode get_scale_mode(nvinfer1::Dims const& weights_shape, nvinfer1::Dims const& tensor_shape)
{
  if (weights_shape.nbDims == 1)
  {
    if (weights_shape.d[0] == 1)
    {
      return nvinfer1::ScaleMode::kUNIFORM;
    }
    // Check for channel wide scale - assume tensor shape is NCHW.
    else if (weights_shape.d[0] == tensor_shape.d[1])
    {
      return nvinfer1::ScaleMode::kCHANNEL;
    }
  }
  return nvinfer1::ScaleMode::kELEMENTWISE;
}

inline nvinfer1::ITensor& makeShapeTensor(IImporterContext* ctx, nvinfer1::Dims dims)
{
    auto tempWeights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto_DataType_INT32, makeDims(1, dims.nbDims));
    for(int i = 0; i < dims.nbDims; i++)
    {
        static_cast<int32_t*>(tempWeights.values)[i] = dims.d[i];
    }
    auto valueWeights = TensorOrWeights{tempWeights};
    return convertToTensor(valueWeights, ctx);
}

// ONNX default alpha value for the specified activation type
inline float getActivationDefaultAlpha(nvinfer1::ActivationType type)
{
    switch (type) {
        case nvinfer1::ActivationType::kRELU:             return 0.f;
        case nvinfer1::ActivationType::kSIGMOID:          return 0.f;
        case nvinfer1::ActivationType::kTANH:             return 0.f;
        case nvinfer1::ActivationType::kLEAKY_RELU:       return 0.01f;
        case nvinfer1::ActivationType::kELU:              return 1.0f;
        case nvinfer1::ActivationType::kSELU:             return 1.67326319217681884765625f;
        case nvinfer1::ActivationType::kSOFTSIGN:         return 0.f;
        case nvinfer1::ActivationType::kSOFTPLUS:         return 0.f;
        case nvinfer1::ActivationType::kCLIP:             return 0.f;
        case nvinfer1::ActivationType::kHARD_SIGMOID:     return 0.2f;
        case nvinfer1::ActivationType::kSCALED_TANH:      return 1.0f;
        case nvinfer1::ActivationType::kTHRESHOLDED_RELU: return 1.0f;
    }
    throw std::runtime_error{"Unrecognized activation type"};
}

// ONNX default beta value for the specified activation type
inline float getActivationDefaultBeta(nvinfer1::ActivationType type)
{
    switch (type) {
        case nvinfer1::ActivationType::kRELU:             return 0.f;
        case nvinfer1::ActivationType::kSIGMOID:          return 0.f;
        case nvinfer1::ActivationType::kTANH:             return 0.f;
        case nvinfer1::ActivationType::kLEAKY_RELU:       return 0.f;
        case nvinfer1::ActivationType::kELU:              return 0.f;
        case nvinfer1::ActivationType::kSELU:             return 1.05070102214813232421875f;
        case nvinfer1::ActivationType::kSOFTSIGN:         return 0.f;
        case nvinfer1::ActivationType::kSOFTPLUS:         return 0.f;
        case nvinfer1::ActivationType::kCLIP:             return 0.f;
        case nvinfer1::ActivationType::kHARD_SIGMOID:     return 0.5f;
        case nvinfer1::ActivationType::kSCALED_TANH:      return 1.0f;
        case nvinfer1::ActivationType::kTHRESHOLDED_RELU: return 0.f;
    }
    throw std::runtime_error{"Unrecognized activation type"};
}

} // namespace onnx2trt
