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

#include "onnx2trt_utils.hpp"
#include "OnnxAttrs.hpp"

namespace onnx2trt {

NodeImportResult activationHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ActivationType op, float* alpha, float* beta)
{
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    nvinfer1::IActivationLayer* layer = ctx->network()->addActivation(input, op);
    if (alpha)
    {
        layer->setAlpha(*alpha);
    }
    if (beta)
    {
        layer->setBeta(*beta);
    }

    return {{layer->getOutput(0)}};
}

NodeImportResult argMinMaxHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::TopKOperation op)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    ASSERT(tensor.getType() != nvinfer1::DataType::kINT32, ErrorCode::kUNSUPPORTED_NODE);
    // Get attributes.
    OnnxAttrs attrs(node, ctx);
    int keepdims = attrs.get("keepdims", 1);
    int axis = attrs.get("axis", 0);

    // Insert a TopK layer with k set to 1.
    int nbDims = tensor.getDimensions().nbDims;
    TRT_CHECK(convertAxis(axis, nbDims));

    uint32_t axisMask = 1 << axis;
    nvinfer1::ITopKLayer* layer = ctx->network()->addTopK(tensor, op, 1, axisMask);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    // We don't care about the TopK values, just the indices.
    nvinfer1::ITensor* indices = layer->getOutput(1);
    indices->setType(nvinfer1::DataType::kINT32);
    if (keepdims)
    {
        // The default behavior of the TopK layer is to keepdims.
        return {{indices}};
    }
    else
    {
        // Otherwise, we need to squeeze the axis dimension - we achieve this by reshaping.
        // The new dimensions are just the old dimensions with all values after axis shifted over.
        nvinfer1::Dims reshapeDims = indices->getDimensions();
        --reshapeDims.nbDims;
        // The axis dimension should be reduced to size 1 after performing the reduction.
        ASSERT(reshapeDims.d[axis] == 1, ErrorCode::kINVALID_VALUE);
        for (int i = axis; i < reshapeDims.nbDims; ++i)
        {
            reshapeDims.d[i] = reshapeDims.d[i + 1];
        }
        nvinfer1::IShuffleLayer* squeezeLayer = ctx->network()->addShuffle(*indices);
        squeezeLayer->setReshapeDimensions(reshapeDims);
        return {{squeezeLayer->getOutput(0)}};
    }
}

void broadcastTensors(IImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2)
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

    t1 == smallTensor ? t1 = reshapeTensor(ctx, *t1, newDims) : t2 = reshapeTensor(ctx, *t2, newDims);
}

nvinfer1::ITensor* constantOfShape(IImporterContext* ctx, nvinfer1::ITensor* constant, nvinfer1::ITensor* shape)
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

Status convertAxis(int& axis, int nbDims)
{
  // Support negative indexing
  if (axis < 0)
  {
    axis += nbDims;
  }
  ASSERT(axis >= 0 && axis < nbDims, ErrorCode::kUNSUPPORTED_NODE);
  return Status::success();
}

bool convertDtype(int32_t onnx_dtype, nvinfer1::DataType* trt_dtype) {
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
    // See convertOnnxWeights for sanity check if all values can be safetly downcasted to INT32
    case ::ONNX_NAMESPACE::TensorProto::INT64:
        *trt_dtype = nvinfer1::DataType::kINT32;
        break;
    default:
        std::cerr << "Unsupported ONNX data type: " << getDtypeName(onnx_dtype) << " (" << std::to_string(onnx_dtype) << ")" << std::endl;
        return false;
    }
    return true;
}

int32_t* convertINT64(const int64_t* weightValues, nvinfer1::Dims shape, IImporterContext* ctx)
{
    std::cout << "WARNING: Your ONNX model has been generated with INT64 weights, "
         << "while TensorRT does not natively support INT64. "
         << "Attempting to cast down to INT32." << std::endl;

    const size_t nbWeights = volume(shape);
    int32_t* int32Weights{reinterpret_cast<int32_t*>(ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::INT32, shape).values)};

    for (size_t i = 0; i < nbWeights; i++)
    {
        if (weightValues[i] > INT32_MAX || weightValues[i] < INT32_MIN)
        {
            std::cerr << "ERROR: Weights cannot be cast down to INT32. Weight at index " << i << ": " << weightValues[i] << " is out of range" << std::endl;
            return nullptr;
        }
        else
        {
            int32Weights[i] = static_cast<int32_t>(weightValues[i]);
        }
    }
    return int32Weights;
}

bool convertOnnxWeights(const ::ONNX_NAMESPACE::TensorProto& onnxTensor, onnx2trt::ShapedWeights* weights, IImporterContext* ctx)
{
    nvinfer1::Dims shape;
    shape.nbDims = onnxTensor.dims().size();
    std::copy(onnxTensor.dims().begin(), onnxTensor.dims().end(), shape.d);

    auto onnxDtype = onnxTensor.data_type();

    void* dataPtr{nullptr}; // TODO: See if can make const*
    size_t nbytes{0};
    if (onnxDtype == ::ONNX_NAMESPACE::TensorProto::INT64)
    {
        if (onnxTensor.raw_data().size() > 0)
        {
            dataPtr = convertINT64(reinterpret_cast<const int64_t*>(onnxTensor.raw_data().data()), shape, ctx);
            nbytes = onnxTensor.raw_data().size() / 2;
        }
        else if (onnxTensor.int64_data().size() > 0)
        {
            dataPtr = convertINT64(onnxTensor.int64_data().data(), shape, ctx);
            nbytes = onnxTensor.int64_data().size() * sizeof(int32_t);
        }
        onnxDtype = ::ONNX_NAMESPACE::TensorProto::INT32;
    }
    else if (onnxTensor.raw_data().size() > 0)
    {
        dataPtr = (void*)(onnxTensor.raw_data().data());
        nbytes = onnxTensor.raw_data().size();
    }
    else if (onnxTensor.float_data().size() > 0)
    {
        assert(onnxDtype == ::ONNX_NAMESPACE::TensorProto::FLOAT);
        dataPtr = (void*)(onnxTensor.float_data().data());
        nbytes = onnxTensor.float_data().size() * sizeof(float);
    }
    else if (onnxTensor.int32_data().size() > 0)
    {
        assert(getDtypeSize(onnxDtype) == 4);
        dataPtr = (void*)(onnxTensor.int32_data().data());
        nbytes = onnxTensor.int32_data().size() * sizeof(int32_t);
    }
    else
    {
        // Unsupported ONNX tensor format!
        return false;
    }

    onnx2trt::ShapedWeights trt_weights(onnxDtype, dataPtr, shape);
    (void)nbytes;
    assert(trt_weights.size_bytes() == nbytes);
    *weights = trt_weights;
    return true;
}

nvinfer1::ITensor* convertToScalar(IImporterContext* ctx, nvinfer1::ITensor* inpTensor)
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


nvinfer1::ITensor& convertToTensor(TensorOrWeights& input, IImporterContext* ctx)
{
    if (input.is_tensor())
    {
        return input.tensor();
    }
    else
    {
        // Handle non-tensor indices input by adding a new constant layer to the network.
        ShapedWeights& weights = input.weights();
        return *(ctx->network()->addConstant(weights.shape, weights)->getOutput(0));
    }

}

bool convertWeightDescriptor(onnxTensorDescriptorV1 const &desc, onnx2trt::ShapedWeights *weights, IImporterContext* ctx)
{
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

    void* dataPtr;
    size_t nbytes;
    int32_t dtype;
    dataPtr = (void*)(desc. buffer);
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
        dataPtr = convertINT64(reinterpret_cast<const int64_t*>(desc.buffer), shape, ctx);
        dtype = ::ONNX_NAMESPACE::TensorProto::INT32;
        nbytes = element_count * sizeof(int32_t);
    } else
    {
        // Unsupported format
        return false;
    }

    onnx2trt::ShapedWeights trt_weights(dtype, dataPtr, shape);
    (void)nbytes;
    assert(trt_weights.size_bytes() == nbytes);
    *weights = trt_weights;
    return true;
}

int divCeil(int n, int d)
{
  return (n - 1) / d + 1;
}

NodeImportResult elementwiseHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ElementWiseOperation binary_op)
{
    ASSERT(!inputs.empty(), ErrorCode::kINVALID_NODE);

    std::vector<nvinfer1::ITensor*> input_tensors;
    int ndim_max = -1;

    // Find maximum number of input dimensions
    for (auto input : inputs)
    {
        ndim_max = std::max(ndim_max, input.shape().nbDims);
    }

    // Convert inputs to tensors and expand their dimensions to ndim_max if necessary
    for (auto input : inputs)
    {
        nvinfer1::ITensor* tensor_ptr = &convertToTensor(input, ctx);
        if (tensor_ptr->getDimensions().nbDims != ndim_max)
        {
            nvinfer1::Dims new_dims = expand_dims(tensor_ptr->getDimensions(), ndim_max);
            tensor_ptr = reshapeTensor(ctx, *tensor_ptr, new_dims);
        }
        ASSERT(tensor_ptr->getDimensions().nbDims == ndim_max, ErrorCode::kUNSUPPORTED_NODE);
        input_tensors.push_back(tensor_ptr);
    }
    // Use the first tensor input as the base for the elementwise operation
    nvinfer1::ITensor* combined = input_tensors.at(0);
    if (input_tensors.size() == 1)
    {
        // Note: Single input must be wrapped in identity to avoid messing up network outputs
        return {{identity(ctx, combined)}};
    }
    for (size_t i = 1; i < input_tensors.size(); ++i)
    {
        nvinfer1::ITensor* tensor = input_tensors.at(i);
        ASSERT(tensor->getDimensions().nbDims == combined->getDimensions().nbDims, ErrorCode::kUNSUPPORTED_NODE);
        auto* layer = ctx->network()->addElementWise(*combined, *tensor, binary_op);
        ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
        combined = layer->getOutput(0);
    }
    return {{combined}};
}

nvinfer1::ITensor* flattenTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, int axis)
{
    nvinfer1::Dims shape = tensor.getDimensions();
    nvinfer1::Dims new_shape = shape;
    for (int i = axis + 1; i < shape.nbDims; ++i)
    {
        new_shape.d[axis] *= shape.d[i];
        new_shape.d[i] = 1;
    }
    return reshapeTensor(ctx, tensor, new_shape);
}

// Helper function to generate padding values for convTranspose
void generatePadding(nvinfer1::Dims input_dims, nvinfer1::Dims output_shape, nvinfer1::Dims kernel_size,
    nvinfer1::Dims strides, nvinfer1::Dims dilations, const int nbSpatialDims, nvinfer1::Dims& beg_padding,
    nvinfer1::Dims& end_padding, nvinfer1::Dims& output_padding, nvinfer1::PaddingMode paddingMode)
{
    // When auto_pad == NONSET or VALID, input padding is explict
    // explicit output shape may require output padding
    if (paddingMode == nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN)
    {
        nvinfer1::Dims expected_output_shape;
        for (int i = 0; i < nbSpatialDims; i++)
        {
            expected_output_shape.d[i] = (input_dims.d[2 + i] - 1) * strides.d[i]
                + (kernel_size.d[i] - 1) * dilations.d[i] + 1 - beg_padding.d[i] - end_padding.d[i];
            output_padding.d[i] = output_shape.d[i] - expected_output_shape.d[i];
        }
    }
    else
    {
        // When auto_pad == SAME_UPPER or SAME_LOWER, output padding is explict
        // explicit output shape may require input padding
        nvinfer1::Dims total_padding = makeDims(nbSpatialDims, 0);
        for (int i = 0; i < nbSpatialDims; i++)
        {
            total_padding.d[i] = (input_dims.d[2 + i] - 1) * strides.d[i] + (kernel_size.d[i] - 1) * dilations.d[i] + 1
                + output_padding.d[i] - output_shape.d[i];
            if (paddingMode == nvinfer1::PaddingMode::kSAME_UPPER)
            {
                beg_padding.d[i] = total_padding.d[i] - (total_padding.d[i] / 2);
                end_padding.d[i] = total_padding.d[i] / 2;
            }
            else
            {
                beg_padding.d[i] = total_padding.d[i] / 2;
                end_padding.d[i] = total_padding.d[i] - (total_padding.d[i] / 2);
            }
        }
    }
}

float getActivationDefaultAlpha(nvinfer1::ActivationType type)
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

float getActivationDefaultBeta(nvinfer1::ActivationType type)
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

nvinfer1::ITensor* getAxisLength(IImporterContext* ctx, nvinfer1::ITensor* inpTensor, int axis, nvinfer1::Dims shape)
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

int getConvOutputSize(int input_size, int filter_size,
                                int stride, int dilation_rate,
                                int total_padding);

const char* getDtypeName(int32_t onnxDtype) {
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

int getDtypeSize(int32_t onnxDtype) {
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

void getKernelParams(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& onnx_node,
                       nvinfer1::Dims* kernel_size,
                       nvinfer1::Dims* strides,
                       nvinfer1::Dims* beg_padding,
                       nvinfer1::Dims* end_padding,
                       nvinfer1::PaddingMode& paddingMode,
                       bool & count_exclude_padding,
                       nvinfer1::Dims* dilations,
                       nvinfer1::Dims* output_padding) {
  const int nbSpatialDims = kernel_size->nbDims;
  OnnxAttrs attrs(onnx_node, ctx);
  if( attrs.count("kernel_shape") ) {
    auto const* onnx_kernel_size = attrs.at("kernel_shape");
    setAttr(kernel_size, onnx_kernel_size, nbSpatialDims, 1);
  }
  if( attrs.count("strides") ) {
    auto const* onnx_strides = attrs.at("strides");
    setAttr(strides, onnx_strides, nbSpatialDims, 1);
  }
  if( dilations && attrs.count("dilations") ) {
    auto const* onnx_dilations = attrs.at("dilations");
    setAttr(dilations, onnx_dilations, nbSpatialDims, 1);
  }
  if( attrs.count("count_include_pad")){
    auto const* include_pad = attrs.at("count_include_pad");
    int val = include_pad->i();
    val == 1 ? count_exclude_padding=false : count_exclude_padding=true;
  }
  //For ConvTranspose Layer
  if( attrs.count("output_padding") ) {
    *output_padding = attrs.get<nvinfer1::Dims>("output_padding");
  }

  paddingMode = nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN;
  auto onnx_auto_pad = attrs.get("auto_pad", std::string("NOTSET"));
  if( onnx_auto_pad != "SAME_LOWER" && onnx_auto_pad != "SAME_UPPER" ) {
    if( attrs.count("pads") ) {
      auto onnx_padding = attrs.get<std::vector<int>>("pads");
      int ndim = onnx_padding.size() / 2;
      for(int i = 0; i < nbSpatialDims; ++i){
        if(i < ndim){
          beg_padding->d[i] = onnx_padding.at(i);
          end_padding->d[i] = onnx_padding.at(i + ndim);
        } else {
          beg_padding->d[i] = 0;
          end_padding->d[i] = 0;
        }
      }
    }
    if ( onnx_auto_pad != "VALID" && onnx_auto_pad != "NOTSET")
    {
      if( onnx_auto_pad == "EXPLICIT_ROUND_UP" )
      {
        paddingMode = nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP;
      }
      else if( onnx_auto_pad == "CAFFE_ROUND_DOWN" )
      {
        paddingMode = nvinfer1::PaddingMode::kCAFFE_ROUND_DOWN;
      }
      else if( onnx_auto_pad == "CAFFE_ROUND_UP" )
      {
        paddingMode = nvinfer1::PaddingMode::kCAFFE_ROUND_UP;
      }
    }
  } else {
    // If auto_pad is SAME_LOWER or SAME_UPPER, input padding should be calculated
    // "pads" attribute should not be specified
    assert(!attrs.count("pads"));
    // Note: ONNX is always NCHW ordering
    if( onnx_auto_pad == "SAME_LOWER" ) {
      paddingMode = nvinfer1::PaddingMode::kSAME_LOWER;
    } else if( onnx_auto_pad == "SAME_UPPER" ) {
      paddingMode = nvinfer1::PaddingMode::kSAME_UPPER;
    } else {
      throw std::invalid_argument("Unexpected auto_pad value: " +
                                  onnx_auto_pad);
    }
  }
}

nvinfer1::IPluginV2* importPluginFromRegistry(IImporterContext* ctx, const std::string& pluginName,
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

bool isTransposeRequired(nvinfer1::Dims const& shape, nvinfer1::Permutation const& perm)
{
    int ndim = shape.nbDims;
    int prev_significant_dim = 0;
    for (int dst_i = 0; dst_i < ndim; ++dst_i)
    {
        int src_i = perm.order[dst_i];
        int dim_i = shape.d[src_i];
        if (dim_i != 1)
        {
            // We must do a transpose for dynamically shaped tensors
            if (dim_i == -1)
            {
                return true;
            }
            if (src_i < prev_significant_dim)
            {
                return true;
            }
            prev_significant_dim = src_i;
        }
    }
    return false;
}

NodeImportResult lstmLegacyImporter(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs)
{
    // Input
    nvinfer1::ITensor& raw_input = convertToTensor(inputs.at(0), ctx);
    ASSERT(3 == raw_input.getDimensions().nbDims && "Input tensor must be 3 dimensional",
           ErrorCode::kINVALID_NODE);
    ASSERT((raw_input.getType() == nvinfer1::DataType::kFLOAT ||
            raw_input.getType() == nvinfer1::DataType::kHALF) &&
           "Only fp16 and fp32 inputs are supported",
           ErrorCode::kUNSUPPORTED_NODE);
    const nvinfer1::DataType input_type = raw_input.getType();
    const int32_t max_seq_len = raw_input.getDimensions().d[0];
    const int32_t batch_size = raw_input.getDimensions().d[1];

    // Attributes
    OnnxAttrs attrs(node, ctx);
    const std::string direction_str = attrs.get<std::string>("direction", "forward");
    ASSERT((direction_str == "forward" || direction_str == "bidirectional") &&
           "Reverse LSTM unsupported",
           ErrorCode::kUNSUPPORTED_NODE);
    const nvinfer1::RNNDirection direction = (direction_str == "forward") ?
                                             nvinfer1::RNNDirection::kUNIDIRECTION :
                                             nvinfer1::RNNDirection::kBIDIRECTION;
    const int num_directions = (direction_str == "forward") ? 1 : 2;
    // There are three distinct uses of an activation function within the LSTM equations
    // One for the input/forget/output gates, one for the cell state, and one for the output
    // RNNv2 only supports the default choice for each, listed here (duplicated for bidirectional)
    std::vector<std::string> default_activations = {"Sigmoid", "Tanh", "Tanh"};
    if (num_directions == 2) {
        default_activations.insert(default_activations.end(), {"Sigmoid", "Tanh", "Tanh"});
    }
    const std::vector<std::string> activations =
        attrs.get<std::vector<std::string>>("activations", default_activations);
    ASSERT(activations == default_activations && "Nonstandard activations within LSTM unsupported",
           ErrorCode::kUNSUPPORTED_NODE);
    const float clip = attrs.get<float>("clip", 0.0f);
    ASSERT(clip == 0.0f && "Clipping unsupported", ErrorCode::kUNSUPPORTED_NODE);
    const int32_t hidden_size = attrs.get<int>("hidden_size");
    ASSERT(hidden_size > 0, ErrorCode::kINVALID_NODE);
    const int32_t input_forget = attrs.get<int>("input_forget", 0);
    ASSERT(0 == input_forget && "Coupled input/forget unsupported", ErrorCode::kUNSUPPORTED_NODE);

    // Optional Inputs
    bool has_bias = false;
    nvinfer1::ITensor* sequence_lens = nullptr;
    nvinfer1::ITensor* initial_h = nullptr;
    nvinfer1::ITensor* initial_c = nullptr;
    for (int i = 3; i < node.input_size(); i++) {
        const std::string& input_name = node.input(i);
        if (input_name == "B") {
            has_bias = true;
        } else if (input_name == "sequence_lens") {
            sequence_lens = &(convertToTensor(inputs.at(i), ctx));
            ASSERT(sequence_lens &&
                   sequence_lens->getType() == nvinfer1::DataType::kINT32 &&
                   "Failed to process sequence_lens (sequence_lens must be int32)",
                   ErrorCode::kINVALID_NODE);
        } else if (input_name == "initial_h" || input_name == "initial_c") {
            nvinfer1::ITensor* output = nullptr;
            if (inputs.at(i).is_weights()) {
                /* constant->shuffle bug (NVBug 2650549), so we do the transpose manually */
                ShapedWeights weights = inputs.at(i).weights();
                const int dtype_size = getDtypeSize(weights.type);
                const size_t len = num_directions * batch_size * hidden_size * dtype_size;
                auto* source = reinterpret_cast<unsigned char*>(weights.values);
                std::vector<unsigned char> buffer;
                buffer.resize(len);
                for (int i = 0; i < num_directions; i++) {
                    for (int j = 0; j < batch_size; j++) {
                        for (int k = 0; k < hidden_size; k++) {
                            for (int b = 0; b < dtype_size; b++) {
                                int src_idx = i*batch_size*hidden_size*dtype_size +
                                              j*hidden_size*dtype_size + k*dtype_size + b;
                                int buf_idx = j*num_directions*hidden_size*dtype_size +
                                              i*hidden_size*dtype_size + k*dtype_size + b;
                                buffer.at(buf_idx) = source[src_idx];
                            }
                        }
                    }
                }
                std::memcpy(weights.values, static_cast<void*>(buffer.data()), len);
                const nvinfer1::Dims new_dims = {3, {batch_size, num_directions, hidden_size}};
                output = ctx->network()->addConstant(new_dims, weights)->getOutput(0);
                ASSERT(output &&
                       "Failed to convert initial_h or initial_c weights to constant layer",
                       ErrorCode::kINTERNAL_ERROR);
            } else {
                /* TODO: Once NVBug 2650549 is fixed, we can use just this path instead */
                /* nvinfer1::ITensor& source = convertToTensor(inputs.at(i), ctx); */
                nvinfer1::ITensor& source = inputs.at(i).tensor();
                auto* shuffle_layer = ctx->network()->addShuffle(source);
                ASSERT(shuffle_layer && "Failed to create initial_h shuffle layer",
                       ErrorCode::kINTERNAL_ERROR);
                shuffle_layer->setFirstTranspose(nvinfer1::Permutation{1,0,2});
                output = shuffle_layer->getOutput(0);
            }
            ASSERT(output->getType() == input_type &&
                   "initial_h and initial_c datatype must match input",
                   ErrorCode::kINVALID_NODE);
            if (input_name == "initial_h") {
                    initial_h = output;
            } else {
                    initial_c = output;
            }
        } else if (input_name == "P") {
            ASSERT(false && "Peephole connections not supported", ErrorCode::kUNSUPPORTED_NODE);
        }
    }

    // Input Shuffle Layer
    auto* input_shuffle_layer = ctx->network()->addShuffle(raw_input);
    ASSERT(input_shuffle_layer && "Failed to create input shuffle layer",
           ErrorCode::kINTERNAL_ERROR);
    input_shuffle_layer->setFirstTranspose(nvinfer1::Permutation{1,0,2});

    // RNNv2 Layer
    nvinfer1::ITensor& input_seqs = *(input_shuffle_layer->getOutput(0));
    const nvinfer1::RNNOperation op = nvinfer1::RNNOperation::kLSTM;
    const int32_t layer_count = 1;
    auto* layer = ctx->network()->addRNNv2(input_seqs, layer_count, hidden_size, max_seq_len, op);
    ASSERT(layer && "Failed to create RNNv2 layer", ErrorCode::kINTERNAL_ERROR);
    layer->setInputMode(nvinfer1::RNNInputMode::kLINEAR);
    layer->setDirection(direction);
    if (sequence_lens) {
        layer->setSequenceLengths(*sequence_lens);
    }
    if (initial_h) {
        layer->setHiddenState(*initial_h);
    }
    if (initial_c) {
        layer->setCellState(*initial_c);
    }

    // Weights
    ASSERT(inputs.at(1).is_weights() && "W must be constant", ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(2).is_weights() && "R must be constant", ErrorCode::kUNSUPPORTED_NODE);
    ShapedWeights gate_weights = inputs.at(1).weights();
    ShapedWeights rcur_weights = inputs.at(2).weights();

    nvinfer1::DataType gate_weights_type, rcur_weights_type;
    ASSERT(convertDtype(gate_weights.type, &gate_weights_type) && "Bad datatype in W",
           ErrorCode::kINTERNAL_ERROR);
    ASSERT(convertDtype(rcur_weights.type, &rcur_weights_type) && "Bad datatype in R",
           ErrorCode::kINTERNAL_ERROR);
    ASSERT(input_type == gate_weights_type && "W datatype must match X",
           ErrorCode::kINVALID_NODE);
    ASSERT(input_type == rcur_weights_type && "R datatype must match X",
           ErrorCode::kINVALID_NODE);

    ShapedWeights bias_weights;
    if (has_bias) {
        ASSERT(inputs.at(3).is_weights() && "B must be constant",
               ErrorCode::kUNSUPPORTED_NODE);
        bias_weights = inputs.at(3).weights();
        nvinfer1::DataType bias_weights_type;
        ASSERT(convertDtype(bias_weights.type, &bias_weights_type) && "Bad datatype in B",
               ErrorCode::kINTERNAL_ERROR);
        ASSERT(input_type == bias_weights_type && "B datatype must match X",
               ErrorCode::kINVALID_NODE);
    }

    const int data_size = (input_type == nvinfer1::DataType::kFLOAT) ? 4 : 2;
    const int input_size = gate_weights.shape.d[2];

    auto weightBuilder = [input_type, data_size, hidden_size, ctx]
                         (int layer_index, ShapedWeights& src, int stride, int idx)
    {
        nvinfer1::Weights w;
        int direction_offset = data_size * layer_index * 4 * hidden_size * stride;
        int gate_offset = data_size * hidden_size * stride * idx;
        w.type   = input_type;
        w.values = reinterpret_cast<void*>(
            reinterpret_cast<unsigned char*>(src.values) + direction_offset + gate_offset);
        w.count  = hidden_size * stride;
        return w;
    };

    // RNNv2 requires that a bias be set, even if none is provided
    auto zeroes = ctx->createTempWeights(gate_weights.type, nvinfer1::Dims{1, {hidden_size}});
    std::memset(zeroes.values, 0, data_size * hidden_size);

    auto biasBuilder = [input_type, data_size, hidden_size, has_bias, zeroes]
                       (int layer_index, ShapedWeights& src, int idx)
    {
        nvinfer1::Weights b;
        int direction_offset = data_size * layer_index * 8 * hidden_size;
        int gate_offset = data_size * hidden_size * idx;
        b.type = input_type;
        if (has_bias) {
            b.values = reinterpret_cast<void*>(
                reinterpret_cast<unsigned char*>(src.values) + direction_offset + gate_offset);
        } else {
            b.values = zeroes.values;
        }
        b.count = hidden_size;
        return b;
    };

    for (int layer_index = 0; layer_index < num_directions; layer_index++) {
        nvinfer1::Weights W_i = weightBuilder(layer_index, gate_weights, input_size,  0);
        nvinfer1::Weights W_o = weightBuilder(layer_index, gate_weights, input_size,  1);
        nvinfer1::Weights W_f = weightBuilder(layer_index, gate_weights, input_size,  2);
        nvinfer1::Weights W_c = weightBuilder(layer_index, gate_weights, input_size,  3);
        nvinfer1::Weights R_i = weightBuilder(layer_index, rcur_weights, hidden_size, 0);
        nvinfer1::Weights R_o = weightBuilder(layer_index, rcur_weights, hidden_size, 1);
        nvinfer1::Weights R_f = weightBuilder(layer_index, rcur_weights, hidden_size, 2);
        nvinfer1::Weights R_c = weightBuilder(layer_index, rcur_weights, hidden_size, 3);

        bool isW = true;
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kINPUT,  isW, W_i);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kOUTPUT, isW, W_o);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kFORGET, isW, W_f);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kCELL,   isW, W_c);
        isW = false;
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kINPUT,  isW, R_i);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kOUTPUT, isW, R_o);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kFORGET, isW, R_f);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kCELL,   isW, R_c);

        nvinfer1::Weights B_wi = biasBuilder(layer_index, bias_weights, 0);
        nvinfer1::Weights B_wo = biasBuilder(layer_index, bias_weights, 1);
        nvinfer1::Weights B_wf = biasBuilder(layer_index, bias_weights, 2);
        nvinfer1::Weights B_wc = biasBuilder(layer_index, bias_weights, 3);
        nvinfer1::Weights B_ri = biasBuilder(layer_index, bias_weights, 4);
        nvinfer1::Weights B_ro = biasBuilder(layer_index, bias_weights, 5);
        nvinfer1::Weights B_rf = biasBuilder(layer_index, bias_weights, 6);
        nvinfer1::Weights B_rc = biasBuilder(layer_index, bias_weights, 7);

        isW = true;
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kINPUT,  isW, B_wi);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kOUTPUT, isW, B_wo);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kFORGET, isW, B_wf);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kCELL,   isW, B_wc);
        isW = false;
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kINPUT,  isW, B_ri);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kOUTPUT, isW, B_ro);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kFORGET, isW, B_rf);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kCELL,   isW, B_rc);
    }

    // Outputs
    ASSERT(layer->getNbOutputs() == 3, ErrorCode::kINTERNAL_ERROR);
    ASSERT(node.output_size() <= 3, ErrorCode::kINVALID_NODE);
    std::vector<TensorOrWeights> outputs;
    for (int i = 0; i < node.output_size(); i++) {
        auto* shuffle_layer = ctx->network()->addShuffle(*(layer->getOutput(i)));
        ASSERT(shuffle_layer && "Failed to create output shuffle layer",
               ErrorCode::kINTERNAL_ERROR);
        shuffle_layer->setFirstTranspose(nvinfer1::Permutation{1,0,2});
        if (i == 0) {
            nvinfer1::Dims Y_dims{4, {max_seq_len, batch_size, num_directions, hidden_size}};
            shuffle_layer->setReshapeDimensions(Y_dims);
            shuffle_layer->setSecondTranspose(nvinfer1::Permutation{0,2,1,3});
        }
        outputs.emplace_back(shuffle_layer->getOutput(0));
    }
    return {outputs};
}

nvinfer1::Dims makeDims(int nbDims, int val)
{
    nvinfer1::Dims dims;
    dims.nbDims = nbDims;
    std::fill_n(dims.d, nbDims, val);
    return dims;
}

nvinfer1::ITensor& makeShapeTensor(IImporterContext* ctx, nvinfer1::Dims dims)
{
    auto tempWeights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto_DataType_INT32, makeDims(1, dims.nbDims));
    for(int i = 0; i < dims.nbDims; i++)
    {
        static_cast<int32_t*>(tempWeights.values)[i] = dims.d[i];
    }
    auto valueWeights = TensorOrWeights{tempWeights};
    return convertToTensor(valueWeights, ctx);
}

NodeImportResult reduceTensor(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, TensorOrWeights input,
    nvinfer1::ReduceOperation operation)
{
    nvinfer1::ITensor& tensor = convertToTensor(input, ctx);
    OnnxAttrs attrs(node, ctx);
    bool keepdims = attrs.get("keepdims", 1);
    int ndim = tensor.getDimensions().nbDims;
    std::vector<int> axes;
    if (attrs.count("axes"))
    {
        axes = attrs.get<std::vector<int>>("axes");
    }
    else
    {
        axes.resize(ndim);
        std::iota(axes.begin(), axes.end(), 0);
    }

    uint32_t axisMask = 0;
    for (int axis : axes)
    {
        TRT_CHECK(convertAxis(axis, ndim));
        axisMask |= 1 << axis;
    }

    auto* layer = ctx->network()->addReduce(tensor, operation, axisMask, keepdims);
    return {{layer->getOutput(0)}};
}

nvinfer1::ITensor* reshapeTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, nvinfer1::Dims shape)
{
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

NodeImportResult scaleHelper(IImporterContext* ctx, nvinfer1::ITensor& tensor_, nvinfer1::ScaleMode mode,
    nvinfer1::Weights shift, nvinfer1::Weights scale, nvinfer1::Weights power)
{
    nvinfer1::ITensor* tensor_ptr = &tensor_;
    nvinfer1::Dims dims = tensor_ptr->getDimensions();

    // TensorRT scale layers support 4D(NCHW) or 5D(NCDHW) input.
    // For input other than 4D or 5D will be expanded to 4D.
    int expectedNbDims = 4;
    bool need_to_expand_dims = (dims.nbDims != 4 && dims.nbDims != 5);
    nvinfer1::Dims orig_shape = dims;
    if (need_to_expand_dims)
    {
        // Expand or squash dims to 4D
        nvinfer1::Dims new_shape = dims;
        while (new_shape.nbDims < expectedNbDims)
        {
            new_shape.d[new_shape.nbDims++] = 1;
        }
        while (new_shape.nbDims > expectedNbDims)
        {
            new_shape.d[3] *= new_shape.d[--new_shape.nbDims];
        }
        tensor_ptr = reshapeTensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensor_ptr->getDimensions();
    }

    ASSERT(dims.nbDims == 4 || dims.nbDims == 5, ErrorCode::kUNSUPPORTED_NODE);

    // Fill in dtype for any unused (dummy) weights
    nvinfer1::DataType* dtype_ptr = nullptr;
    if (shift.count)
    {
        dtype_ptr = &shift.type;
    }
    if (scale.count)
    {
        ASSERT(!dtype_ptr || *dtype_ptr == scale.type, ErrorCode::kUNSUPPORTED_NODE);
        dtype_ptr = &scale.type;
    }
    if (power.count)
    {
        ASSERT(!dtype_ptr || *dtype_ptr == power.type, ErrorCode::kUNSUPPORTED_NODE);
        dtype_ptr = &power.type;
    }
    ASSERT(dtype_ptr, ErrorCode::kINTERNAL_ERROR);
    shift.type = *dtype_ptr;
    scale.type = *dtype_ptr;
    power.type = *dtype_ptr;
    auto* layer = ctx->network()->addScaleNd(*tensor_ptr, mode, shift, scale, power, 1);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    tensor_ptr = layer->getOutput(0);

    if (need_to_expand_dims)
    {
        tensor_ptr = reshapeTensor(ctx, *tensor_ptr, orig_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    }

    return {{tensor_ptr}};
}

void setAttr(nvinfer1::Dims * trtAttr, ::ONNX_NAMESPACE::AttributeProto const* onnxAttr, int nbSpatialDims, int defaultVal){
  assert(trtAttr->nbDims == nbSpatialDims);
  int ndim = onnxAttr->ints().size();
  for(int i = 0; i < nbSpatialDims; ++i){
      if(i < ndim){
        trtAttr->d[i] = onnxAttr->ints(i);
      } else {
        trtAttr->d[i] = defaultVal;
      }
  }
}

nvinfer1::ITensor* transposeTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, nvinfer1::Permutation const& perm, bool permute_dim_types)
{
    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
    if (!layer)
    {
        return nullptr;
    }
    nvinfer1::Dims shape = tensor.getDimensions();
    // If a transpose is required, add transpose property to the shuffle layer.
    if (isTransposeRequired(shape, perm))
    {
        layer->setFirstTranspose(perm);
    }
    // Else, the transpose can be simplified to a reshape.
    else
    {
        nvinfer1::Dims new_shape;
        new_shape.nbDims = shape.nbDims;
        for (int i = 0; i < new_shape.nbDims; ++i)
        {
            new_shape.d[i] = shape.d[perm.order[i]];
        }
        layer->setReshapeDimensions(new_shape);
    }
    return layer->getOutput(0);
}

NodeImportResult unaryHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::UnaryOperation op)
{
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    nvinfer1::IUnaryLayer* layer = ctx->network()->addUnary(input, op);
    return {{layer->getOutput(0)}};
}

int64_t volume(const nvinfer1::Dims& dims)
{
    std::for_each(dims.d, dims.d + dims.nbDims, [](int d){ assert(d >= 0 && "volume makes no sense for dynamic shapes");});
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>{});
}

Status weightsToVector(TensorOrWeights weights, std::vector<int64_t>* weightVector)
{
    ASSERT(weights.is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    ASSERT((weights.weights().type == ::ONNX_NAMESPACE::TensorProto::INT32) || (weights.weights().type == ::ONNX_NAMESPACE::TensorProto::INT64),
        ErrorCode::kINVALID_NODE);
    weightVector->resize(weights.weights().count());
    if (weights.weights().type == ::ONNX_NAMESPACE::TensorProto::INT64)
    {
        auto array_start = static_cast<int64_t*>(weights.weights().values);
        std::copy(array_start, array_start + weights.weights().count(), weightVector->begin());
    }
    else
    {
        auto array_start = static_cast<int32_t*>(weights.weights().values);
        std::copy(array_start, array_start + weights.weights().count(), weightVector->begin());
    }
    return Status(ErrorCode::kSUCCESS);
}


} // namespace onnx2trt
