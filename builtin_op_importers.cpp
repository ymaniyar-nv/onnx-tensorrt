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

#include "ModelImporter.hpp"
#include "builtin_op_importers.hpp"
#include "NvInferPlugin.h"
#include "onnx2trt_utils.hpp"
#include "OnnxAttrs.hpp"

#include <algorithm> // For std::min, std::max
#include <numeric>   // For std::iota
#include <cstring>   // For std::memcpy, std::memset
#include <tuple>
#include <array>

namespace onnx2trt
{

string_map<NodeImporter>& getBuiltinOpImporterMap()
{
    static string_map<NodeImporter> builtin_op_importers;
    return builtin_op_importers;
}

namespace
{

#define IGNORE_UNUSED_GLOBAL(x)                                                                                        \
    static void _ignore_unused2_##x();                                                                                 \
    static void _ignore_unused1_##x()                                                                                  \
    {                                                                                                                  \
        (void) _ignore_unused2_##x;                                                                                    \
        (void) x;                                                                                                      \
    }                                                                                                                  \
    static void _ignore_unused2_##x()                                                                                  \
    {                                                                                                                  \
        (void) _ignore_unused1_##x;                                                                                    \
    }                                                                                                                  \
    struct SwallowSemicolon##x                                                                                         \
    {                                                                                                                  \
    }

#define DECLARE_BUILTIN_OP_IMPORTER(op)                                                                                \
    NodeImportResult import##op(                                                                                       \
        IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs)

#define DEFINE_BUILTIN_OP_IMPORTER(op)                                                                                 \
    NodeImportResult import##op(                                                                                       \
        IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs);         \
    static const bool op##_registered_builtin_op = registerBuiltinOpImporter(#op, import##op);                         \
    IGNORE_UNUSED_GLOBAL(op##_registered_builtin_op);                                                                  \
    NodeImportResult import##op(                                                                                       \
        IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs)

#define RETURN_FIRST_OUTPUT(layer)                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        nvinfer1::ILayer* layer_ptr = layer;                                                                           \
        ASSERT(layer_ptr, ErrorCode::kUNSUPPORTED_NODE);                                                               \
        return {{layer_ptr->getOutput(0)}};                                                                            \
    } while (0)

#define RETURN_IDENTITY(input)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        TensorOrWeights output = identity(ctx, input);                                                                 \
        ASSERT(output, ErrorCode::kUNSUPPORTED_NODE);                                                                  \
        return {{output}};                                                                                             \
    } while (0)

#define RETURN_ALL_OUTPUTS(layer)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        nvinfer1::ILayer* layer_ptr = layer;                                                                           \
        ASSERT(layer_ptr, ErrorCode::kUNSUPPORTED_NODE);                                                               \
        std::vector<TensorOrWeights> outputs;                                                                          \
        for (int i = 0; i < layer_ptr->getNbOutputs(); ++i)                                                            \
            outputs.push_back(layer_ptr->getOutput(i));                                                                \
        return {outputs};                                                                                              \
    } while (0)

bool registerBuiltinOpImporter(std::string op, NodeImporter const& importer)
{
    bool inserted = getBuiltinOpImporterMap().insert({op, importer}).second;
    assert(inserted);
    return inserted;
}

DEFINE_BUILTIN_OP_IMPORTER(Sin)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kSIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Cos)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kCOS);
}

DEFINE_BUILTIN_OP_IMPORTER(Tan)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kTAN);
}

DEFINE_BUILTIN_OP_IMPORTER(Sinh)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kSINH);
}

DEFINE_BUILTIN_OP_IMPORTER(Cosh)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kCOSH);
}

DEFINE_BUILTIN_OP_IMPORTER(Asin)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kASIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Acos)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kACOS);
}

DEFINE_BUILTIN_OP_IMPORTER(Atan)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kATAN);
}

DEFINE_BUILTIN_OP_IMPORTER(Asinh)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kASINH);
}

DEFINE_BUILTIN_OP_IMPORTER(Acosh)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kACOSH);
}

DEFINE_BUILTIN_OP_IMPORTER(Atanh)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kATANH);
}

DEFINE_BUILTIN_OP_IMPORTER(Ceil)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kCEIL);
}

DEFINE_BUILTIN_OP_IMPORTER(Floor)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kFLOOR);
}

DEFINE_BUILTIN_OP_IMPORTER(Erf)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kERF);
}

DEFINE_BUILTIN_OP_IMPORTER(Abs)
{
    return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kABS);
}

DEFINE_BUILTIN_OP_IMPORTER(Add)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kSUM);
}

DEFINE_BUILTIN_OP_IMPORTER(ArgMax)
{
    return argMinMaxHelper(ctx, node, inputs, nvinfer1::TopKOperation::kMAX);
}

DEFINE_BUILTIN_OP_IMPORTER(ArgMin)
{
    return argMinMaxHelper(ctx, node, inputs, nvinfer1::TopKOperation::kMIN);
}

DEFINE_BUILTIN_OP_IMPORTER(AveragePool)
{
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = tensor_ptr->getDimensions();
    bool need_to_expand_dims = (dims.nbDims == 3);
    if (need_to_expand_dims)
    {
        // Expand spatial dims from 1D to 2D
        nvinfer1::DimsNCHW new_shape(dims.d[0], dims.d[1], dims.d[2], 1);
        tensor_ptr = reshapeTensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensor_ptr->getDimensions();
    }

    // Support for opset10 ceil_mode
    CeilingPoolDim ceilingPool;
    // Ceiling and dialations added in opset 10
    if (ctx->getOpsetVersion() >= 10)
    {
        OnnxAttrs attrs(node, ctx);
        const auto ceil_mode = attrs.get<int>("ceil_mode", 0);
        const auto dilations = attrs.get<std::vector<int>>("dilations", std::vector<int>(2, 1));
        for (size_t i = 0; i < dilations.size(); i++)
            ASSERT(dilations[i] == 1, ErrorCode::kUNSUPPORTED_NODE); // Do not support pooling dilations currently
        if (ceil_mode != 0) // Need to set pooling formula to use ceiling instead of floor
        {
            ctx->network()->setPoolingOutputDimensionsFormula(&ceilingPool);
        }
    }

    ASSERT(dims.nbDims >= 4, ErrorCode::kUNSUPPORTED_NODE);

    int nbSpatialDims = dims.nbDims - 2;
    ASSERT(nbSpatialDims == 2 || nbSpatialDims == 3, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Dims kernel_size = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims beg_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims end_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::PaddingMode paddingMode;

    bool exclude_padding(true);
    getKernelParams(ctx, node, &kernel_size, &strides, &beg_padding, &end_padding, paddingMode, exclude_padding);
    nvinfer1::IPoolingLayer* pooling_layer
        = ctx->network()->addPoolingNd(*tensor_ptr, nvinfer1::PoolingType::kAVERAGE, kernel_size);
    nvinfer1::ILayer* layer = pooling_layer;
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    pooling_layer->setStrideNd(strides);
    pooling_layer->setAverageCountExcludesPadding(exclude_padding);
    pooling_layer->setPaddingMode(paddingMode);

    pooling_layer->setPrePadding(beg_padding);
    pooling_layer->setPostPadding(end_padding);
    // Note: Average pooling requires special care with asymmetric padding
    //       because the need to exclude padding pixels from the average
    //       means we can't just use a pre-padding layer.
    nvinfer1::DimsHW pre_crop(0, 0), post_crop(0, 0);
    for (int d = 0; d < 2; ++d)
    {
        if (end_padding.d[d] == beg_padding.d[d])
        {
            // Symmetric padding, nothing special needed
        }
        else if (end_padding.d[d] == beg_padding.d[d] + 1)
        {
            // Pad symmetrically such that we get one more output element at
            // the beginning, and then crop it off after the pooling operation.
            beg_padding.d[d] += strides.d[d];
            pre_crop.d[d] = 1;
        }
        else
        {
            bool supported_form_of_asymmetric_padding_for_AveragePool = false;
            ASSERT(supported_form_of_asymmetric_padding_for_AveragePool, ErrorCode::kUNSUPPORTED_NODE);
        }
    }
    tensor_ptr = layer->getOutput(0);
    dims = tensor_ptr->getDimensions();
    if (need_to_expand_dims)
    {
        // Un-expand spatial dims back to 1D
        nvinfer1::Dims new_shape{3, {dims.d[0], dims.d[1], dims.d[2]}};
        tensor_ptr = reshapeTensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    }
    return {{tensor_ptr}};
}

DEFINE_BUILTIN_OP_IMPORTER(BatchNormalization)
{
    // Scale, bias, mean, and variance must be initializers
    ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(3).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(4).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    auto scale_weights = inputs.at(1).weights();
    auto bias_weights = inputs.at(2).weights();
    auto mean_weights = inputs.at(3).weights();
    auto variance_weights = inputs.at(4).weights();
    OnnxAttrs attrs(node, ctx);
    float eps = attrs.get<float>("epsilon", 1e-5f);
    ASSERT(scale_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT
            && bias_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT
            && mean_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT
            && variance_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT,
        ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Dims dims = tensor_ptr->getDimensions();

    bool need_to_expand_dims = (dims.nbDims == 3);
    if (need_to_expand_dims)
    {
        // Expand spatial dims from 1D to 2D
        nvinfer1::Dims new_shape{4, {dims.d[0], dims.d[1], dims.d[2], 1}};
        tensor_ptr = reshapeTensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensor_ptr->getDimensions();
    }

    int nchan = dims.d[1];
    nvinfer1::Dims weights_shape{1, {nchan}};
    ASSERT(scale_weights.shape == weights_shape, ErrorCode::kINVALID_NODE);
    ASSERT(bias_weights.shape == weights_shape, ErrorCode::kINVALID_NODE);
    ASSERT(mean_weights.shape == weights_shape, ErrorCode::kINVALID_NODE);
    ASSERT(variance_weights.shape == weights_shape, ErrorCode::kINVALID_NODE);
    auto combined_scale_weights = ctx->createTempWeights(scale_weights.type, scale_weights.shape);
    auto combined_bias_weights = ctx->createTempWeights(bias_weights.type, bias_weights.shape);
    size_t nweight = nchan;
    // Fold the weights together into a single bias and scale
    for (size_t i = 0; i < nweight; ++i)
    {
        float scale = (static_cast<float const*>(scale_weights.values))[i];
        float bias = (static_cast<float const*>(bias_weights.values))[i];
        float mean = (static_cast<float const*>(mean_weights.values))[i];
        float variance = (static_cast<float const*>(variance_weights.values))[i];
        float& combined_scale_ref = const_cast<float*>(static_cast<float const*>(combined_scale_weights.values))[i];
        float& combined_bias_ref = const_cast<float*>(static_cast<float const*>(combined_bias_weights.values))[i];
        combined_scale_ref = scale / sqrtf(variance + eps);
        combined_bias_ref = bias - mean * combined_scale_ref;
    }

    // If dimensions were not expanded return the output of the scale operation
    if (!need_to_expand_dims)
    {
        return scaleHelper(ctx, *tensor_ptr, nvinfer1::ScaleMode::kCHANNEL, combined_bias_weights, combined_scale_weights, {});
    }
    else
    {
        auto scaledResult = scaleHelper(ctx, *tensor_ptr, nvinfer1::ScaleMode::kCHANNEL, combined_bias_weights, combined_scale_weights, {});
        // Squeeze spatial dims back to 1D
        tensor_ptr = &convertToTensor(scaledResult.value().at(0), ctx);
        nvinfer1::Dims new_shape{3, {dims.d[0], dims.d[1], dims.d[2]}};
        tensor_ptr = reshapeTensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        return {{tensor_ptr}};
    }
}

DEFINE_BUILTIN_OP_IMPORTER(Cast)
{
    // Get input node.
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs(node, ctx);
    // Get data type to cast to.
    nvinfer1::DataType dtype = attrs.get<nvinfer1::DataType>("to");
    ASSERT(static_cast<int>(dtype) != -1, ErrorCode::kINVALID_VALUE);
    LOG_VERBOSE("Casting to type: " << dtype);
    // Add the layer.
    nvinfer1::IIdentityLayer* layer = ctx->network()->addIdentity(tensor);
    layer->setPrecision(dtype);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Clip)
{
    OnnxAttrs attrs(node, ctx);
    // beta is the upper bound.
    float alpha = attrs.get("min", std::numeric_limits<float>::lowest());
    float beta = attrs.get("max", std::numeric_limits<float>::max());
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kCLIP, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Concat)
{
    std::vector<nvinfer1::ITensor*> tensors;
    for (auto& input : inputs)
    {
        tensors.push_back(&convertToTensor(input, ctx));
    }
    OnnxAttrs attrs(node, ctx);
    int axis = attrs.get<int>("axis");
    int nbDims = inputs.at(0).shape().nbDims;
    TRT_CHECK(convertAxis(axis, nbDims));
    auto* layer = ctx->network()->addConcatenation(tensors.data(), tensors.size());
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    layer->setAxis(axis);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Constant)
{
    // TODO: This silently fails if the dtype is not supported
    OnnxAttrs attrs(node, ctx);
    // Having the trt_outputs_range_min attributes means it's from
    // serialized iNetworkDefinition.
    if (!attrs.get<std::vector<float>>("trt_outputs_range_min", {}).empty()) {
        // just create a constant layer here for 1-1 mapping during network deserialization
        auto weights = attrs.get<ShapedWeights>("value");
        auto* layer = ctx->network()->addConstant(weights.shape, weights);
        RETURN_FIRST_OUTPUT(layer);
    }
    return {{attrs.get<ShapedWeights>("value")}};
}

DEFINE_BUILTIN_OP_IMPORTER(ConstantOfShape)
{
    OnnxAttrs attrs(node, ctx);
    nvinfer1::ITensor* shape = &convertToTensor(inputs.at(0), ctx);

    ShapedWeights zeroWeights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, nvinfer1::Dims{1, 1});
    static_cast<float*>(zeroWeights.values)[0] = 0.f;
    auto valueWeights = TensorOrWeights{attrs.get("value", zeroWeights)};

    nvinfer1::ITensor* value = &convertToTensor(valueWeights, ctx);
    return {{constantOfShape(ctx, value, shape)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Conv)
{
    // Convolution Weights must be an initializer
    ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    auto kernel_weights = inputs.at(1).weights();
    nvinfer1::Dims dims = tensor_ptr->getDimensions();
    LOG_VERBOSE("Convolution input dimensions: " << dims);

    bool need_to_expand_dims = (dims.nbDims == 3);
    if (need_to_expand_dims)
    {
        // Expand spatial dims from 1D to 2D
        nvinfer1::Dims new_shape{4, {dims.d[0], dims.d[1], dims.d[2], 1}};
        tensor_ptr = reshapeTensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensor_ptr->getDimensions();
    }
    if (kernel_weights.shape.nbDims == 3)
    {
        kernel_weights.shape.nbDims = 4;
        kernel_weights.shape.d[3] = 1;
    }

    const int nbSpatialDims = dims.nbDims - 2;
    // Check that the number of spatial dimensions and the kernel shape matches up.
    ASSERT(nbSpatialDims == kernel_weights.shape.nbDims - 2, ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::Weights bias_weights;
    if (inputs.size() == 3)
    {
        ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
        auto shaped_bias_weights = inputs.at(2).weights();
        ASSERT(shaped_bias_weights.shape.nbDims == 1, ErrorCode::kINVALID_NODE);
        ASSERT(shaped_bias_weights.shape.d[0] == kernel_weights.shape.d[0], ErrorCode::kINVALID_NODE);
        bias_weights = shaped_bias_weights;
    }
    else
    {
        bias_weights = ShapedWeights::empty(kernel_weights.type);
    }
    nvinfer1::Dims kernel_size;
    kernel_size.nbDims = nbSpatialDims;
    for (int i = 1; i <= nbSpatialDims; ++i)
    {
        kernel_size.d[nbSpatialDims - i] = kernel_weights.shape.d[kernel_weights.shape.nbDims - i];
    }
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims beg_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims end_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims dilations = makeDims(nbSpatialDims, 1);
    nvinfer1::PaddingMode paddingMode;
    bool exclude_padding;
    getKernelParams(ctx,
        node, &kernel_size, &strides, &beg_padding, &end_padding, paddingMode, exclude_padding, &dilations);

    for (int i = 1; i <= nbSpatialDims; ++i)
    {
        ASSERT(kernel_size.d[nbSpatialDims - i] == kernel_weights.shape.d[kernel_weights.shape.nbDims - i],
            ErrorCode::kUNSUPPORTED_NODE);
    }

    int nchan = dims.d[1];
    int noutput = kernel_weights.shape.d[0];
    nvinfer1::IConvolutionLayer* layer
        = ctx->network()->addConvolutionNd(*tensor_ptr, noutput, kernel_size, kernel_weights, bias_weights);

    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    layer->setStrideNd(strides);
    layer->setPaddingMode(paddingMode);
    layer->setPrePadding(beg_padding);
    layer->setPostPadding(end_padding);
    layer->setDilationNd(dilations);
    OnnxAttrs attrs(node, ctx);
    int ngroup = attrs.get("group", 1);
    ASSERT(nchan == -1 || kernel_weights.shape.d[1] * ngroup == nchan, ErrorCode::kINVALID_NODE);
    layer->setNbGroups(ngroup);
    tensor_ptr = layer->getOutput(0);
    dims = tensor_ptr->getDimensions();

    if (need_to_expand_dims)
    {
        // Un-expand spatial dims back to 1D
        nvinfer1::Dims new_shape{3, {dims.d[0], dims.d[1], dims.d[2]}};
        tensor_ptr = reshapeTensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    }

    LOG_VERBOSE("Using kernel: " << kernel_size << ", strides: " << strides << ", padding: " << beg_padding
                                 << ", dilations: " << dilations << ", numOutputs: " << noutput);
    LOG_VERBOSE("Convolution output dimensions: " << dims);
    return {{tensor_ptr}};
}

// TRT only supports 2D or 3D deconvolutions (Layout: [N,C,D1,D2,(D3)])
// Inputs should be of dimension 4 or 5.
// When input.nbDims = 3, we expand it to 4D
DEFINE_BUILTIN_OP_IMPORTER(ConvTranspose)
{
    // Deconvolution input must be at least 3D.
    ASSERT(inputs.at(0).shape().nbDims >= 3, ErrorCode::kUNSUPPORTED_NODE);
    // Deconvolution weights must be an initializer
    ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);

    // Kernel weights have layout [C, M/group, k1, k2, (k3)]
    auto kernel_weights = inputs.at(1).weights();
    nvinfer1::Dims dims = tensor_ptr->getDimensions();

    bool need_to_expand_dims = (dims.nbDims == 3);
    if (need_to_expand_dims)
    {
        nvinfer1::Dims new_shape{4, {dims.d[0], dims.d[1], dims.d[2], 1}};
        tensor_ptr = reshapeTensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensor_ptr->getDimensions();
    }
    if (kernel_weights.shape.nbDims == 3)
    {
        kernel_weights.shape.nbDims = 4;
        kernel_weights.shape.d[3] = 1;
    }

    const int nbSpatialDims = dims.nbDims - 2;
    // Check that the number of spatial dimensions and the kernel shape matches up.
    ASSERT(nbSpatialDims == kernel_weights.shape.nbDims - 2, ErrorCode::kUNSUPPORTED_NODE);

    // Check for bias_weights
    nvinfer1::Weights bias_weights;
    if (inputs.size() == 3)
    {
        ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
        auto shaped_bias_weights = inputs.at(2).weights();
        // ONNX requires shaped_bias_weights to be 1D
        ASSERT(shaped_bias_weights.shape.nbDims == 1, ErrorCode::kINVALID_NODE);
        ASSERT(shaped_bias_weights.shape.d[0] == kernel_weights.shape.d[1], ErrorCode::kINVALID_NODE);
        bias_weights = shaped_bias_weights;
    }
    else
    {
        bias_weights = ShapedWeights::empty(kernel_weights.type);
    }

    // Get all attributes
    OnnxAttrs attrs(node, ctx);
    nvinfer1::Dims output_shape;
    nvinfer1::Dims output_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims kernel_size;
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims beg_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims end_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims dilations = makeDims(nbSpatialDims, 1);
    nvinfer1::PaddingMode paddingMode;
    bool exclude_padding = false;
    bool explicit_output_shape = false;
    int nchan = dims.d[1];

    ASSERT(kernel_weights.shape.d[0] == nchan, ErrorCode::kINVALID_NODE);
    int ngroup = attrs.get("group", 1);
    int noutput = kernel_weights.shape.d[1] * ngroup; // Note: Weights order is CKRS

    if (attrs.count("output_shape"))
    {
        output_shape = attrs.get<nvinfer1::Dims>("output_shape");
        explicit_output_shape = true;
    }

    kernel_size.nbDims = nbSpatialDims;
    for (int i = 1; i <= nbSpatialDims; ++i)
    {
        kernel_size.d[nbSpatialDims - i] = kernel_weights.shape.d[kernel_weights.shape.nbDims - i];
    }

    getKernelParams(ctx, node, &kernel_size, &strides, &beg_padding, &end_padding, paddingMode, exclude_padding,
        &dilations, &output_padding);
    // TRT only support 2D padding
    ASSERT(output_padding.nbDims == 2 || (output_padding.nbDims == 3 && output_padding.d[0] == 0),
        ErrorCode::kUNSUPPORTED_NODE);

    for (int i = 1; i <= nbSpatialDims; ++i)
    {
        ASSERT(kernel_size.d[nbSpatialDims - i] == kernel_weights.shape.d[kernel_weights.shape.nbDims - i],
            ErrorCode::kUNSUPPORTED_NODE);
        // TRT does not support dilated deconvolutions
        ASSERT(dilations.d[nbSpatialDims - i] == 1, ErrorCode::kUNSUPPORTED_GRAPH);
    }
    // If output shape is given, calculate the input or output padding values
    if (explicit_output_shape)
    {
        generatePadding(dims, output_shape, kernel_size, strides, dilations, nbSpatialDims, beg_padding,
            end_padding, output_padding, paddingMode);
        // TRT only support 2D padding
        ASSERT(output_padding.nbDims == 2 || (output_padding.nbDims == 3 && output_padding.d[0] == 0),
            ErrorCode::kUNSUPPORTED_NODE);
    }

    nvinfer1::IDeconvolutionLayer* deconv_layer
        = ctx->network()->addDeconvolutionNd(*tensor_ptr, noutput, kernel_size, kernel_weights, bias_weights);
    ASSERT(deconv_layer, ErrorCode::kUNSUPPORTED_NODE);

    deconv_layer->setStrideNd(strides);
    deconv_layer->setPaddingMode(paddingMode);
    deconv_layer->setPrePadding(beg_padding);
    deconv_layer->setPostPadding(end_padding);
    deconv_layer->setNbGroups(ngroup);
    tensor_ptr = deconv_layer->getOutput(0);

    nvinfer1::DimsHW output_padding_HW;
    if (output_padding.nbDims == 2)
    {
        output_padding_HW = nvinfer1::DimsHW(output_padding.d[0], output_padding.d[1]);
    }
    else
    {
        output_padding_HW = nvinfer1::DimsHW(output_padding.d[1], output_padding.d[2]);
    }

    if (output_padding_HW != nvinfer1::DimsHW(0, 0))
    {
        tensor_ptr = ctx->network()->addPadding(*tensor_ptr, nvinfer1::DimsHW(), output_padding_HW)->getOutput(0);
    }

    dims = tensor_ptr->getDimensions();

    if (need_to_expand_dims)
    {
        nvinfer1::Dims new_shape{3, {dims.d[0], dims.d[1], dims.d[2]}};
        tensor_ptr = reshapeTensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    }
    return {{tensor_ptr}};
}

DEFINE_BUILTIN_OP_IMPORTER(DepthToSpace)
{
    // The Tensor has to be of shape NCHW
    ASSERT(inputs.at(0).shape().nbDims == 4, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(*tensor_ptr);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    OnnxAttrs attrs(node, ctx);
    int block_size = attrs.get<int>("blocksize");
    nvinfer1::Dims dims = tensor_ptr->getDimensions();
    int ndim_spatial = dims.nbDims - 2;
    nvinfer1::Dims new_shape1;
    new_shape1.nbDims = dims.nbDims + ndim_spatial;
    new_shape1.d[0] = dims.d[0];
    new_shape1.d[ndim_spatial + 1] = dims.d[1];
    for (int i = 1; i < ndim_spatial + 1; ++i)
    {
        ASSERT(new_shape1.d[ndim_spatial + 1] % block_size == 0, ErrorCode::kINVALID_NODE);
        new_shape1.d[ndim_spatial + 1] /= block_size;
        new_shape1.d[i] = block_size;
        new_shape1.d[ndim_spatial + 1 + i] = dims.d[1 + i];
    }
    layer->setReshapeDimensions(new_shape1);
    nvinfer1::Permutation perm;
    perm.order[0] = 0;
    perm.order[1] = ndim_spatial + 1;
    for (int i = 1; i < ndim_spatial + 1; ++i)
    {
        perm.order[2 * i] = ndim_spatial + 1 + i;
        perm.order[2 * i + 1] = i;
    }
    layer->setSecondTranspose(perm);
    tensor_ptr = layer->getOutput(0);
    dims = tensor_ptr->getDimensions();
    nvinfer1::Dims new_shape2;
    new_shape2.nbDims = dims.nbDims - ndim_spatial;
    new_shape2.d[0] = dims.d[0];
    new_shape2.d[1] = dims.d[1];
    for (int i = 1; i < ndim_spatial + 1; ++i)
    {
        new_shape2.d[1 + i] = dims.d[2 * i] * dims.d[1 + 2 * i];
    }
    tensor_ptr = reshapeTensor(ctx, *tensor_ptr, new_shape2);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    return {{tensor_ptr}};
}

DECLARE_BUILTIN_OP_IMPORTER(Mul);
DEFINE_BUILTIN_OP_IMPORTER(Div)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kDIV);
}

DEFINE_BUILTIN_OP_IMPORTER(Dropout)
{
    int noutputs = node.output().size();
    if (noutputs == 1)
    {
        RETURN_IDENTITY(inputs.at(0));
    }
    else
    {
        // Error if opset version >= 10 as boolean not supported right now
        ASSERT(ctx->getOpsetVersion() < 10, ErrorCode::kUNSUPPORTED_NODE);
        // Add identity layer twice for both Dropout outputs: (output + mask)
        std::vector<TensorOrWeights> outputs;
        outputs.push_back(identity(ctx, inputs.at(0)));
        outputs.push_back(identity(ctx, inputs.at(0)));
        return outputs;
    }
}

DEFINE_BUILTIN_OP_IMPORTER(Elu)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha", 1.f);
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kELU, &alpha);
}

DEFINE_BUILTIN_OP_IMPORTER(Equal)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kEQUAL);
}

DEFINE_BUILTIN_OP_IMPORTER(Exp)
{
    return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kEXP);
}

DEFINE_BUILTIN_OP_IMPORTER(Expand)
{
    auto* n = ctx->network();

    auto* inputTensor = &convertToTensor(inputs.at(0), ctx);
    auto* expandShapeTensor = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::Dims shapeTensorDims = expandShapeTensor->getDimensions();
    nvinfer1::Dims inputDims = inputTensor->getDimensions();
    ASSERT(shapeTensorDims.nbDims == 1, ErrorCode::kINVALID_VALUE);
    int nbOutputDims = shapeTensorDims.d[0];

    if (nbOutputDims > inputDims.nbDims)
    {
        nvinfer1::Dims newDims = expand_dims(inputDims, nbOutputDims);
        nvinfer1::IShuffleLayer* unsqueeze = n->addShuffle(*inputTensor);
        unsqueeze->setReshapeDimensions(newDims);
        inputTensor = unsqueeze->getOutput(0);
        inputDims = inputTensor->getDimensions();
        LOG_VERBOSE("Unsqueezed input to: " << inputDims);
    }

    const int nbDims = inputDims.nbDims;

    nvinfer1::Dims dimsShapeZero = makeDims(nbDims, 0);
    nvinfer1::Dims dimsShapeOne = makeDims(nbDims, 1);
    nvinfer1::Dims dimsShapeNegOne = makeDims(nbDims, -1);

    auto* shapeTensorZero = &makeShapeTensor(ctx, dimsShapeZero);
    auto* shapeTensorOne = &makeShapeTensor(ctx, dimsShapeOne);
    auto* shapeTensorInput = &makeShapeTensor(ctx, inputDims);

    // Get the maximum shape from the input and the expanded tensor so that there is no overflow.
    // Formula: size = max(inputShape, expandedShape)
    auto* maxSizedShape
        = n->addElementWise(*shapeTensorInput, *expandShapeTensor, nvinfer1::ElementWiseOperation::kMAX);
    ASSERT(maxSizedShape, ErrorCode::kINTERNAL_ERROR);
    auto* shapeTensorSize = maxSizedShape->getOutput(0);

    // The slice layer's stride should be: stride = (inputShape == 1) ? 0 : 1,
    // Use (inputShape - 1) / max(inputShape - 1, 1) to implement it here. This assume the input shape is all positive integers.
    auto* inputSubOne = n->addElementWise(*shapeTensorInput, *shapeTensorOne, nvinfer1::ElementWiseOperation::kSUB);
    ASSERT(inputSubOne, ErrorCode::kINTERNAL_ERROR);
    auto* shapeTensorInputSubOne = inputSubOne->getOutput(0);

    auto* avoidZero = n->addElementWise(*shapeTensorInputSubOne, *shapeTensorOne, nvinfer1::ElementWiseOperation::kMAX);
    ASSERT(avoidZero, ErrorCode::kINTERNAL_ERROR);
    auto* shapeTensorAvoidZero = avoidZero->getOutput(0);

    auto* computeStride
        = n->addElementWise(*shapeTensorInputSubOne, *shapeTensorAvoidZero, nvinfer1::ElementWiseOperation::kDIV);
    ASSERT(computeStride, ErrorCode::kINTERNAL_ERROR);
    auto* shapeTensorStride = computeStride->getOutput(0);

    auto* layerSlice = n->addSlice(*inputTensor, dimsShapeNegOne, dimsShapeNegOne, dimsShapeNegOne);
    int i{1};
    for (auto& a : {shapeTensorZero, shapeTensorSize, shapeTensorStride})
    {
        layerSlice->setInput(i++, *a);
    }

    RETURN_FIRST_OUTPUT(layerSlice);
}

DEFINE_BUILTIN_OP_IMPORTER(Flatten)
{
    OnnxAttrs attrs(node, ctx);
    int axis = attrs.get("axis", 1);
    nvinfer1::Dims dims = inputs.at(0).shape();
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    int dim0 = 1;
    int dim1 = 1;
    for (int i = 0; i < axis; i++)
    {
        dim0 *= dims.d[i];
    }
    for (int i = axis; i < dims.nbDims; i++)
    {
        dim1 *= dims.d[i];
    }
    nvinfer1::Dims new_shape{2, {dim0, dim1}};
    tensor_ptr = reshapeTensor(ctx, *tensor_ptr, new_shape);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    return {{tensor_ptr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Gather)
{
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor& indices = convertToTensor(inputs.at(1), ctx);
    OnnxAttrs attrs(node, ctx);
    int axis = attrs.get<int>("axis", 0);
    int nbDims = inputs.at(0).shape().nbDims;
    TRT_CHECK(convertAxis(axis, nbDims));
    LOG_VERBOSE("Using Gather axis: " << axis);
    RETURN_FIRST_OUTPUT(ctx->network()->addGather(data, indices, axis));
}

DEFINE_BUILTIN_OP_IMPORTER(Gemm)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get("alpha", 1.f);
    float beta = attrs.get("beta", 1.f);
    bool transA = attrs.get("transA", false);
    bool transB = attrs.get("transB", false);
    nvinfer1::ITensor& inputA = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* inputB = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor& inputC = convertToTensor(inputs.at(2), ctx);

    // Use FC if it is likely to be faster - which is usually when no Shuffles are required.
    bool canUseFC = inputs.at(0).is_tensor() && inputs.at(1).is_weights() && inputs.at(2).is_weights() && alpha == 1.f
        && beta == 1.f && inputs.at(0).tensor().getDimensions().nbDims == 3 && inputs.at(1).weights().shape.nbDims == 2
        && inputs.at(2).weights().shape.nbDims == 1;
    if (canUseFC)
    {
        LOG_VERBOSE("GEMM: using FC layer instead of MM because all criteria were met.");
        nvinfer1::ITensor& tensor = inputs.at(0).tensor();
        ShapedWeights weights = inputs.at(1).weights();
        if (!transB)
        {
            auto transposedWeights = ctx->createTempWeights(weights.type, weights.shape);
            ASSERT(transposeWeights(weights, {1, 0}, &transposedWeights), ErrorCode::kUNSUPPORTED_NODE);
            weights = transposedWeights;
        }
        ShapedWeights biases = inputs.at(2).weights();
        RETURN_FIRST_OUTPUT(ctx->network()->addFullyConnected(tensor, biases.shape.d[0], weights, biases));
    }

    // If input B is a constant, we transpose at parse time if necessary,
    // because In some cases, A * Bt is much slower than A * B.
    if (inputs.at(1).is_weights())
    {
        ShapedWeights weights = inputs.at(1).weights();
        if (transB)
        {
            auto transposedWeights = ctx->createTempWeights(weights.type, weights.shape);
            ASSERT(transposeWeights(weights, {1, 0}, &transposedWeights), ErrorCode::kUNSUPPORTED_NODE);
            weights = transposedWeights;
            // Since we've already transposed now, we can set transpose to false.
            transB = false;
        }
        nvinfer1::IConstantLayer* weightsLayer
            = ctx->network()->addConstant(weights.shape, static_cast<nvinfer1::Weights>(weights));
        inputB = weightsLayer->getOutput(0);
    }
    else
    {
        inputB = &inputs.at(1).tensor();
    }

    LOG_VERBOSE("GEMM: A: " << inputA.getDimensions() << ", B: " << inputB->getDimensions()
                            << ", C: " << inputC.getDimensions());
    nvinfer1::ITensor* inputASqueezed = &inputA;
    nvinfer1::Dims newDims = squeeze_trailing_dims(inputA.getDimensions());
    // When A has more than 2 dimensions, it needs to be flattened.
    if (newDims.nbDims > 2)
    {
        newDims = nvinfer1::Dims{1, {-1}};
    }
    // Due to other TRT layers, inputA may sometimes have trailing 1s that need to be removed.
    if (newDims.nbDims < inputA.getDimensions().nbDims)
    {
        nvinfer1::IShuffleLayer* squeeze = ctx->network()->addShuffle(inputA);
        squeeze->setReshapeDimensions(newDims);
        inputASqueezed = squeeze->getOutput(0);
    }

    constexpr auto getMatrixOp = [](const nvinfer1::ITensor& input, bool transpose) {
        if (input.getDimensions().nbDims == 1)
        {
            return nvinfer1::MatrixOperation::kVECTOR;
        }
        else if (transpose)
        {
            return nvinfer1::MatrixOperation::kTRANSPOSE;
        }
        return nvinfer1::MatrixOperation::kNONE;
    };

    nvinfer1::MatrixOperation opA = getMatrixOp(*inputASqueezed, transA);
    nvinfer1::MatrixOperation opB = getMatrixOp(*inputB, transB);

    LOG_VERBOSE("Using opA: " << static_cast<int>(opA) << " opB: " << static_cast<int>(opB));
    LOG_VERBOSE("GEMM: A, after squeezing: " << inputASqueezed->getDimensions());

    nvinfer1::IMatrixMultiplyLayer* matmul = ctx->network()->addMatrixMultiply(*inputASqueezed, opA, *inputB, opB);
    nvinfer1::ITensor* matmulTensor = matmul->getOutput(0);

    // Scale A*B if needed.
    if (alpha != 1.f)
    {
        nvinfer1::IConstantLayer* alphaConstant = addConstantScalar(ctx, alpha, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        nvinfer1::ITensor* alphaConstantTensor = alphaConstant->getOutput(0);
        broadcastTensors(ctx, alphaConstantTensor, matmulTensor);
        nvinfer1::IElementWiseLayer* scaledMatmul = ctx->network()->addElementWise(
            *alphaConstantTensor, *matmulTensor, nvinfer1::ElementWiseOperation::kPROD);
        matmulTensor = scaledMatmul->getOutput(0);
    }
    // Scale C if needed.
    nvinfer1::ITensor* biasTensor = &inputC;

    if (beta != 1.f)
    {
        nvinfer1::IConstantLayer* betaConstant = addConstantScalar(ctx, beta, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        nvinfer1::ITensor* betaConstantTensor = betaConstant->getOutput(0);
        broadcastTensors(ctx, betaConstantTensor, biasTensor);
        nvinfer1::IElementWiseLayer* scaledBias
            = ctx->network()->addElementWise(*betaConstantTensor, *biasTensor, nvinfer1::ElementWiseOperation::kPROD);
        biasTensor = scaledBias->getOutput(0);
    }
    // A*B may be lower rank than C in TRT, so need to squeeze C.
    if (ctx->getOpsetVersion() < 7 && !attrs.get("broadcast", false))
    {
        nvinfer1::Dims squeezeDims = squeeze_leading_dims(biasTensor->getDimensions());
        biasTensor = reshapeTensor(ctx, *biasTensor, squeezeDims);
    }
    broadcastTensors(ctx, matmulTensor, biasTensor);
    nvinfer1::IElementWiseLayer* biasAdd
        = ctx->network()->addElementWise(*matmulTensor, *biasTensor, nvinfer1::ElementWiseOperation::kSUM);
    return {{biasAdd->getOutput(0)}};
}

DEFINE_BUILTIN_OP_IMPORTER(GlobalAveragePool)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = tensor.getDimensions();
    ASSERT(dims.nbDims == 4, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::DimsHW kernel_size(dims.d[2], dims.d[3]);
    RETURN_FIRST_OUTPUT(ctx->network()->addPooling(tensor, nvinfer1::PoolingType::kAVERAGE, kernel_size));
}

// TODO: GlobalLpPool: pow(reduce_mean(pow(abs(x), p)), 1./p)

DEFINE_BUILTIN_OP_IMPORTER(GlobalMaxPool)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = tensor.getDimensions();
    ASSERT(dims.nbDims == 4, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::DimsHW kernel_size(dims.d[2], dims.d[3]);
    RETURN_FIRST_OUTPUT(ctx->network()->addPooling(tensor, nvinfer1::PoolingType::kMAX, kernel_size));
}

DEFINE_BUILTIN_OP_IMPORTER(Greater)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kGREATER);
}

DEFINE_BUILTIN_OP_IMPORTER(GRU)
{
    using nvinfer1::Dims;
    using nvinfer1::Dims3;
    using mOp = nvinfer1::MatrixOperation;
    using eOp = nvinfer1::ElementWiseOperation;
    using trtAct = nvinfer1::ActivationType;
    nvinfer1::INetworkDefinition* net = ctx->network();

    OnnxAttrs attrs{node, ctx};
    constexpr int NUM_GATES = 3;
    const std::string direction = attrs.get<std::string>("direction", "forward");
    const int numDirections = (direction == "bidirectional") ? 2 : 1;
    const int hiddenSize = attrs.get<int>("hidden_size");
    const int linearBeforeReset = attrs.get<int>("linear_before_reset", 0);
    const float clip = attrs.get("clip", -1.f); // Clipping cannot be negative, so -1.0 is a good sentinel value.

    ASSERT(clip == -1.f && "Clipping is unsupported in the GRU converter", ErrorCode::kUNSUPPORTED_NODE);

    // The input is in SBE format
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor& weights = convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor& recurrenceWeights = convertToTensor(inputs.at(2), ctx);

    constexpr int NUM_ACTIVATIONS = 2;
    std::vector<trtAct> defaultActs{trtAct::kSIGMOID, trtAct::kTANH};
    if (numDirections == 2)
    {
        defaultActs.insert(defaultActs.end(), {trtAct::kSIGMOID, trtAct::kTANH});
    }
    std::vector<trtAct> activations = attrs.get<std::vector<trtAct>>("activations", defaultActs);

    std::vector<float> activationAlphas = attrs.get<std::vector<float>>("activation_alpha", std::vector<float>{});
    std::transform(activations.begin() + activationAlphas.size(), activations.end(), std::back_inserter(activationAlphas), &getActivationDefaultAlpha);

    std::vector<float> activationBetas = attrs.get<std::vector<float>>("activation_beta", std::vector<float>{});
    std::transform(activations.begin() + activationBetas.size(), activations.end(), std::back_inserter(activationBetas), &getActivationDefaultBeta);

    // TODO: Support cases where in bidirectional GRUs, activations of reverse iteration do not match forward pass.
    // TODO: This will require splitting the input tensor in the loop when applying activations.
    if (numDirections == 2)
    {
        ASSERT(std::equal(activations.begin(), activations.begin() + NUM_ACTIVATIONS, activations.begin() + NUM_ACTIVATIONS) && "The parser does not currently support cases where activations for the reverse pass of the GRU do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(std::equal(activationAlphas.begin(), activationAlphas.begin() + NUM_ACTIVATIONS, activationAlphas.begin() + NUM_ACTIVATIONS) && "The parser does not currently support cases where activations for the reverse pass of the GRU do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(std::equal(activationBetas.begin(), activationBetas.begin() + NUM_ACTIVATIONS, activationBetas.begin() + NUM_ACTIVATIONS) && "The parser does not currently support cases where activations for the reverse pass of the GRU do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
    }

    // Need to split weights/biases into ZR gates and H gate, because h(t) computations depend on z(t) and r(t).
    nvinfer1::ITensor* numDirectionsTensor = addConstantScalar(ctx, numDirections, ::ONNX_NAMESPACE::TensorProto::INT32, Dims{1, 1})->getOutput(0);
    nvinfer1::ITensor* hiddenSizeTensor = addConstantScalar(ctx, hiddenSize, ::ONNX_NAMESPACE::TensorProto::INT32, Dims{1, 1})->getOutput(0);
    nvinfer1::ITensor* hiddenSizeDoubledTensor = addConstantScalar(ctx, 2 * hiddenSize, ::ONNX_NAMESPACE::TensorProto::INT32, Dims{1, 1})->getOutput(0);
    nvinfer1::ITensor* eDimTensor = getAxisLength(ctx, input, 2, Dims{1, 1});

    nvinfer1::ITensor* weightsZRStart = addConstant(ctx, std::vector<int32_t>{0, 0, 0}, ::ONNX_NAMESPACE::TensorProto::INT32, Dims{1, 3})->getOutput(0);
    nvinfer1::ITensor* weightsZRSize = net->addConcatenation(
        std::array<nvinfer1::ITensor*, 3>{{
            numDirectionsTensor, hiddenSizeDoubledTensor, eDimTensor
        }}.data(), 3)->getOutput(0);
    nvinfer1::ISliceLayer* weightsZRLayer = net->addSlice(weights, Dims{3}, Dims{3}, Dims3{1, 1, 1});
    weightsZRLayer->setInput(1, *weightsZRStart);
    weightsZRLayer->setInput(2, *weightsZRSize);
    nvinfer1::ITensor* weightsZR = weightsZRLayer->getOutput(0);
    LOG_VERBOSE("Weights for ZR gates shape is: " << weightsZR->getDimensions());

    nvinfer1::ITensor* weightsHStart = addConstant(ctx, std::vector<int32_t>{0, 2 * hiddenSize, 0}, ::ONNX_NAMESPACE::TensorProto::INT32, Dims{1, 3})->getOutput(0);
    nvinfer1::ITensor* weightsHSize = net->addConcatenation(
        std::array<nvinfer1::ITensor*, 3>{{
            numDirectionsTensor, hiddenSizeTensor, eDimTensor
        }}.data(), 3)->getOutput(0);
    nvinfer1::ISliceLayer* weightsHLayer = net->addSlice(weights, Dims{3}, Dims{3}, Dims3{1, 1, 1});
    weightsHLayer->setInput(1, *weightsHStart);
    weightsHLayer->setInput(2, *weightsHSize);
    nvinfer1::ITensor* weightsH = weightsHLayer->getOutput(0);
    LOG_VERBOSE("Weights for H gate shape is: " << weightsH->getDimensions());

    nvinfer1::ITensor* recurrenceWeightsZR = net->addSlice(recurrenceWeights, Dims3{0, 0, 0}, Dims3{numDirections, 2 * hiddenSize, hiddenSize}, Dims3{1, 1, 1})->getOutput(0);
    LOG_VERBOSE("Recurrence weights for ZR gates shape is: " << recurrenceWeightsZR->getDimensions());
    nvinfer1::ITensor* recurrenceWeightsH = net->addSlice(recurrenceWeights, Dims3{0, 2 * hiddenSize, 0}, Dims3{numDirections, hiddenSize, hiddenSize}, Dims3{1, 1, 1})->getOutput(0);
    LOG_VERBOSE("Recurrence weights for H gate shape is: " << recurrenceWeightsH->getDimensions());

    // bias/recurrenceBias will have shape (numDirections, NUM_GATES * hiddenSize)
    nvinfer1::ITensor* biasZR{nullptr};
    nvinfer1::ITensor* biasH{nullptr};
    nvinfer1::ITensor* recurrenceBiasZR{nullptr};
    nvinfer1::ITensor* recurrenceBiasH{nullptr};
    if (inputs.size() > 3 && inputs.at(3))
    {
        // ONNX bias is a concatenation of Wb and Rb on the second axis, so has shape (numDirections, 2 * NUM_GATES * hiddenSize)
        // Unsqueeze so we can broadcast later
        nvinfer1::ITensor* concatenatedBias = &convertToTensor(inputs.at(3), ctx);
        nvinfer1::IShuffleLayer* unsqueeze = net->addShuffle(*concatenatedBias);
        unsqueeze->setReshapeDimensions(Dims3{1, numDirections, 2 * NUM_GATES * hiddenSize});
        concatenatedBias = unsqueeze->getOutput(0);

        biasZR = net->addSlice(*concatenatedBias, Dims3{0, 0, 0}, Dims3{1, numDirections, 2 * hiddenSize}, Dims3{1, 1, 1})->getOutput(0);
        LOG_VERBOSE("Bias for ZR gates shape is: " << biasZR->getDimensions());
        biasH = net->addSlice(*concatenatedBias, Dims3{0, 0, 2 * hiddenSize}, Dims3{1, numDirections, hiddenSize}, Dims3{1, 1, 1})->getOutput(0);
        LOG_VERBOSE("Bias for H gate shape is: " << biasH->getDimensions());

        recurrenceBiasZR = net->addSlice(*concatenatedBias, Dims3{0, 0, NUM_GATES * hiddenSize}, Dims3{1, numDirections, 2 * hiddenSize}, Dims3{1, 1, 1})->getOutput(0);
        LOG_VERBOSE("Recurrence bias for ZR gates shape is: " << recurrenceBiasZR->getDimensions());
        recurrenceBiasH = net->addSlice(*concatenatedBias, Dims3{0, 0, (NUM_GATES + 2) * hiddenSize}, Dims3{1, numDirections, hiddenSize}, Dims3{1, 1, 1})->getOutput(0);
        LOG_VERBOSE("Recurrence bias for H gate shape is: " << recurrenceBiasH->getDimensions());
    }

    // TODO: Add support for clipping.

    // Get a shape tensor containing: (numDirections, batchSize, hiddenSize)
    const auto initialStateShape = [&ctx, &numDirections, &hiddenSize, &input, &net] () -> nvinfer1::ITensor*
    {
        // Get batchSize from input shape
        nvinfer1::ITensor* numDirectionsTensor = addConstantScalar(ctx, numDirections, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, Dims{1, 1})->getOutput(0);
        LOG_VERBOSE("numDirections is: " << numDirections << ", numDirections Tensor shape: " << numDirectionsTensor->getDimensions());
        nvinfer1::ITensor* hiddenSizeTensor = addConstantScalar(ctx, hiddenSize, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, Dims{1, 1})->getOutput(0);
        LOG_VERBOSE("hiddenSize is: " << hiddenSize << ", hiddenSizeTensor shape: " << hiddenSizeTensor->getDimensions());
        nvinfer1::ITensor* batchSizeTensor = getAxisLength(ctx, input, 1, Dims{1, 1});
        LOG_VERBOSE("batchSizeTensor shape: " << batchSizeTensor->getDimensions());

        nvinfer1::IConcatenationLayer* concatenatedShape = net->addConcatenation(
            std::array<nvinfer1::ITensor*, 3>{{
                numDirectionsTensor, batchSizeTensor, hiddenSizeTensor
            }}.data(), 3);
        return concatenatedShape->getOutput(0);
    };
    nvinfer1::ITensor* gateOutputShape = initialStateShape();
    LOG_VERBOSE("Gate output rank (equal to initial hidden/cell state rank): " << gateOutputShape->getDimensions());

    LOG_VERBOSE("Entering Loop");
    // Scan over the S dimension of the input
    auto loop = net->addLoop();
    nvinfer1::ITensor* tripLimit = getAxisLength(ctx, input, 0);
    loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);

    // Unsqueeze the provided iterator output to (1, B, E).
    const auto unsqueezeIterator = [&ctx, &net] (nvinfer1::ITensor* iterator)
    {
        nvinfer1::IShuffleLayer* unsqueeze = net->addShuffle(*iterator);
        // Since we need to copy dimensions, use a hack of reshaping to (B, E, 1) then permuting to (1, B, E)
        unsqueeze->setReshapeDimensions(Dims3{0, 0, 1});
        unsqueeze->setSecondTranspose(nvinfer1::Permutation{2, 0, 1});
        LOG_VERBOSE("Permuted forward iterator to shape: " << unsqueeze->getOutput(0)->getDimensions());
        return unsqueeze->getOutput(0);
    };

    // Add X(t)
    // In the forward/reverse cases, we only use a single iterator. In the bidirectional case, a forward and reverse iterator must be concatenated.
    nvinfer1::ITensor* iterationInput{nullptr};
    if (direction == "forward")
    {
        iterationInput = unsqueezeIterator(loop->addIterator(*input)->getOutput(0));
    }
    else if (direction == "reverse")
    {
         nvinfer1::IIteratorLayer* reverseIterator = loop->addIterator(*input);
         reverseIterator->setReverse(true);
         iterationInput = unsqueezeIterator(reverseIterator->getOutput(0));
    }
    else
    {
        ASSERT(direction == "bidirectional", ErrorCode::kINVALID_NODE);
        nvinfer1::IIteratorLayer* forward = loop->addIterator(*input);
        nvinfer1::IIteratorLayer* reverse = loop->addIterator(*input);
        reverse->setReverse(true);
        // Stack on the 0th axis to create a (numDirections, B, E) tensor.
        nvinfer1::IConcatenationLayer* concat = net->addConcatenation(
            std::array<nvinfer1::ITensor*, 2>{{
                unsqueezeIterator(forward->getOutput(0)),
                unsqueezeIterator(reverse->getOutput(0))
            }}.data(), 2);
        concat->setAxis(0);
        iterationInput = concat->getOutput(0);
    }
    LOG_VERBOSE("Input shape: " << iterationInput->getDimensions());

    // H(t-1)
    const auto getInitialInputValue = [&ctx, &gateOutputShape, &inputs] (size_t inputIdx) -> nvinfer1::ITensor*
    {
        if (inputs.size() > inputIdx && inputs.at(inputIdx))
        {
            return &convertToTensor(inputs.at(inputIdx), ctx);
        }
        return constantOfShape(ctx, addConstantScalar(ctx, 0.f, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, Dims{1, 1})->getOutput(0), gateOutputShape);
    };

    nvinfer1::ITensor* initialHidden = getInitialInputValue(5);
    LOG_VERBOSE("Initial hidden state shape: " << initialHidden->getDimensions());

    nvinfer1::IRecurrenceLayer* Ht1 = loop->addRecurrence(*initialHidden);
    LOG_VERBOSE("Hidden state shape: " << Ht1->getOutput(0)->getDimensions());

    // Compute stackedZR(t) = f(X(t) * W[zr]^T + H(t-1) * R[zr]^T + (Wb[zr] + Rb[zr])). stackedZR(t) has shape (numDirections, batchSize, 2 * hiddenSize)
    nvinfer1::ITensor* xtWTZR = net->addMatrixMultiply(*iterationInput, mOp::kNONE, *weightsZR, mOp::kTRANSPOSE)->getOutput(0);
    LOG_VERBOSE("X(t) * W[zr]^T -> " << xtWTZR->getDimensions());

    nvinfer1::ITensor* ht1RT = net->addMatrixMultiply(*Ht1->getOutput(0), mOp::kNONE, *recurrenceWeightsZR, mOp::kTRANSPOSE)->getOutput(0);
    LOG_VERBOSE("H(t-1) * R[zr]^T -> " << ht1RT->getDimensions());

    nvinfer1::ITensor* stackedZRt = net->addElementWise(*xtWTZR, *ht1RT, eOp::kSUM)->getOutput(0);
    if (biasZR && recurrenceBiasZR)
    {
        stackedZRt = net->addElementWise(*stackedZRt, *biasZR, eOp::kSUM)->getOutput(0);
        stackedZRt = net->addElementWise(*stackedZRt, *recurrenceBiasZR, eOp::kSUM)->getOutput(0);
    }

    nvinfer1::IActivationLayer* stackedZRtLayer = net->addActivation(*stackedZRt, activations.at(0));
    stackedZRtLayer->setAlpha(activationAlphas.at(0));
    stackedZRtLayer->setBeta(activationBetas.at(0));
    stackedZRt = stackedZRtLayer->getOutput(0);
    LOG_VERBOSE("stackedZR(t) -> " << stackedZRt->getDimensions());


    const auto isolateGate = [&ctx, &hiddenSize, &gateOutputShape, &net] (nvinfer1::ITensor* gates, int gateIndex) -> nvinfer1::ITensor*
    {
        nvinfer1::ISliceLayer* isolateGate = net->addSlice(*gates, Dims3{0, 0, 0}, Dims3{0, 0, 0}, Dims3{1, 1, 1});
        isolateGate->setInput(1, *addConstant(ctx, std::vector<int>{0, 0, gateIndex * hiddenSize}, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, Dims{1, 3})->getOutput(0)); // Start
        isolateGate->setInput(2, *gateOutputShape); // Size
        return isolateGate->getOutput(0);
    };

    // zt = stackedZRt[:, :. 0:H]
    nvinfer1::ITensor* zt = isolateGate(stackedZRt, 0);
    LOG_VERBOSE("z(t) -> " << zt->getDimensions());

    // rt = stackedZRt[:, :. H:2H]
    nvinfer1::ITensor* rt = isolateGate(stackedZRt, 1);
    LOG_VERBOSE("r(t) -> " << rt->getDimensions());

    // Compute h(t)
    nvinfer1::ITensor* ht{nullptr};
    // xtWTH = X(t) * (W[h]^T)
    nvinfer1::ITensor* xtWTH = net->addMatrixMultiply(*iterationInput, mOp::kNONE, *weightsH, mOp::kTRANSPOSE)->getOutput(0);
    if (linearBeforeReset == 0)
    {
        // h(t) = g(xtWTH + (r(t) . H(t-1)) * (R[h]^T) + Rb[h] + Wb[h])
        // rtHt1 = (r(t) . H(t-1))
        nvinfer1::ITensor* rtHt1 = net->addElementWise(*rt, *Ht1->getOutput(0), eOp::kPROD)->getOutput(0);
        // rtHt1Rh = (r(t) . H(t-1)) * (R[h]^T)
        nvinfer1::ITensor* rtHt1Rh = net->addMatrixMultiply(*rtHt1, mOp::kNONE, *recurrenceWeightsH, mOp::kTRANSPOSE)->getOutput(0);

        // (xtWTH + rtHt1Rh) + (Rb[h] + Wb[h])
        nvinfer1::ITensor* actInput = net->addElementWise(*xtWTH, *rtHt1Rh, eOp::kSUM)->getOutput(0);

        // If bias is defines, both recurrence and normal bias must be present
        if (recurrenceBiasH && biasH)
        {
            nvinfer1::ITensor* secondSum = net->addElementWise(*recurrenceBiasH, *biasH, eOp::kSUM)->getOutput(0);
            actInput = net->addElementWise(*actInput, *secondSum, eOp::kSUM)->getOutput(0);
        }

        nvinfer1::IActivationLayer* htLayer = net->addActivation(*actInput, activations.at(1));
        htLayer->setAlpha(activationAlphas.at(1));
        htLayer->setBeta(activationBetas.at(1));
        ht = htLayer->getOutput(0);
    }
    else
    {
        // h(t) = g(xtWTH + (r(t) . (H(t-1) * (R[h]^T) + Rb[h])) + Wb[h])
        // ht1Rh = H(t-1) * (R[h]^T)
        nvinfer1::ITensor* ht1Rh = net->addMatrixMultiply(*Ht1->getOutput(0), mOp::kNONE, *recurrenceWeightsH, mOp::kTRANSPOSE)->getOutput(0);

        // rtHtRhRbh = r(t) . (ht1Rh + Rb[h])
        if (recurrenceBiasH)
        {
            ht1Rh = net->addElementWise(*ht1Rh, *recurrenceBiasH, eOp::kSUM)->getOutput(0);
        }
        nvinfer1::ITensor* rtHtRhRbh = net->addElementWise(*rt, *ht1Rh, eOp::kPROD)->getOutput(0);

        // h(t) = g(xtWTH + rtHtRhRbh + Wb[h])
        if (biasH)
        {
            rtHtRhRbh = net->addElementWise(*rtHtRhRbh, *biasH, eOp::kSUM)->getOutput(0);
        }
        nvinfer1::IActivationLayer* htLayer = net->addActivation(*net->addElementWise(*xtWTH, *rtHtRhRbh, eOp::kSUM)->getOutput(0), activations.at(1));
        htLayer->setAlpha(activationAlphas.at(1));
        htLayer->setBeta(activationBetas.at(1));
        ht = htLayer->getOutput(0);
    }
    LOG_VERBOSE("h(t) -> " << ht->getDimensions());

    // H(t) = (1 - z(t)) . h(t) + (z(t) . H(t-1))
    nvinfer1::ITensor* Ht = net->addElementWise(
        *net->addElementWise(
            *net->addElementWise(
                *addConstantScalar(ctx, 1.f, ::ONNX_NAMESPACE::TensorProto::FLOAT, Dims3{1, 1, 1})->getOutput(0),
                *zt,
                eOp::kSUB
            )->getOutput(0),
            *ht,
            eOp::kPROD
        )->getOutput(0),
        *net->addElementWise(*zt, *Ht1->getOutput(0), eOp::kPROD)->getOutput(0),
        eOp::kSUM
    )->getOutput(0);

    Ht1->setInput(1, *Ht);
    LOG_VERBOSE("H(t) -> " << Ht->getDimensions());

    std::vector<TensorOrWeights> outputs{};
    // Y = concatenation of all H(t) for each element of the sequence
    nvinfer1::ILoopOutputLayer* scanOut = loop->addLoopOutput(*Ht, nvinfer1::LoopOutput::kCONCATENATE, 0);
    scanOut->setInput(1, *getAxisLength(ctx, input, 0));
    outputs.emplace_back(scanOut->getOutput(0));
    // Yh = last value of H(t)
    outputs.emplace_back(loop->addLoopOutput(*Ht1->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0));
    return {{outputs}};
}


DEFINE_BUILTIN_OP_IMPORTER(HardSigmoid)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha", 0.2f);
    float beta = attrs.get<float>("beta", 0.5f);
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kHARD_SIGMOID, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Identity)
{
    RETURN_FIRST_OUTPUT(ctx->network()->addIdentity(convertToTensor(inputs.at(0), ctx)));
}

DEFINE_BUILTIN_OP_IMPORTER(ImageScaler)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs{node, ctx};
    // Shift the input by a per-channel bias value.
    std::vector<float> biases = attrs.get<std::vector<float>>("bias");
    nvinfer1::Dims dims{1, static_cast<int>(biases.size())};
    ShapedWeights shiftWeights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, dims);
    std::copy(biases.begin(), biases.end(), static_cast<float*>(shiftWeights.values));
    // Scale is applied to every element of the input, but we need to duplicate it over every channel.
    float scale = attrs.get<float>("scale", 1.0f);
    ShapedWeights scaleWeights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, dims);
    std::fill(static_cast<float*>(scaleWeights.values), static_cast<float*>(scaleWeights.values) + scaleWeights.count(),
        scale);
    // Finally add the scale layer.
    RETURN_FIRST_OUTPUT(ctx->network()->addScale(
        tensor, nvinfer1::ScaleMode::kCHANNEL, shiftWeights, scaleWeights, nvinfer1::Weights{}));
}

DEFINE_BUILTIN_OP_IMPORTER(InstanceNormalization)
{
    // Scales and biases must be initializers
    ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    auto scale_weights = inputs.at(1).weights();
    auto bias_weights = inputs.at(2).weights();
    OnnxAttrs attrs(node, ctx);
    float epsilon = attrs.get("epsilon", 1e-5f);
    // TensorRT only supports epsilon values >= 1e-4.
    epsilon = std::max(epsilon, 1e-4f);

    // Populate instanceNormalization plugin properties.
    const std::string pluginName = "InstanceNormalization_TRT";
    const std::string pluginVersion = "001";
    std::vector<nvinfer1::PluginField> f;
    f.emplace_back("epsilon", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1);
    f.emplace_back("scales", scale_weights.values, nvinfer1::PluginFieldType::kFLOAT32, scale_weights.count());
    f.emplace_back("bias", bias_weights.values, nvinfer1::PluginFieldType::kFLOAT32, bias_weights.count());

    // Create plugin from registry
    nvinfer1::IPluginV2* plugin = importPluginFromRegistry(ctx, pluginName, pluginVersion, node.name(), f);

    ASSERT(plugin != nullptr && "InstanceNormalization plugin was not found in the plugin registry!", ErrorCode::kUNSUPPORTED_NODE);

    RETURN_FIRST_OUTPUT(ctx->network()->addPluginV2(&tensor_ptr, 1, *plugin));
}

DEFINE_BUILTIN_OP_IMPORTER(LeakyRelu)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha", 0.01f);
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kLEAKY_RELU, &alpha);
}

DEFINE_BUILTIN_OP_IMPORTER(Less)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kLESS);
}

DEFINE_BUILTIN_OP_IMPORTER(Log)
{
    return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kLOG);
}

DECLARE_BUILTIN_OP_IMPORTER(Softmax);
DEFINE_BUILTIN_OP_IMPORTER(LogSoftmax)
{
    auto result = importSoftmax(ctx, node, inputs);
    if (result.is_error())
    {
        return result;
    }
    else
    {
        auto& input = result.value().at(0);
        return apply_unary_function(ctx, input, nvinfer1::UnaryOperation::kLOG);
    }
}

DEFINE_BUILTIN_OP_IMPORTER(LRN)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs(node, ctx);
    int size = attrs.get<int>("size");
    float alpha = attrs.get<float>("alpha", 0.0001);
    float beta = attrs.get<float>("beta", 0.75);
    float bias = attrs.get<float>("bias", 1.0);
    RETURN_FIRST_OUTPUT(ctx->network()->addLRN(tensor, size, alpha, beta, bias));
}

DEFINE_BUILTIN_OP_IMPORTER(LSTM)
{
    using trtAct = nvinfer1::ActivationType;

    OnnxAttrs attrs{node, ctx};
    constexpr int NUM_GATES = 4;
    const std::string direction = attrs.get<std::string>("direction", "forward");
    const int numDirections = (direction == "bidirectional") ? 2 : 1;
    const int hiddenSize = attrs.get<int>("hidden_size");
    const int inputForget = attrs.get("input_forget", 0);
    const float clip = attrs.get("clip", -1.f); // Clipping cannot be negative, so -1.0 is a good sentinel value.

    ASSERT(inputForget == 0 && "Coupled input/forget is unsupported in the LSTM converter", ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(!(inputs.size() > 7 && inputs.at(7)) && "Peephole connections are currently unsupported in the LSTM converter", ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(clip == -1.f && "Clipping is unsupported in the LSTM converter", ErrorCode::kUNSUPPORTED_NODE);

    // The input is in SBE format
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* weights = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor* recurrenceWeights = &convertToTensor(inputs.at(2), ctx);


    constexpr int NUM_ACTIVATIONS = 3;
    std::vector<trtAct> defaultActs{trtAct::kSIGMOID, trtAct::kTANH, trtAct::kTANH};
    if (numDirections == 2)
    {
        defaultActs.insert(defaultActs.end(), {trtAct::kSIGMOID, trtAct::kTANH, trtAct::kTANH});
    }
    std::vector<trtAct> activations = attrs.get<std::vector<trtAct>>("activations", defaultActs);

    std::vector<float> activationAlphas = attrs.get<std::vector<float>>("activation_alpha", std::vector<float>{});
    std::transform(activations.begin() + activationAlphas.size(), activations.end(), std::back_inserter(activationAlphas), &getActivationDefaultAlpha);

    std::vector<float> activationBetas = attrs.get<std::vector<float>>("activation_beta", std::vector<float>{});
    std::transform(activations.begin() + activationBetas.size(), activations.end(), std::back_inserter(activationBetas), &getActivationDefaultBeta);

    // TODO: Support cases where in bidirectional LSTMs, activations of reverse iteration do not match forward pass.
    // TODO: This will require splitting the input tensor in the loop when applying activations.
    if (numDirections == 2)
    {
        ASSERT(std::equal(activations.begin(), activations.begin() + NUM_ACTIVATIONS, activations.begin() + NUM_ACTIVATIONS) && "The parser does not currently support cases where activations for the reverse pass of the LSTM do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(std::equal(activationAlphas.begin(), activationAlphas.begin() + NUM_ACTIVATIONS, activationAlphas.begin() + NUM_ACTIVATIONS) && "The parser does not currently support cases where activations for the reverse pass of the LSTM do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(std::equal(activationBetas.begin(), activationBetas.begin() + NUM_ACTIVATIONS, activationBetas.begin() + NUM_ACTIVATIONS) && "The parser does not currently support cases where activations for the reverse pass of the LSTM do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
    }

    // Roll Rb into Wb (and RBb into WBb). Bias is in the form  [Wb[iofc], Rb[iofc], WBb[iofc], RBb[iofc]].
    // So reshape such that we can perform a reduction to add Wb and Rb.
    nvinfer1::ITensor* combinedBias{nullptr};
    if (inputs.size() > 3 && inputs.at(3))
    {
        nvinfer1::ITensor* bias = &convertToTensor(inputs.at(3), ctx);
        LOG_VERBOSE("Bias shape is: " << bias->getDimensions());
        // Reshape to [[Wb[iofc], Rb[iofc]], [WBb[iofc], RBb[iofc]]]
        nvinfer1::IShuffleLayer* reshapeBias = ctx->network()->addShuffle(*bias);
        reshapeBias->setReshapeDimensions(nvinfer1::Dims3{numDirections, 2, NUM_GATES * hiddenSize});
        LOG_VERBOSE("Reshaping bias to: " << reshapeBias->getOutput(0)->getDimensions());
        combinedBias = ctx->network()->addReduce(*reshapeBias->getOutput(0), nvinfer1::ReduceOperation::kSUM, /*axis=*/0b010, /*keepDimensions=*/true)->getOutput(0);
        LOG_VERBOSE("After reduction, bias shape is: " << combinedBias->getDimensions());
    }

    // TODO: Add support for clipping.

    // Get a shape tensor containing: (numDirections, batchSize, hiddenSize)
    const auto initialStateShape = [&ctx, &numDirections, &hiddenSize, &input] () -> nvinfer1::ITensor*
    {
        // Get batchSize from input shape
        nvinfer1::ITensor* numDirectionsTensor = addConstantScalar(ctx, numDirections, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, 1})->getOutput(0);
        LOG_VERBOSE("numDirectionsTensor shape: " << numDirectionsTensor->getDimensions());
        nvinfer1::ITensor* hiddenSizeTensor = addConstantScalar(ctx, hiddenSize, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, 1})->getOutput(0);
        LOG_VERBOSE("hiddenSizeTensor shape: " << hiddenSizeTensor->getDimensions());
        nvinfer1::ITensor* batchSizeTensor = getAxisLength(ctx, input, 1, nvinfer1::Dims{1, 1});
        LOG_VERBOSE("batchSizeTensor shape: " << batchSizeTensor->getDimensions());

        std::array<nvinfer1::ITensor*, 3> tensors{{numDirectionsTensor, batchSizeTensor, hiddenSizeTensor}};
        nvinfer1::IConcatenationLayer* concatenatedShape = ctx->network()->addConcatenation(tensors.data(), 3);
        return concatenatedShape->getOutput(0);
    };
    nvinfer1::ITensor* gateOutputShape = initialStateShape();
    LOG_VERBOSE("Gate output rank (equal to initial hidden/cell state rank): " << gateOutputShape->getDimensions());

    const auto getInitialInputValue = [&ctx, &gateOutputShape, &inputs] (size_t inputIdx) -> nvinfer1::ITensor*
    {
        if (inputs.size() > inputIdx && inputs.at(inputIdx))
        {
            return &convertToTensor(inputs.at(inputIdx), ctx);
        }
        return constantOfShape(ctx, addConstantScalar(ctx, 0.f, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, nvinfer1::Dims{1, 1})->getOutput(0), gateOutputShape);
    };

    nvinfer1::ITensor* initialHidden = getInitialInputValue(5);
    LOG_VERBOSE("Initial hidden state shape: " << initialHidden->getDimensions());

    nvinfer1::ITensor* initialCellState = getInitialInputValue(6);
    LOG_VERBOSE("Initial cell state shape: " << initialCellState->getDimensions());

    LOG_VERBOSE("Entering Loop");
    // Scan over the S dimension of the input
    auto loop = ctx->network()->addLoop();
    nvinfer1::ITensor* tripLimit = getAxisLength(ctx, input, 0);
    loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);

    // Unsqueeze the provided iterator output to (1, B, E).
    const auto unsqueezeIterator = [&ctx] (nvinfer1::ITensor* iterator)
    {
        nvinfer1::IShuffleLayer* unsqueeze = ctx->network()->addShuffle(*iterator);
        // Since we need to copy dimensions, use a hack of reshaping to (B, E, 1) then permuting to (1, B, E)
        unsqueeze->setReshapeDimensions(nvinfer1::Dims3{0, 0, 1});
        unsqueeze->setSecondTranspose(nvinfer1::Permutation{2, 0, 1});
        LOG_VERBOSE("Permuted forward iterator to shape: " << unsqueeze->getOutput(0)->getDimensions());
        return unsqueeze->getOutput(0);
    };

    // Add X(t)
    // In the forward/reverse cases, we only use a single iterator. In the bidirectional case, a forward and reverse iterator must be concatenated.
    nvinfer1::ITensor* iterationInput{nullptr};
    if (direction == "forward")
    {
        iterationInput = unsqueezeIterator(loop->addIterator(*input)->getOutput(0));
    }
    else if (direction == "reverse")
    {
         nvinfer1::IIteratorLayer* reverseIterator = loop->addIterator(*input);
         reverseIterator->setReverse(true);
         iterationInput = unsqueezeIterator(reverseIterator->getOutput(0));
    }
    else
    {
        ASSERT(direction == "bidirectional", ErrorCode::kINVALID_NODE);
        nvinfer1::IIteratorLayer* forward = loop->addIterator(*input);
        nvinfer1::IIteratorLayer* reverse = loop->addIterator(*input);
        reverse->setReverse(true);
        // Stack on the 0th axis to create a (numDirections, B, E) tensor.
        std::array<nvinfer1::ITensor*, 2> tensors{{unsqueezeIterator(forward->getOutput(0)), unsqueezeIterator(reverse->getOutput(0))}};
        nvinfer1::IConcatenationLayer* concat = ctx->network()->addConcatenation(tensors.data(), 2);
        concat->setAxis(0);
        iterationInput = concat->getOutput(0);
    }
    LOG_VERBOSE("Input shape: " << iterationInput->getDimensions());

    // H(t-1)
    nvinfer1::IRecurrenceLayer* hiddenState = loop->addRecurrence(*initialHidden);
    LOG_VERBOSE("Hidden state shape: " << hiddenState->getOutput(0)->getDimensions());

    // C(t-1)
    nvinfer1::IRecurrenceLayer* cellState = loop->addRecurrence(*initialCellState);
    LOG_VERBOSE("Cell state shape: " << cellState->getOutput(0)->getDimensions());

    // Compute intermediate(t) = (X(t) * W^T + H(t-1) * R^T + (Wb + Rb)). intermediate(t) has shape (numDirections, batchSize, 4 * hiddenSize)
    nvinfer1::ITensor* xtWT = ctx->network()->addMatrixMultiply(*iterationInput, nvinfer1::MatrixOperation::kNONE, *weights, nvinfer1::MatrixOperation::kTRANSPOSE)->getOutput(0);
    LOG_VERBOSE("X(t) * W^T -> " << xtWT->getDimensions());

    nvinfer1::ITensor* ht1RT = ctx->network()->addMatrixMultiply(*hiddenState->getOutput(0), nvinfer1::MatrixOperation::kNONE, *recurrenceWeights, nvinfer1::MatrixOperation::kTRANSPOSE)->getOutput(0);
    LOG_VERBOSE("H(t-1) * R^T -> " << ht1RT->getDimensions());

    nvinfer1::ITensor* intermediatet = ctx->network()->addElementWise(*xtWT, *ht1RT, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
    if (combinedBias)
    {
        intermediatet = ctx->network()->addElementWise(*intermediatet, *combinedBias, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
    }
    LOG_VERBOSE("intermediate(t) -> " << intermediatet->getDimensions());


    const auto isolateGate = [&ctx, &hiddenSize, &gateOutputShape] (nvinfer1::ITensor* gates, int gateIndex) -> nvinfer1::ITensor*
    {
        nvinfer1::ISliceLayer* isolateGate = ctx->network()->addSlice(*gates, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{1, 1, 1});
        isolateGate->setInput(1, *addConstant(ctx, std::vector<int>{0, 0, gateIndex * hiddenSize}, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, 3})->getOutput(0)); // Start
        isolateGate->setInput(2, *gateOutputShape); // Size
        return isolateGate->getOutput(0);
    };

    // c(t) = g(intermediate(t)[:, :, 3H:4H])
    nvinfer1::IActivationLayer* gAct = ctx->network()->addActivation(*isolateGate(intermediatet, 3), activations.at(1));
    gAct->setAlpha(activationAlphas.at(1));
    gAct->setBeta(activationBetas.at(1));

    nvinfer1::ITensor* ctGate = gAct->getOutput(0);
    LOG_VERBOSE("c(t) -> " << ctGate->getDimensions());

    nvinfer1::ISliceLayer* isolateIOF = ctx->network()->addSlice(*intermediatet, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{1, 1, 1});
    isolateIOF->setInput(1, *addConstant(ctx, std::vector<int>{0, 0, 0}, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, 3})->getOutput(0)); // Start
    // threeGateShape is (numDirections, batchSize, 3 * hiddenSize)
    nvinfer1::ITensor* threeGateShape = ctx->network()->addElementWise(*gateOutputShape,
        *addConstant(ctx, std::vector<int>{1, 1, 3}, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, 3})->getOutput(0),
        nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
    isolateIOF->setInput(2, *threeGateShape); // Size

    // TODO: Compute peephole connections
    nvinfer1::IActivationLayer* fAct = ctx->network()->addActivation(*isolateIOF->getOutput(0), activations.at(0));
    fAct->setAlpha(activationAlphas.at(0));
    fAct->setBeta(activationBetas.at(0));

    nvinfer1::ITensor* ioftGates = fAct->getOutput(0);
    LOG_VERBOSE("iof(t) -> " << ioftGates->getDimensions());

    nvinfer1::ITensor* itGate = isolateGate(ioftGates, 0);
    nvinfer1::ITensor* otGate = isolateGate(ioftGates, 1);
    nvinfer1::ITensor* ftGate = isolateGate(ioftGates, 2);

    // . represents a hadamard product
    // C(t) = f(t) . C(t - 1) + i(t) . c(t)
    nvinfer1::ITensor* Ct = ctx->network()->addElementWise(
        *ctx->network()->addElementWise(*ftGate, *cellState->getOutput(0), nvinfer1::ElementWiseOperation::kPROD)->getOutput(0),
        *ctx->network()->addElementWise(*itGate, *ctGate, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0),
        nvinfer1::ElementWiseOperation::kSUM
    )->getOutput(0);
    cellState->setInput(1, *Ct);
    LOG_VERBOSE("C(t) -> " << Ct->getDimensions());

    // H(t) = o(t) . h(C(t))
    nvinfer1::IActivationLayer* hAct = ctx->network()->addActivation(*Ct, activations.at(2));
    hAct->setAlpha(activationAlphas.at(2));
    hAct->setBeta(activationBetas.at(2));

    nvinfer1::ITensor* Ht = ctx->network()->addElementWise(
        *otGate,
        *hAct->getOutput(0),
        nvinfer1::ElementWiseOperation::kPROD
    )->getOutput(0);
    hiddenState->setInput(1, *Ht);
    LOG_VERBOSE("H(t) -> " << Ht->getDimensions());

    std::vector<TensorOrWeights> outputs{};
    // Y = concatenation of all H(t) for each element of the sequence
    nvinfer1::ILoopOutputLayer* scanOut = loop->addLoopOutput(*Ht, nvinfer1::LoopOutput::kCONCATENATE, 0);
    scanOut->setInput(1, *getAxisLength(ctx, input, 0));
    outputs.emplace_back(scanOut->getOutput(0));
    // Yh = last value of H(t)
    outputs.emplace_back(loop->addLoopOutput(*hiddenState->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0));
    // Yc = last value of C(t)
    outputs.emplace_back(loop->addLoopOutput(*cellState->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0));

    return {{outputs}};
}

DEFINE_BUILTIN_OP_IMPORTER(MatMul)
{
    nvinfer1::ITensor* inputA = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* inputB = &convertToTensor(inputs.at(1), ctx);

    broadcastTensors(ctx, inputA, inputB);

    constexpr auto getMatrixOp = [](const nvinfer1::ITensor& input) {
        return (input.getDimensions().nbDims == 1) ? nvinfer1::MatrixOperation::kVECTOR
                                                   : nvinfer1::MatrixOperation::kNONE;
    };

    nvinfer1::MatrixOperation opA = getMatrixOp(*inputA);
    nvinfer1::MatrixOperation opB = getMatrixOp(*inputB);

    nvinfer1::IMatrixMultiplyLayer* matmul = ctx->network()->addMatrixMultiply(*inputA, opA, *inputB, opB);
    return {{matmul->getOutput(0)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Max)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kMAX);
}

DEFINE_BUILTIN_OP_IMPORTER(MaxPool)
{
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = tensor_ptr->getDimensions();
    ASSERT(dims.nbDims >= 2, ErrorCode::kINVALID_NODE);

    bool need_to_expand_dims = (dims.nbDims == 3);
    if (need_to_expand_dims)
    {
        // Expand spatial dims from 1D to 2D
        nvinfer1::DimsNCHW new_shape(dims.d[0], dims.d[1], dims.d[2], 1);
        tensor_ptr = reshapeTensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensor_ptr->getDimensions();
    }

    int nbSpatialDims = dims.nbDims - 2;

    // Support for opset10 ceil_mode
    CeilingPoolDim ceilingPool;
    // Ceiling and dialations added in opset 10
    if (ctx->getOpsetVersion() >= 10)
    {
        OnnxAttrs attrs(node, ctx);
        const auto ceil_mode = attrs.get<int>("ceil_mode", 0);
        const auto dilations = attrs.get<std::vector<int>>("dilations", std::vector<int>(2, 1));
        for (size_t i = 0; i < dilations.size(); i++)
            ASSERT(dilations[i] == 1, ErrorCode::kUNSUPPORTED_NODE); // Do not support pooling dilations currently
        if (ceil_mode != 0) // Need to set pooling formula to use ceiling instead of floor
        {
            ctx->network()->setPoolingOutputDimensionsFormula(&ceilingPool);
        }
    }
    ASSERT(nbSpatialDims == 2 || nbSpatialDims == 3, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Dims kernel_size = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims beg_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims end_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::PaddingMode paddingMode;
    bool exclude_padding;
    getKernelParams(ctx, node, &kernel_size, &strides, &beg_padding, &end_padding, paddingMode, exclude_padding);
    nvinfer1::IPoolingLayer* layer
        = ctx->network()->addPoolingNd(*tensor_ptr, nvinfer1::PoolingType::kMAX, kernel_size);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    layer->setStrideNd(strides);
    layer->setPaddingMode(paddingMode);
    layer->setPrePadding(beg_padding);
    layer->setPostPadding(end_padding);
    tensor_ptr = layer->getOutput(0);
    dims = tensor_ptr->getDimensions();

    if (need_to_expand_dims)
    {
        // Un-expand spatial dims back to 1D
        nvinfer1::Dims new_shape{3, {dims.d[0], dims.d[1], dims.d[2]}};
        tensor_ptr = reshapeTensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    }
    return {{tensor_ptr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Mean)
{
    auto sum_result = elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kSUM);
    if (sum_result.is_error())
    {
        return sum_result;
    }
    auto& sum_input = sum_result.value().at(0);
    ASSERT(sum_input.is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor& sum_tensor = sum_input.tensor();

    int ndim = sum_tensor.getDimensions().nbDims;
    float scale_value = 1.f / inputs.size();
    auto scale_dtype = ::ONNX_NAMESPACE::TensorProto::FLOAT;
    auto scale_shape = nvinfer1::Dims{ndim, {1, 1, 1, 1, 1, 1, 1, 1}};
    auto scale_weights = ctx->createTempWeights(scale_dtype, scale_shape);
    static_cast<float*>(scale_weights.values)[0] = scale_value;
    auto* constant_layer = ctx->network()->addConstant(scale_weights.shape, scale_weights);
    ASSERT(constant_layer, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor& scale_constant = *constant_layer->getOutput(0);
    RETURN_FIRST_OUTPUT(
        ctx->network()->addElementWise(sum_tensor, scale_constant, nvinfer1::ElementWiseOperation::kPROD));
}

DEFINE_BUILTIN_OP_IMPORTER(Min)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kMIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Mul)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kPROD);
}

DEFINE_BUILTIN_OP_IMPORTER(Neg)
{
    return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kNEG);
}

DEFINE_BUILTIN_OP_IMPORTER(Not)
{
    return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kNOT);
}

DEFINE_BUILTIN_OP_IMPORTER(Pad)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    nvinfer1::DimsHW beg_padding, end_padding;
    OnnxAttrs attrs(node, ctx);
    auto mode = attrs.get<std::string>("mode", "constant");
    float value = attrs.get<float>("value", 0.f);
    ASSERT(mode == "constant" && value == 0, ErrorCode::kUNSUPPORTED_NODE);
    if (attrs.count("paddings"))
    {
        // TODO: This is a WAR for old versions of ONNX and should be removed in future
        auto onnx_padding = attrs.get<std::vector<int>>("paddings");
        ASSERT(onnx_padding.size() == 8, ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(onnx_padding[0] == 0 && onnx_padding[1] == 0 && onnx_padding[2] == 0 && onnx_padding[3] == 0,
            ErrorCode::kUNSUPPORTED_NODE);
        beg_padding.h() = onnx_padding[4];
        end_padding.h() = onnx_padding[5];
        beg_padding.w() = onnx_padding[6];
        end_padding.w() = onnx_padding[7];
        RETURN_FIRST_OUTPUT(ctx->network()->addPadding(tensor, beg_padding, end_padding));
    }
    auto onnx_padding = attrs.get<std::vector<int>>("pads");
    ASSERT(onnx_padding.size() == 8, ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(onnx_padding[0] == 0 && onnx_padding[1] == 0 && onnx_padding[4] == 0 && onnx_padding[5] == 0,
        ErrorCode::kUNSUPPORTED_NODE);
    beg_padding.h() = onnx_padding[2];
    beg_padding.w() = onnx_padding[3];
    end_padding.h() = onnx_padding[6];
    end_padding.w() = onnx_padding[7];
    RETURN_FIRST_OUTPUT(ctx->network()->addPadding(tensor, beg_padding, end_padding));
}

DEFINE_BUILTIN_OP_IMPORTER(Pow)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kPOW);
}

DEFINE_BUILTIN_OP_IMPORTER(PRelu)
{
    ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    const auto& shape1 = inputs.at(0).shape();
    nvinfer1::ITensor* slopes{};
    if (inputs.at(1).is_tensor())
    {
        if (inputs.at(1).shape().nbDims < shape1.nbDims)
        {
            nvinfer1::IShuffleLayer* reshapeLayer = ctx->network()->addShuffle(inputs.at(1).tensor());
            ASSERT(reshapeLayer, ErrorCode::kUNSUPPORTED_NODE);
            reshapeLayer->setReshapeDimensions(expand_dims(inputs.at(1).shape(), shape1.nbDims));
            slopes = reshapeLayer->getOutput(0);
        }
        else
        {
            slopes = &convertToTensor(inputs.at(1), ctx);
        }
        const auto& shape2 = slopes->getDimensions();
        ASSERT(shape1.nbDims == shape2.nbDims, ErrorCode::kUNSUPPORTED_NODE);
        for (int i = 0; i < shape1.nbDims; ++i)
        {
            ASSERT(shape1.d[i] == shape2.d[i] || shape2.d[i] == 1, ErrorCode::kUNSUPPORTED_NODE);
        }
    }
    else
    {
        auto weights = inputs.at(1).weights();
        if (inputs.at(1).shape().nbDims < shape1.nbDims)
        {
            weights.shape = expand_dims(weights.shape, shape1.nbDims);
        }
        auto constantLayer = ctx->network()->addConstant(weights.shape, weights);
        ASSERT(constantLayer, ErrorCode::kUNSUPPORTED_NODE);
        slopes = constantLayer->getOutput(0);
    }
    ASSERT(input.getType() != nvinfer1::DataType::kINT32, ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(slopes->getType() != nvinfer1::DataType::kINT32, ErrorCode::kUNSUPPORTED_NODE);
    RETURN_FIRST_OUTPUT(ctx->network()->addParametricReLU(input, *slopes));
}

DEFINE_BUILTIN_OP_IMPORTER(Reciprocal)
{
    return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kRECIP);
}

DEFINE_BUILTIN_OP_IMPORTER(ReduceL1)
{
    NodeImportResult abs_result = apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kABS);
    if (abs_result.is_error())
    {
        return abs_result;
    }
    TensorOrWeights abs_input = abs_result.value().at(0);
    return reduceTensor(ctx, node, abs_input, nvinfer1::ReduceOperation::kSUM);
}
DECLARE_BUILTIN_OP_IMPORTER(ReduceSum);
DEFINE_BUILTIN_OP_IMPORTER(ReduceLogSum)
{
    auto sum_result = importReduceSum(ctx, node, inputs);
    if (sum_result.is_error())
    {
        return sum_result;
    }
    TensorOrWeights sum_input = sum_result.value().at(0);
    return apply_unary_function(ctx, sum_input, nvinfer1::UnaryOperation::kLOG);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceLogSumExp)
{
    // TODO: Abstract this sequence with a function or macro
    auto exp_result = apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kEXP);
    if (exp_result.is_error())
    {
        return exp_result;
    }
    auto exp_inputs = exp_result.value();
    return importReduceLogSum(ctx, node, exp_inputs);
}
DECLARE_BUILTIN_OP_IMPORTER(ReduceSumSquare);
DEFINE_BUILTIN_OP_IMPORTER(ReduceL2)
{
    auto sum_sqr_result = importReduceSumSquare(ctx, node, inputs);
    if (sum_sqr_result.is_error())
    {
        return sum_sqr_result;
    }
    TensorOrWeights sum_sqr = sum_sqr_result.value().at(0);
    return apply_unary_function(ctx, sum_sqr, nvinfer1::UnaryOperation::kSQRT);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceMax)
{
    return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kMAX);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceMean)
{
    return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kAVG);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceMin)
{
    return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kMIN);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceProd)
{
    return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kPROD);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceSum)
{
    return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kSUM);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceSumSquare)
{
    nvinfer1::ITensor& tensor = inputs.at(0).tensor();
    auto* sqr_layer = ctx->network()->addElementWise(tensor, tensor, nvinfer1::ElementWiseOperation::kPROD);
    ASSERT(sqr_layer, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* sqr_tensor_ptr = sqr_layer->getOutput(0);
    return reduceTensor(ctx, node, sqr_tensor_ptr, nvinfer1::ReduceOperation::kSUM);
}

DEFINE_BUILTIN_OP_IMPORTER(Relu)
{
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kRELU);
}

DEFINE_BUILTIN_OP_IMPORTER(Reshape)
{
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor& newShape = convertToTensor(inputs.at(1), ctx);

    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(input);
    layer->setInput(1, newShape);

    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(ScaledTanh)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha");
    float beta = attrs.get<float>("beta");
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSCALED_TANH, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Loop)
{
    constexpr int NB_NON_STATE_INPUTS = 2; // First 2 inputs are trip count and condition respectively.
    constexpr int NB_DISCARDED_OUTPUTS = 1; // First output is the updated value of the condition, and is ignored by the outer loop node.
    constexpr int MAX_SCAN_OUTPUT_LENGTH = 1024; // Maximum length for scan outputs if trip count is not set.
    ASSERT(inputs.size() >= 2, ErrorCode::kINVALID_NODE);
    OnnxAttrs attrs(node, ctx);
    const int nbInputs = node.input().size();
    // The number of state variables on the input and output is the same.
    const int nbStateVars = nbInputs - NB_NON_STATE_INPUTS;

    const ::ONNX_NAMESPACE::GraphProto& body = attrs.get<::ONNX_NAMESPACE::GraphProto>("body");

    auto loop = ctx->network()->addLoop();

    // Trip count and condition are optional inputs.
    nvinfer1::ITensor* tripLimit{nullptr};
    if (inputs[0])
    {
        tripLimit = convertToScalar(ctx, &convertToTensor(inputs[0], ctx));
        ASSERT(tripLimit, ErrorCode::kINVALID_NODE);
        loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);
    }
    if (inputs[1])
    {
        nvinfer1::ITensor* cond = convertToScalar(ctx, &convertToTensor(inputs[1], ctx));
        ASSERT(cond, ErrorCode::kINVALID_NODE);
        loop->addTripLimit(*cond, nvinfer1::TripLimit::kWHILE);
    }
    // Add initial state inputs using recurrent layers.
    std::vector<nvinfer1::IRecurrenceLayer*> stateVars{};
    for (size_t i = 2; i < inputs.size(); ++i)
    {
        stateVars.emplace_back(loop->addRecurrence(convertToTensor(inputs[i], ctx)));
        ctx->registerTensor(TensorOrWeights{stateVars.back()->getOutput(0)}, body.input(i).name());
    }

    // Loop body
    TRT_CHECK(onnx2trt::parseGraph(ctx, body));

    // Set final values of state variables.
    std::vector<TensorOrWeights> nodeOutputs{};
    for (int i = 0; i < nbStateVars; ++i)
    {
        // The first output of the body graph is the updated condition, which is ignored by the Loop node.
        const int index = i + NB_DISCARDED_OUTPUTS;
        const auto& bodyOutputName = body.output(index).name();
        auto& stateOutput = convertToTensor(ctx->tensors().at(bodyOutputName), ctx);
        LOG_VERBOSE("For state variable output: " << bodyOutputName << ", found matching tensor: " << stateOutput.getName() << ", with shape: " << stateOutput.getDimensions());
        stateVars.at(i)->setInput(1, stateOutput);
        // Each state variable is also a loop output
        nodeOutputs.emplace_back(loop->addLoopOutput(*stateVars.at(i)->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0));
    }
    // Finally, set up scan outputs.
    for (int i = nbStateVars + NB_DISCARDED_OUTPUTS; i < nbInputs; ++i)
    {
        const auto& bodyOutputName = body.output(i).name();
        auto& scanOutput = convertToTensor(ctx->tensors().at(bodyOutputName), ctx);
        LOG_VERBOSE("For scan output: " << bodyOutputName << ", found matching tensor: " << scanOutput.getName()
            << ", with shape: " << scanOutput.getDimensions());
        nvinfer1::ILoopOutputLayer* trtScanOut = loop->addLoopOutput(scanOutput, nvinfer1::LoopOutput::kCONCATENATE, 0);
        // If trip limit is set, we can set the loop output to the tripLimit, otherwise, set to some dummy constant value.
        // In the latter case, the scan outputs must not be used in the rest of the model.
        if (tripLimit)
        {
            trtScanOut->setInput(1, *tripLimit);
        }
        else
        {
            trtScanOut->setInput(1, *addConstantScalar(ctx, MAX_SCAN_OUTPUT_LENGTH, ::ONNX_NAMESPACE::TensorProto_DataType_INT32)->getOutput(0));
        }
        nodeOutputs.emplace_back(trtScanOut->getOutput(0));
    }

    return {nodeOutputs};
}

DEFINE_BUILTIN_OP_IMPORTER(Scan)
{
    OnnxAttrs attrs(node, ctx);
    const int nbInputs = node.input().size();
    const int nbScanInputs = attrs.get<int>("num_scan_inputs");
    // The number of state variables on the input and output is the same.
    const int nbStateVars = nbInputs - nbScanInputs;
    const int nbScanOutputs = node.output().size() - nbStateVars;

    std::vector<int> defaultScanInputArgs(nbScanInputs);
    std::fill(defaultScanInputArgs.begin(), defaultScanInputArgs.end(), 0);
    const std::vector<int> scanInputAxes(attrs.get("scan_input_axes", defaultScanInputArgs));
    const std::vector<int> scanInputDirections(attrs.get("scan_input_directions", defaultScanInputArgs));

    std::vector<int> defaultScanOutputArgs(nbScanOutputs);
    std::fill(defaultScanOutputArgs.begin(), defaultScanOutputArgs.end(), 0);
    const std::vector<int> scanOutputAxes(attrs.get("scan_output_axes", defaultScanOutputArgs));
    const std::vector<int> scanOutputDirections(attrs.get("scan_output_directions", defaultScanOutputArgs));

    const ::ONNX_NAMESPACE::GraphProto& body = attrs.get<::ONNX_NAMESPACE::GraphProto>("body");

    auto loop = ctx->network()->addLoop();
    // When multiple scan inputs are present, scan behaves like zip, so it is sufficient
    // to use only one scan input to determine trip limit.
    nvinfer1::ITensor* tripLimit = getAxisLength(ctx, &convertToTensor(inputs.back(), ctx), scanInputAxes.back());
    loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);

    // Add initial state inputs using recurrent layers, and scan inputs using iterators.
    std::vector<nvinfer1::IRecurrenceLayer*> stateVars{};
    for (int i = 0; i < nbStateVars; ++i)
    {
        stateVars.emplace_back(loop->addRecurrence(convertToTensor(inputs.at(i), ctx)));
        ctx->registerTensor(TensorOrWeights{stateVars.back()->getOutput(0)}, body.input(i).name());
    }
    for (int i = 0; i < nbScanInputs; ++i)
    {
        const int index = nbStateVars + i; // Scan Inputs are after the state variables.
        nvinfer1::IIteratorLayer* scanInput = loop->addIterator(convertToTensor(inputs.at(index), ctx));
        scanInput->setAxis(scanInputAxes.at(i));
        scanInput->setReverse(scanInputDirections.at(i) == 1);
        ctx->registerTensor(TensorOrWeights{scanInput->getOutput(0)}, body.input(index).name());
    }

    // Loop Body. This is handled by dispatching to other op converters.
    TRT_CHECK(onnx2trt::parseGraph(ctx, body));

    // Set up recurrence outputs (first N body graph outputs).
    std::vector<TensorOrWeights> nodeOutputs{};
    for (int i = 0; i < nbStateVars; ++i)
    {
        const auto& bodyOutputName = body.output(i).name();
        auto& stateOutput = convertToTensor(ctx->tensors().at(bodyOutputName), ctx);
        LOG_VERBOSE("For state variable output: " << bodyOutputName << ", found matching tensor: " << stateOutput.getName() << ", with shape: " << stateOutput.getDimensions());
        stateVars.at(i)->setInput(1, stateOutput);
        // Each state variable is also a loop output
        nodeOutputs.emplace_back(loop->addLoopOutput(*stateVars.at(i)->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0));
    }
    // Finally, set up scan outputs.
    for (int i = 0; i < nbScanOutputs; ++i)
    {
        const int index = nbStateVars + i;
        const auto& bodyOutputName = body.output(index).name();
        auto& scanOutput = convertToTensor(ctx->tensors().at(bodyOutputName), ctx);
        // For scanOutputDirections, 0 indicates appending, and 1, prepending.
        const auto scanDirection = (scanOutputDirections.at(i) == 0) ? nvinfer1::LoopOutput::kCONCATENATE : nvinfer1::LoopOutput::kREVERSE;
        const auto scanAxis = scanOutputAxes.at(i);
        LOG_VERBOSE("For scan output: " << bodyOutputName << ", found matching tensor: " << scanOutput.getName()
            << ", with shape: " << scanOutput.getDimensions() << ". Using scan direction: "
            << static_cast<int>(scanDirection) << ", and scan axis: " << scanAxis);
        nvinfer1::ILoopOutputLayer* trtScanOut = loop->addLoopOutput(scanOutput, scanDirection, scanAxis);
        trtScanOut->setInput(1, *tripLimit);
        nodeOutputs.emplace_back(trtScanOut->getOutput(0));
    }

    return {nodeOutputs};
}

DEFINE_BUILTIN_OP_IMPORTER(Selu) {
  OnnxAttrs attrs(node, ctx);
  float alpha = attrs.get("alpha", 1.6732f);
  float beta = attrs.get("gamma", 1.0507f);
  return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSELU, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Shape)
{
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    RETURN_FIRST_OUTPUT(ctx->network()->addShape(input));
}

DEFINE_BUILTIN_OP_IMPORTER(Sigmoid)
{
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSIGMOID);
}

DEFINE_BUILTIN_OP_IMPORTER(Size)
{
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    auto shape = tensor_ptr->getDimensions();
    // Avoid TRT err: Unused Input
    ctx->network()->addIdentity(*tensor_ptr);
    nvinfer1::Dims weight_dims;
    weight_dims.nbDims = 1;
    weight_dims.d[0] = 1;
    // Note: Should technically be int64, but int32 allows for TRT compatibility
    auto weights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::INT32, weight_dims);
    int32_t size = get_shape_size(shape);
    *static_cast<int32_t*>(const_cast<void*>(weights.values)) = size;
    return {{weights}};
}

DEFINE_BUILTIN_OP_IMPORTER(Softmax)
{
    OnnxAttrs attrs(node, ctx);
    int axis = attrs.get("axis", 1);
    int nbDims = inputs.at(0).shape().nbDims;
    TRT_CHECK(convertAxis(axis, nbDims));
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims shape = tensor_ptr->getDimensions();

    // Work around dynamic input shapes like [-1, -1, X] which cannot be reshape to 2D
    // Having the trt_outputs_range_min attributes means it's from serialized iNetworkDefinition.
    // Do not do a reshape here because we have to maintain a 1-1 mapping during deserialization.
    if (shape.nbDims == axis + 1
        || !attrs.get<std::vector<float>>("trt_outputs_range_min", {}).empty())
    {
        auto* layer = ctx->network()->addSoftMax(*tensor_ptr);
        ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
        // Set softmax bitmask to the last dimension
        layer->setAxes(1 << axis);
        tensor_ptr = layer->getOutput(0);
        return {{tensor_ptr}};
    }
    auto* shape0 = ctx->network()->addShape(*tensor_ptr)->getOutput(0);
    nvinfer1::ITensor* dim0{nullptr};

    if (axis == 0)
    {
        dim0 = &makeShapeTensor(ctx, nvinfer1::Dims{1, {1}, {}});
    }
    else
    {
        auto* firstSlice = ctx->network()->addSlice(
            *shape0, nvinfer1::Dims{1, {0}, {}}, nvinfer1::Dims{1, {axis}, {}}, nvinfer1::Dims{1, {1}, {}});
        // Axes is set to 1 << 0
        auto* reduce0 = ctx->network()->addReduce(*firstSlice->getOutput(0), nvinfer1::ReduceOperation::kPROD, 1, true);
        dim0 = reduce0->getOutput(0);
    }

    auto* secondSlice = ctx->network()->addSlice(*shape0, nvinfer1::Dims{1, {axis}, {}},
        nvinfer1::Dims{1, {shape.nbDims - axis}, {}}, nvinfer1::Dims{1, {1}, {}});

    // Axes is set to 1 << 0
    auto* reduce1 = ctx->network()->addReduce(*secondSlice->getOutput(0), nvinfer1::ReduceOperation::kPROD, 1, true);
    auto* dim1 = reduce1->getOutput(0);

    std::vector<nvinfer1::ITensor*> tensors{dim0, dim1};
    auto* concatLayer = ctx->network()->addConcatenation(tensors.data(), 2);
    concatLayer->setAxis(0);
    nvinfer1::ITensor* concat = concatLayer->getOutput(0);

    auto* shuffleIn = ctx->network()->addShuffle(*tensor_ptr);
    ASSERT(shuffleIn, ErrorCode::kUNSUPPORTED_NODE);
    shuffleIn->setInput(1, *concat);
    auto* softmaxIn = shuffleIn->getOutput(0);
    auto* layer = ctx->network()->addSoftMax(*softmaxIn);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    layer->setAxes(1 << 1);
    auto* shuffleOut = ctx->network()->addShuffle(*layer->getOutput(0));
    ASSERT(shuffleOut, ErrorCode::kUNSUPPORTED_NODE);
    shuffleOut->setInput(1, *shape0);
    auto* res = shuffleOut->getOutput(0);
    return {{res}};
}

DEFINE_BUILTIN_OP_IMPORTER(Softsign)
{
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSOFTSIGN);
}

DEFINE_BUILTIN_OP_IMPORTER(Softplus)
{
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSOFTPLUS);
}

DEFINE_BUILTIN_OP_IMPORTER(ParametricSoftplus)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha");
    float beta = attrs.get<float>("beta");
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSOFTPLUS, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(SpaceToDepth)
{
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(*tensor_ptr);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    OnnxAttrs attrs(node, ctx);
    int block_size = attrs.get<int>("blocksize");
    nvinfer1::Dims dims = tensor_ptr->getDimensions();
    int ndim_spatial = dims.nbDims - 1;
    nvinfer1::Dims new_shape1;
    new_shape1.nbDims = dims.nbDims + ndim_spatial;
    new_shape1.d[0] = dims.d[0];
    for (int i = 0; i < ndim_spatial; ++i)
    {
        ASSERT(dims.d[1 + i] % block_size == 0, ErrorCode::kINVALID_NODE);
        new_shape1.d[1 + 2 * i + 0] = dims.d[1 + i] / block_size;
        new_shape1.d[1 + 2 * i + 1] = block_size;
    }
    layer->setReshapeDimensions(new_shape1);
    nvinfer1::Permutation perm;
    perm.order[ndim_spatial] = 0;
    for (int i = 0; i < ndim_spatial; ++i)
    {
        perm.order[ndim_spatial + 1 + i] = 1 + 2 * i + 0;
        perm.order[i] = 1 + 2 * i + 1;
    }
    layer->setSecondTranspose(perm);
    tensor_ptr = layer->getOutput(0);
    dims = tensor_ptr->getDimensions();
    nvinfer1::Dims new_shape2;
    new_shape2.nbDims = dims.nbDims - ndim_spatial;
    new_shape2.d[0] = dims.d[ndim_spatial];
    for (int i = 0; i < ndim_spatial; ++i)
    {
        new_shape2.d[0] *= dims.d[i];
        new_shape2.d[1 + i] = dims.d[ndim_spatial + 1 + i];
    }
    tensor_ptr = reshapeTensor(ctx, *tensor_ptr, new_shape2);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    dims = tensor_ptr->getDimensions();
    return {{tensor_ptr}};
}

// TODO: Legacy op for pre-1.0 ONNX spec; can be removed at some point
DEFINE_BUILTIN_OP_IMPORTER(SpatialBN)
{
    return importBatchNormalization(ctx, node, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(Split)
{
    ASSERT(inputs.size() == 1, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = tensor_ptr->getDimensions();
    int nbDims = dims.nbDims;
    OnnxAttrs attrs(node, ctx);
    int axis = attrs.get<int>("axis", 0);
    TRT_CHECK(convertAxis(axis, nbDims));
    std::vector<int> output_lengths;
    int noutput = node.output().size();
    if (attrs.count("split"))
    {
        output_lengths = attrs.get<std::vector<int>>("split");
        ASSERT(static_cast<int>(output_lengths.size()) == noutput, ErrorCode::kINVALID_NODE);
    }
    else
    {
        ASSERT(dims.d[axis] == -1 || dims.d[axis] % noutput == 0, ErrorCode::kINVALID_NODE);
        output_lengths.assign(noutput, dims.d[axis] / noutput);
    }
    nvinfer1::IPluginV2* plugin = createSplitPlugin(axis, output_lengths.data(), noutput);
    nvinfer1::IPluginV2Layer* layer = ctx->network()->addPluginV2(&tensor_ptr, 1, *plugin);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(layer->getNbOutputs() == noutput, ErrorCode::kINTERNAL_ERROR);
    std::vector<TensorOrWeights> outputs;
    for (int i = 0; i < noutput; ++i)
    {
        outputs.push_back(layer->getOutput(i));
    }
    return outputs;
}

DEFINE_BUILTIN_OP_IMPORTER(Sqrt)
{
    return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kSQRT);
}

DEFINE_BUILTIN_OP_IMPORTER(Squeeze)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    int rank = tensor.getDimensions().nbDims;

    OnnxAttrs attrs(node, ctx);
    auto axes = attrs.get<std::vector<int>>("axes");
    for (auto& axis : axes)
    {
        TRT_CHECK(convertAxis(axis, rank));
    }
    std::set<int> axisSet(axes.begin(), axes.end());

    std::vector<int> gatherIndices{};
    for (int i = 0; i < rank; ++i)
    {
        if (!axisSet.count(i))
        {
            gatherIndices.emplace_back(i);
        }
    }

    nvinfer1::IShapeLayer* shape = ctx->network()->addShape(tensor);

    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);

    if (gatherIndices.size() > 0)
    {
        nvinfer1::Dims gatherIndicesShape{1, static_cast<int>(gatherIndices.size())};
        nvinfer1::IGatherLayer* newShape = ctx->network()->addGather(*shape->getOutput(0), *addConstant(ctx, gatherIndices, ::ONNX_NAMESPACE::TensorProto::INT32, gatherIndicesShape)->getOutput(0), 0);
        layer->setInput(1, *newShape->getOutput(0));
    }
    else
    {
        layer->setReshapeDimensions(nvinfer1::Dims{0});
    }
    RETURN_FIRST_OUTPUT(layer);
}

DECLARE_BUILTIN_OP_IMPORTER(Add);
DEFINE_BUILTIN_OP_IMPORTER(Sub)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kSUB);
}

DEFINE_BUILTIN_OP_IMPORTER(Sum)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kSUM);
}

DEFINE_BUILTIN_OP_IMPORTER(Tanh)
{
    RETURN_FIRST_OUTPUT(ctx->network()->addActivation(inputs.at(0).tensor(), nvinfer1::ActivationType::kTANH));
}

DEFINE_BUILTIN_OP_IMPORTER(ThresholdedRelu)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha", 1.f);
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kTHRESHOLDED_RELU, &alpha);
}

DEFINE_BUILTIN_OP_IMPORTER(Tile)
{
    nvinfer1::ITensor* inp = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* repeats = &convertToTensor(inputs.at(1), ctx);

    nvinfer1::ITensor* inpShape = ctx->network()->addShape(*inp)->getOutput(0);

    int rank = inp->getDimensions().nbDims;

    std::vector<int> starts(rank);
    std::fill(starts.begin(), starts.end(), 0);

    nvinfer1::Dims strides{rank};
    std::fill(strides.d, strides.d + strides.nbDims, 1);

    nvinfer1::ISliceLayer* tile = ctx->network()->addSlice(*inp, nvinfer1::Dims{}, nvinfer1::Dims{}, strides);
    tile->setMode(nvinfer1::SliceMode::kWRAP);
    tile->setInput(1, *addConstant(ctx, starts, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, rank})->getOutput(0));

    nvinfer1::ITensor* tiledShape = ctx->network()->addElementWise(*inpShape, *repeats, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
    tile->setInput(2, *tiledShape);

    RETURN_FIRST_OUTPUT(tile);
}

DEFINE_BUILTIN_OP_IMPORTER(TopK)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    ASSERT(tensor.getType() != nvinfer1::DataType::kINT32, ErrorCode::kUNSUPPORTED_NODE);
    OnnxAttrs attrs(node, ctx);
    int axis = attrs.get("axis", -1);
    int k;
    // Don't support TopK with k as a tensor
    if (ctx->getOpsetVersion() >= 10)
    {
        ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(inputs.at(1).weights().count() == 1, ErrorCode::kUNSUPPORTED_NODE);
        k = *static_cast<int*>(inputs.at(1).weights().values);
    }
    else
    {
        ASSERT(attrs.count("k"), ErrorCode::kINVALID_NODE);
        k = attrs.get<int>("k");
    }

    int nbDims = tensor.getDimensions().nbDims;
    TRT_CHECK(convertAxis(axis, nbDims));
    uint32_t axisMask = 1 << axis;
    nvinfer1::ITopKLayer* layer = ctx->network()->addTopK(tensor, nvinfer1::TopKOperation::kMAX, k, axisMask);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    return {{layer->getOutput(0), layer->getOutput(1)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Transpose)
{
    TensorOrWeights input = inputs.at(0);
    OnnxAttrs attrs(node, ctx);
    int ndim = input.shape().nbDims;
    ASSERT(ndim <= nvinfer1::Dims::MAX_DIMS, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Permutation default_perm; // Default is to reverse dims
    for (int i = 0; i < ndim; ++i)
    {
        default_perm.order[i] = ndim - 1 - i;
    }
    nvinfer1::Permutation perm = attrs.get("perm", default_perm);
    if (input.is_tensor())
    {
        // Note: Dimension types kept unchanged in order to avoid TRT complaining about CHW order
        nvinfer1::ITensor* output_tensor = transposeTensor(ctx, input.tensor(), perm, false);
        ASSERT(output_tensor, ErrorCode::kUNSUPPORTED_NODE);
        return {{output_tensor}};
    }
    else
    {
        auto weights = input.weights();
        auto new_weights = ctx->createTempWeights(weights.type, weights.shape);
        ASSERT(transposeWeights(weights, perm, &new_weights), ErrorCode::kUNSUPPORTED_NODE);
        weights = new_weights;

        return {{weights}};
    }
}

DEFINE_BUILTIN_OP_IMPORTER(Unsqueeze)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims old_shape = tensor.getDimensions();
    int ndim_in = old_shape.nbDims;
    OnnxAttrs attrs(node, ctx);
    auto axes = attrs.get<std::vector<int>>("axes");
    std::set<int> axes_set(axes.begin(), axes.end());
    int ndim_out = ndim_in + axes_set.size();
    ASSERT(ndim_out <= nvinfer1::Dims::MAX_DIMS, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Dims new_shape;
    new_shape.nbDims = ndim_out;
    nvinfer1::Permutation perm;

    // Append 1 to new_shape for each axes and permute them into the right position
    // Align dynamic dimensions in input and 0s in new_shape
    // i: from 0 to old_shape.nbDims
    // j: from 0 to number of axes
    for (int i = 0, j = 0 ; (i + j) < ndim_in + (int)axes_set.size(); )
    {
        if (axes_set.count(i+j))
        {
            perm.order[i + j] = ndim_in + j;
            new_shape.d[ndim_in + j] = 1;
            j ++;
        }
        else
        {
            perm.order[i + j] = i;
            old_shape.d[i] < 0 ? new_shape.d[i] = 0 : new_shape.d[i] =old_shape.d[i];
            i ++;
        }
    }

    LOG_VERBOSE("Unsqueezing from " << old_shape << " to " << new_shape);
    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    layer->setReshapeDimensions(new_shape);
    layer->setSecondTranspose(perm);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Resize)
{
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    int input_dims = input.getDimensions().nbDims;
    ASSERT(input_dims > 0, ErrorCode::kUNSUPPORTED_NODE);

    // Add resize layer
    nvinfer1::IResizeLayer* layer = ctx->network()->addResize(input);

    // Retrive and validate scale factors.
    // Scale factors include batch dimensions as well.
    ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
    auto scales = inputs.at(1);
    // Support for scales as weights
    ASSERT(scales.is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    ShapedWeights scales_weights = scales.weights();
    ASSERT(scales_weights.shape.nbDims == 1, ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(scales_weights.count() == static_cast<size_t>(input_dims), ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(scales_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT, ErrorCode::kINVALID_NODE);
    // Get floating point scale factors.
    float const* scales_ptr = static_cast<float const*>(scales_weights.values);
    layer->setScales(scales_ptr, input_dims);

    // Set resize mode
    OnnxAttrs attrs(node, ctx);
    auto mode = attrs.get<std::string>("mode", "nearest");
    ASSERT(mode == "nearest" || mode == "linear", ErrorCode::kUNSUPPORTED_NODE);
    // Set default resize mode. Nearest resize support N-D (where 0 < N <= 8) resize.
    nvinfer1::ResizeMode resizeMode = nvinfer1::ResizeMode::kNEAREST;
    if (mode == "linear")
    {
        // Linear resize support 1-D, 2-D and 3D resize.
        ASSERT((input_dims >= 1) && (input_dims <= 3), ErrorCode::kUNSUPPORTED_NODE);
        resizeMode = nvinfer1::ResizeMode::kLINEAR;
    }
    layer->setResizeMode(resizeMode);

    // Set other attributes. ONNX spec does not have this attribute yet.
    // Default: False. Set it any way.
    layer->setAlignCorners(false);

    // Return layer output
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Upsample)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    const int nbDims = tensor.getDimensions().nbDims;
    ASSERT(nbDims > 0, ErrorCode::kUNSUPPORTED_NODE);
    OnnxAttrs attrs(node, ctx);
    std::vector<float> scale_factors(nbDims, 1.0f);
    if (ctx->getOpsetVersion() >= 9)
    {
        // Get scale factors from inputs[1]
        ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
        auto scales_input = inputs.at(1);
        // Retrieve and validate scale factors.
        ASSERT(scales_input.is_weights(), ErrorCode::kUNSUPPORTED_NODE);
        ShapedWeights scales_weights = scales_input.weights();
        ASSERT(scales_weights.shape.nbDims == 1, ErrorCode::kUNSUPPORTED_NODE);
        // Scale factors has batch dimension.
        ASSERT(scales_weights.count() == static_cast<size_t>(nbDims), ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(scales_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT, ErrorCode::kINVALID_NODE);
        float const* scales_ptr = static_cast<float const*>(scales_weights.values);
        for (int i = 0; i < nbDims; i++)
        {
            scale_factors[i] = scales_ptr[i];
        }
    }
    else
    {
        ASSERT(attrs.count("scales"), ErrorCode::kUNSUPPORTED_NODE);
        // Get scale factors from OnnxAttrs.
        auto scales = attrs.get<std::vector<float>>("scales");
        // Scale factors has batch dimension.
        ASSERT(static_cast<int>(scales.size()) == nbDims, ErrorCode::kUNSUPPORTED_NODE);
        for (int i = 0; i < nbDims; i++)
        {
            scale_factors[i] = scales[i];
        }
    }
    auto mode = attrs.get<std::string>("mode", "nearest");
    ASSERT(mode == "nearest" || mode == "linear", ErrorCode::kUNSUPPORTED_NODE);
    // Set default resize mode. Nearest resize support N-D (where 0 < N <= 8) resize.
    nvinfer1::ResizeMode resizeMode = nvinfer1::ResizeMode::kNEAREST;
    if (mode == "linear")
    {
        // Linear resize support 1-D, 2-D and 3D resize.
        ASSERT((nbDims >= 1) && (nbDims <= 3), ErrorCode::kUNSUPPORTED_NODE);
        resizeMode = nvinfer1::ResizeMode::kLINEAR;
    }
    // Add resize layer
    nvinfer1::IResizeLayer* const layer = ctx->network()->addResize(tensor);
    layer->setScales(scale_factors.data(), nbDims);
    layer->setResizeMode(resizeMode);
    RETURN_FIRST_OUTPUT(layer);
}

// TRT-7031: add tests
DEFINE_BUILTIN_OP_IMPORTER(Slice)
{
    // If opset version >= 10 slice paramerters are weights instead of attributes
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    const int rank = tensor.getDimensions().nbDims;
    std::vector<int64_t> starts;
    std::vector<int64_t> ends;
    std::vector<int64_t> axes;
    std::vector<int64_t> steps;
    if (ctx->getOpsetVersion() >= 10)
    {
        int nbInputs = node.input().size();
        ASSERT(nbInputs >= 3 && nbInputs <= 5, ErrorCode::kUNSUPPORTED_NODE);

        weightsToVector(inputs.at(1), &starts);
        weightsToVector(inputs.at(2), &ends);

        if (inputs.size() > 3 && inputs.at(3))
        {
            weightsToVector(inputs.at(3), &axes);
        }
        else
        {
            axes.resize(starts.size());
            std::iota(axes.begin(), axes.end(), 0);
        }

        if (inputs.size() > 4 && inputs.at(4))
        {
            weightsToVector(inputs.at(4), &steps);
        }
        else
        {
            steps = std::vector<int64_t>(starts.size(), 1);
        }
    }
    else
    {
        OnnxAttrs attrs(node, ctx);
        starts = attrs.get<std::vector<int64_t>>("starts");
        ends = attrs.get<std::vector<int64_t>>("ends");

        std::vector<int64_t> defaultAxes(starts.size());
        std::iota(defaultAxes.begin(), defaultAxes.end(), 0);
        axes = attrs.get<std::vector<int64_t>>("axes", defaultAxes);

        steps = std::vector<int64_t>(starts.size(), 1);
    }

    std::unordered_map<int64_t, std::tuple<int64_t, int64_t, int64_t>> sliceBounds{}; // Maps axis to [start, end, step]
    for (size_t i = 0; i < axes.size(); ++i)
    {
        sliceBounds[axes[i]] = std::make_tuple(starts[i], ends[i], steps[i]);
    }

    std::vector<nvinfer1::ITensor*> startIndices{};
    std::vector<nvinfer1::ITensor*> sizeIndices{};
    std::vector<nvinfer1::ITensor*> strideIndices{};
    nvinfer1::Dims indexShape{1, 1};

    for (int i = 0; i < rank; ++i)
    {
        static const auto handleNegativeIndex = [&ctx, &indexShape, &tensor, &i](int64_t index) -> nvinfer1::ITensor*
        {
            if (index > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
            {
                std::cout << "WARNING: Slice index is out of bounds of INT32, clamping to INT32_MAX" << std::endl;
                index = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
            }

            nvinfer1::ITensor* indexTensor{addConstantScalar(ctx, index, ::ONNX_NAMESPACE::TensorProto::INT32, indexShape)->getOutput(0)};
            nvinfer1::ITensor* dimLength{getAxisLength(ctx, &tensor, i, indexShape)};
            if (index < 0)
            {
                indexTensor = ctx->network()->addElementWise(*dimLength, *indexTensor, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
            }
            return indexTensor;
        };

        nvinfer1::ITensor* endIndex;
        if (sliceBounds.count(i))
        {
            int64_t start, end, stride;
            std::tie(start, end, stride) = sliceBounds.at(i);

            startIndices.emplace_back(handleNegativeIndex(start));
            endIndex = handleNegativeIndex(end);
            strideIndices.emplace_back(handleNegativeIndex(stride)); // This should never be negative anyway.
        }
        else
        {
            // Default values take the entire length of the axis.
            startIndices.emplace_back(handleNegativeIndex(0));
            endIndex = getAxisLength(ctx, &tensor, i, indexShape);
            strideIndices.emplace_back(handleNegativeIndex(1));
        }
        // size[i] = min(end[i], getAxisLength(ctx, &tensor, i, indexShape)) - start[i]) / stride[i]
        sizeIndices.emplace_back(
            ctx->network()->addElementWise(
                *ctx->network()->addElementWise(
                    *ctx->network()->addElementWise(
                        *endIndex,
                        *getAxisLength(ctx, &tensor, i, indexShape),
                        nvinfer1::ElementWiseOperation::kMIN
                    )->getOutput(0),
                    *startIndices.back(),
                    nvinfer1::ElementWiseOperation::kSUB
                )->getOutput(0),
                *strideIndices.back(),
                nvinfer1::ElementWiseOperation::kDIV
            )->getOutput(0)
        );
    }

    nvinfer1::ITensor* startsTensor = ctx->network()->addConcatenation(startIndices.data(), startIndices.size())->getOutput(0);
    nvinfer1::ITensor* sizeTensor = ctx->network()->addConcatenation(sizeIndices.data(), sizeIndices.size())->getOutput(0);
    nvinfer1::ITensor* stridesTensor = ctx->network()->addConcatenation(strideIndices.data(), strideIndices.size())->getOutput(0);

    nvinfer1::ISliceLayer* slice = ctx->network()->addSlice(tensor, nvinfer1::Dims{rank}, nvinfer1::Dims{rank}, nvinfer1::Dims{rank});
    slice->setInput(1, *startsTensor);
    slice->setInput(2, *sizeTensor);
    slice->setInput(3, *stridesTensor);
    RETURN_FIRST_OUTPUT(slice);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Scale)
{
    ASSERT(inputs.size() >= 1, nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    if (inputs.size() >= 2)
    {
        ASSERT(inputs.at(1).is_weights(), nvonnxparser::ErrorCode::kINVALID_NODE);
    }
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    int counter = 1;

    nvinfer1::ScaleMode mode = attrs.get<nvinfer1::ScaleMode>("mode");

    // check if there's no weigths at all
    // if no weights, just choose datatype of the input tensor
    // This is based on the assumption that weights should be
    // the same datatype as inputs
    auto type = inputs.size() > 1 ? inputs.at(1).weights().type : trtDataTypeToONNX(inputs.at(0).tensor().getType());

    auto scale = ShapedWeights::empty(type);
    auto shift = ShapedWeights::empty(type);
    auto power = ShapedWeights::empty(type);

    if (attrs.get<bool>("scale"))
    {
        ASSERT(inputs.at(counter).is_weights(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        scale = inputs.at(counter++).weights();
    }
    if (attrs.get<bool>("shift"))
    {
        ASSERT(inputs.at(counter).is_weights(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        shift = inputs.at(counter++).weights();
    }
    if (attrs.get<bool>("power"))
    {
        ASSERT(inputs.at(counter).is_weights(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        power = inputs.at(counter++).weights();
    }

    nvinfer1::IScaleLayer* layer = ctx->network()->addScale(input, mode, shift, scale, power);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Shuffle)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    nvinfer1::Permutation perm1 = attrs.get<nvinfer1::Permutation>("first_perm");
    nvinfer1::Permutation perm2 = attrs.get<nvinfer1::Permutation>("second_perm");

    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(input);
    layer->setFirstTranspose(perm1);
    layer->setSecondTranspose(perm2);

    if (inputs.size() == 1)
    {
        if (attrs.count("reshape_dims") > 0)
        {
            nvinfer1::Dims reshapeDims = attrs.get<nvinfer1::Dims>("reshape_dims");
            layer->setReshapeDimensions(reshapeDims);
        }
    }
    else
    {
        ASSERT(inputs.at(1).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
        layer->setInput(1, inputs.at(1).tensor());
    }

    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_TopK_Min)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    ASSERT(inputs.at(1).is_weights(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& kWeights = inputs.at(1).weights();
    int k = *static_cast<int*>(kWeights.values);

    int32_t axes = 1 << (attrs.get<int>("axis"));

    nvinfer1::ITopKLayer* layer = ctx->network()->addTopK(input, nvinfer1::TopKOperation::kMIN, k, axes);

    RETURN_ALL_OUTPUTS(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MatMul)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT(inputs.at(1).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input0 = inputs.at(0).tensor();
    auto& input1 = inputs.at(1).tensor();

    OnnxAttrs attrs(node, ctx);
    nvinfer1::MatrixOperation op0 = attrs.get<nvinfer1::MatrixOperation>("op_0");
    nvinfer1::MatrixOperation op1 = attrs.get<nvinfer1::MatrixOperation>("op_1");

    nvinfer1::IMatrixMultiplyLayer* layer = ctx->network()->addMatrixMultiply(input0, op0, input1, op1);
    RETURN_FIRST_OUTPUT(layer);
}

typedef std::function<void(int, nvinfer1::RNNGateType, nvinfer1::Weights)> RNNWeightsAdder;

bool addRNNv2Weights(RNNWeightsAdder adder, int layerNb, std::vector<nvinfer1::RNNGateType> const& gates,
    std::vector<TensorOrWeights>& inputs, int& counter)
{
    for (nvinfer1::RNNGateType gate : gates)
    {
        if (!inputs.at(counter).is_weights())
            return false;
        auto& weights = inputs.at(counter++).weights();
        adder(layerNb, gate, weights);
    }
    return true;
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_RNNv2)
{
    OnnxAttrs attrs(node, ctx);

    int layerCount = attrs.get<int>("layer_count");
    int hiddenSize = attrs.get<int>("hidden_size");
    int maxSeqLen = attrs.get<int>("max_seq_length");
    nvinfer1::RNNOperation op = attrs.get<nvinfer1::RNNOperation>("rnn_op");
    nvinfer1::RNNInputMode inputMode = attrs.get<nvinfer1::RNNInputMode>("input_mode");
    nvinfer1::RNNDirection direction = attrs.get<nvinfer1::RNNDirection>("direction");

    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    int counter = 1;
    nvinfer1::IRNNv2Layer* layer = ctx->network()->addRNNv2(input, layerCount, hiddenSize, maxSeqLen, op);
    layer->setInputMode(inputMode);
    layer->setDirection(direction);

    if (attrs.get<bool>("has_hidden_state"))
    {
        ASSERT(inputs.at(counter).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
        layer->setHiddenState(inputs.at(counter++).tensor());
    }
    if (op == nvinfer1::RNNOperation::kLSTM && attrs.get<bool>("has_cell_state", false))
    {
        ASSERT(inputs.at(counter).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
        layer->setCellState(inputs.at(counter++).tensor());
    }
    if (attrs.get<bool>("has_seq_lengths"))
    {
        ASSERT(inputs.at(counter).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
        layer->setSequenceLengths(inputs.at(counter++).tensor());
    }

    int nbWeights = (direction == nvinfer1::RNNDirection::kBIDIRECTION ? 2 : 1) * layerCount;
    const int K = direction == nvinfer1::RNNDirection::kUNIDIRECTION ? 1 : 2;
    std::vector<nvinfer1::RNNGateType> gates;
    switch (op)
    {
    case nvinfer1::RNNOperation::kRELU:
    case nvinfer1::RNNOperation::kTANH:
        gates = std::vector<nvinfer1::RNNGateType>({nvinfer1::RNNGateType::kINPUT});
        break;
    case nvinfer1::RNNOperation::kLSTM:
        gates = std::vector<nvinfer1::RNNGateType>({nvinfer1::RNNGateType::kINPUT, nvinfer1::RNNGateType::kOUTPUT,
            nvinfer1::RNNGateType::kFORGET, nvinfer1::RNNGateType::kCELL});
        break;
    case nvinfer1::RNNOperation::kGRU:
        gates = std::vector<nvinfer1::RNNGateType>(
            {nvinfer1::RNNGateType::kUPDATE, nvinfer1::RNNGateType::kRESET, nvinfer1::RNNGateType::kHIDDEN});
        break;
    }

    RNNWeightsAdder weightsAdder = [&layer](int n, nvinfer1::RNNGateType gate, nvinfer1::Weights weights) {
        layer->setWeightsForGate(n, gate, true, weights);
    };
    RNNWeightsAdder recurrentWeightsAdder = [&layer](int n, nvinfer1::RNNGateType gate, nvinfer1::Weights weights) {
        layer->setWeightsForGate(n, gate, false, weights);
    };
    RNNWeightsAdder biasAdder = [&layer](int n, nvinfer1::RNNGateType gate, nvinfer1::Weights weights) {
        layer->setBiasForGate(n, gate, true, weights);
    };
    RNNWeightsAdder recurrentBiasAdder = [&layer](int n, nvinfer1::RNNGateType gate, nvinfer1::Weights weights) {
        layer->setBiasForGate(n, gate, false, weights);
    };

    for (int n = 0; n < nbWeights; ++n)
    {
        if (n >= K || inputMode == nvinfer1::RNNInputMode::kLINEAR)
        {
            ASSERT(addRNNv2Weights(weightsAdder, n, gates, inputs, counter), nvonnxparser::ErrorCode::kINVALID_NODE);
        }
        ASSERT(
            addRNNv2Weights(recurrentWeightsAdder, n, gates, inputs, counter), nvonnxparser::ErrorCode::kINVALID_NODE);
        ASSERT(addRNNv2Weights(biasAdder, n, gates, inputs, counter), nvonnxparser::ErrorCode::kINVALID_NODE);
        ASSERT(addRNNv2Weights(recurrentBiasAdder, n, gates, inputs, counter), nvonnxparser::ErrorCode::kINVALID_NODE);
    }

    RETURN_ALL_OUTPUTS(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_RaggedSoftmax)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT(inputs.at(1).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();
    auto& bounds = inputs.at(1).tensor();

    nvinfer1::IRaggedSoftMaxLayer* layer = ctx->network()->addRaggedSoftMax(input, bounds);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_FullyConnected)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    int nbChannels = attrs.get<int>("channels");

    ASSERT(inputs.at(1).is_weights(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& kernelWeights = inputs.at(1).weights();

    ShapedWeights biasWeights = ShapedWeights::empty(kernelWeights.type);
    if (inputs.size() == 3)
    {
        ASSERT(inputs.at(2).is_weights(), nvonnxparser::ErrorCode::kINVALID_NODE);
        biasWeights = inputs.at(2).weights();
    }

    nvinfer1::IFullyConnectedLayer* layer
        = ctx->network()->addFullyConnected(input, nbChannels, kernelWeights, biasWeights);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MaxAverageBlendPool)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    int nbSpatialDims = attrs.get<nvinfer1::Dims>("kernel_shape").nbDims;
    nvinfer1::Dims kernel_size = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims beg_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims end_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::PaddingMode paddingMode;
    bool exclude_padding(true);
    getKernelParams(ctx, node, &kernel_size, &strides, &beg_padding, &end_padding, paddingMode, exclude_padding);
    float blend = attrs.get<float>("blend");

    nvinfer1::IPoolingLayer* layer
        = ctx->network()->addPoolingNd(input, nvinfer1::PoolingType::kMAX_AVERAGE_BLEND, kernel_size);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    layer->setStrideNd(strides);
    layer->setAverageCountExcludesPadding(exclude_padding);
    layer->setPaddingMode(paddingMode);

    layer->setPrePadding(beg_padding);
    layer->setPostPadding(end_padding);

    layer->setBlendFactor(blend);

    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_PluginV2)
{
    std::vector<nvinfer1::ITensor*> tensors;
    for (auto& input : inputs)
    {
        ASSERT(input.is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        tensors.push_back(&input.tensor());
    }
    OnnxAttrs attrs(node, ctx);

    nvinfer1::IPluginRegistry* registry = getPluginRegistry();

    std::string name = attrs.get<std::string>("name");
    std::string version = attrs.get<std::string>("version");
    std::string nspace = attrs.get<std::string>("namespace");
    std::string buffer = attrs.get<std::string>("data");

    nvinfer1::IPluginCreator* creator = registry->getPluginCreator(name.c_str(), version.c_str(), nspace.c_str());
    ASSERT(creator != nullptr, nvonnxparser::ErrorCode::kINVALID_NODE);

    nvinfer1::IPluginV2* plugin = creator->deserializePlugin("", buffer.data(), buffer.size());

    nvinfer1::IPluginV2Layer* layer = ctx->network()->addPluginV2(tensors.data(), tensors.size(), *plugin);
    RETURN_ALL_OUTPUTS(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Gather)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(1).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    auto& data = inputs.at(0).tensor();
    auto& indices = inputs.at(1).tensor();
    OnnxAttrs attrs(node, ctx);
    int axis = attrs.get<int>("axis", 0);
    int nbElementWiseDims = attrs.get<int>("nbElementWiseDims", 0);
    int r = data.getDimensions().nbDims;

    ASSERT(indices.getType() == nvinfer1::DataType::kINT32, nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT(axis != -r, nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(r >= 1, nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT(-r <= axis && axis <= r, nvonnxparser::ErrorCode::kINVALID_NODE);

    if (axis < 0)
    {
        axis += r;
    }

    nvinfer1::IGatherLayer* layer = ctx->network()->addGather(data, indices, axis);
    layer->setNbElementWiseDims(nbElementWiseDims);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Slice)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    auto& input = inputs.at(0).tensor();

    nvinfer1::ISliceLayer* layer;
    // inputs' size is only 1 means this is a Slice with
    // only input as tensor.
    // start, size, stride are all attributes
    if (inputs.size() == 1)
    {
        OnnxAttrs attrs(node, ctx);
        auto start = attrs.get<nvinfer1::Dims>("start");
        auto size = attrs.get<nvinfer1::Dims>("size");
        auto stride = attrs.get<nvinfer1::Dims>("stride");
        layer = ctx->network()->addSlice(input, start, size, stride);
    }
    else
    {
        // start, size, stride are all tensors
        ASSERT(inputs.at(1).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(inputs.at(2).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(inputs.at(3).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        auto& start = inputs.at(1).tensor();
        auto& size = inputs.at(2).tensor();
        auto& stride = inputs.at(3).tensor();

        nvinfer1::Dims dummy{start.getDimensions().nbDims, {0}, {}};
        layer = ctx->network()->addSlice(input, dummy, dummy, dummy);
        layer->setInput(1, start);
        layer->setInput(2, size);
        layer->setInput(3, stride);
    }
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Resize)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    auto& input = inputs.at(0).tensor();

    nvinfer1::IResizeLayer* layer;
    layer = ctx->network()->addResize(input);

    OnnxAttrs attrs(node, ctx);
    auto alignCorners = attrs.get<bool>("align_corners", false);
    auto mode = attrs.get<nvinfer1::ResizeMode>("mode");
    layer->setAlignCorners(alignCorners);
    layer->setResizeMode(mode);

    if (inputs.size() == 1)
    {
        auto outputDims = attrs.get<nvinfer1::Dims>("output_dims", nvinfer1::Dims{-1, {}});
        if (outputDims.nbDims > 0)
        {
            layer->setOutputDimensions(outputDims);
        }
        else
        {
            auto scales = attrs.get<std::vector<float>>("scales");
            ASSERT(scales.size() > 0, nvonnxparser::ErrorCode::kINVALID_NODE);
            layer->setScales(&scales[0], scales.size());
        }
    }
    else
    {
        ASSERT(inputs.at(1).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        layer->setInput(1, inputs.at(1).tensor());
    }
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_FloorDiv)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kFLOOR_DIV);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Conv)
{
    return importConv(ctx, node, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Deconv)
{
    return importConvTranspose(ctx, node, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MaxPool)
{
    return importMaxPool(ctx, node, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_AveragePool)
{
    return importAveragePool(ctx, node, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(Where)
{
    nvinfer1::ITensor& condition = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor& x = convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor& y = convertToTensor(inputs.at(2), ctx);

    nvinfer1::Dims cDims = condition.getDimensions();
    nvinfer1::Dims xDims = x.getDimensions();
    nvinfer1::Dims yDims = y.getDimensions();

    ASSERT(cDims.nbDims == xDims.nbDims, ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(cDims.nbDims == yDims.nbDims, ErrorCode::kUNSUPPORTED_NODE);

    auto* layer = ctx->network()->addSelect(condition, x, y);

    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(RandomUniformLike)
{
    ASSERT(inputs.size() == 1, ErrorCode::kINTERNAL_ERROR);
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    auto& input = inputs.at(0).tensor();

    nvinfer1::IFillLayer* layer{};
    layer = ctx->network()->addFill(makeDims(1, 0), nvinfer1::FillOperation::kRANDOM_UNIFORM);

    OnnxAttrs attrs(node, ctx);
    auto high = attrs.get<float>("high", 1.f);
    auto low = attrs.get<float>("low", 0.f);
    layer->setAlpha(low);
    layer->setBeta(high);
    layer->setInput(0, input);

    if (attrs.count("dtype"))
    {
        auto dtype = attrs.get<int>("dtype", 0);
        switch (dtype){
        case ::ONNX_NAMESPACE::TensorProto::FLOAT:
            layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
            break;
        case ::ONNX_NAMESPACE::TensorProto::FLOAT16:
            layer->setOutputType(0, nvinfer1::DataType::kHALF);
            break;
        default:
            return MAKE_ERROR("Unsupported data type", ErrorCode::kINVALID_VALUE);
        }
    }
    else
    {
        layer->setOutputType(0, input.getType());
    }

    if (attrs.count("seed"))
    {
        auto seed = attrs.get<float>("seed", 0.f);
        nvinfer1::IConstantLayer* seedConstant = addConstantScalar(ctx, seed, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        nvinfer1::ITensor* seedConstantTensor = seedConstant->getOutput(0);
        layer->setInput(3, *seedConstantTensor);
    }
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Range)
{
    ASSERT(inputs.size() == 3, ErrorCode::kINTERNAL_ERROR);
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(1).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(2).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    auto& start = inputs.at(0).tensor();
    auto& limit = inputs.at(1).tensor();
    auto& delta = inputs.at(2).tensor();

    nvinfer1::IFillLayer* layer{};
    layer = ctx->network()->addFill(makeDims(1, 0), nvinfer1::FillOperation::kLINSPACE);
    layer->setInput(1, start);
    layer->setInput(2, delta);

    auto* lengthLayer = ctx->network()->addElementWise(limit, start, nvinfer1::ElementWiseOperation::kSUB);
    auto* lengthTensor = lengthLayer->getOutput(0);
    auto* countLayer = ctx->network()->addElementWise(*lengthTensor, delta, nvinfer1::ElementWiseOperation::kDIV);
    auto* countTensor = countLayer->getOutput(0);
    layer->setInput(0, *countTensor);

    RETURN_FIRST_OUTPUT(layer);
}


} // namespace

} // namespace onnx2trt
