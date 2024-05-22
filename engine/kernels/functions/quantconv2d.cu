/**
 * @file quantconv2d.cu
 * @brief QuantConv2d function implementation.
 * @version 0.0.1
 * @date 2024-02-23
*/

#include <cuda.h>
#include <torch/extension.h>
#include <vector>
#include "funcs.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCKSIZE 32


/**
 * QuantConv2d CUDA kernel.
 * 
 * @param input The input tensor of 1D shape, which contains elements of unsigned char.
 * @param input_scale The scale of input tensor, which is one element tensor (per_tensor) or 1D tensor (per_channel).
 * @param input_zero The zero of input tensor, which is one element tensor (per_tensor) or 1D tensor (per_channel).
 * @param input_per_tensor A flag indicating whether the quantized input tensor is per tensor.
 * @param input_n_bits The number of bits to quantize the input tensor.
 * @param input_offset The offset of input tensor.
 * @param weight The weight tensor of 1D shape, which contains elements of unsigned char.
 * @param weight_scale The scale of weight tensor, which is one element tensor (per_tensor) or 1D tensor (per_channel).
 * @param weight_zero The zero of weight tensor, which is one element tensor (per_tensor) or 1D tensor (per_channel).
 * @param weight_per_tensor A flag indicating whether the quantized weight tensor is per tensor.
 * @param weight_n_bits The number of bits to quantize the weight tensor.
 * @param weight_offset The offset of weight tensor.
 * @param bias The bias tensor of shape (output_channel) or None.
 * @param output The output tensor of shape (batch_size, output_channel, output_height, output_width).
 * @param batch_size The batch size.
 * @param input_channel The input channel.
 * @param input_height The input height.
 * @param input_width The input width.
 * @param output_channel The output channel.
 * @param kernel_height The kernel height.
 * @param kernel_width The kernel width.
 * @param output_height The output height.
 * @param output_width The output width.
 * @param stride The stride of convolution.
 * @param padding The padding of convolution.
*/
template <typename input_t, typename weight_t>
__global__ void quantconv2d_cuda_kernel(
    const unsigned char * __restrict__ input,
    const float * __restrict__ input_scale,
    const float * __restrict__ input_zero,
    const bool input_per_tensor,
    const int input_n_bits,
    const unsigned char input_offset,
    const unsigned char * __restrict__ weight,
    const float * __restrict__ weight_scale,
    const float * __restrict__ weight_zero,
    const bool weight_per_tensor,
    const int weight_n_bits,
    const unsigned char weight_offset,
    const float * __restrict__ bias,
    float * __restrict__ output,
    const int batch_size,
    const int input_channel,
    const int input_height,
    const int input_width,
    const int output_channel,
    const int kernel_height,
    const int kernel_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding)
{
    // Get the index of the current thread
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the index is valid
    if (index < batch_size * output_channel * output_height * output_width){
        // Get the batch index
        auto batch = index / (output_channel * output_height * output_width);

        // Get the output channel index
        auto output_index = index % (output_channel * output_height * output_width);
        auto outw = output_index % output_width;
        auto outh = (output_index / output_width) % output_height;
        auto outc = (output_index / (output_height * output_width)) % output_channel;

        // Compute the output
        float output_value = bias == nullptr ? 0 : bias[outc];

        #pragma unroll
        for (auto inc = 0; inc < input_channel; inc++){
            for (auto keh = 0; keh < kernel_height; keh++){
                for (auto kew = 0; kew < kernel_width; kew++){
                    int inh = outh * stride + keh - padding;
                    int inw = outw * stride + kew - padding;

                    if (inh >= 0 && inh < input_height && inw >=0 && inw < input_width){
                        // Get the input value
                        auto ele_idx = batch * input_channel * input_height * input_width + inc * input_height * input_width + inh * input_width + inw;
                        auto byte_idx = ele_idx * input_n_bits / 8;
                        auto bit_idx = ele_idx * input_n_bits % 8;
                        unsigned char input_value = (input[byte_idx] >> bit_idx) & ((1 << input_n_bits) - 1);
                        if (bit_idx + input_n_bits > 8)
                            input_value |= (input[byte_idx + 1] << (8 - bit_idx)) & ((1 << input_n_bits) - 1);

                        // Dequantize the input value
                        input_value -= input_offset;
                        input_t _input_value = (input_t)input_value;
                        float input_value_f = input_per_tensor ?
                            (_input_value - input_zero[0]) * input_scale[0] :
                            (_input_value - input_zero[inc]) * input_scale[inc];

                        // Get the weight value
                        ele_idx = outc * input_channel * kernel_height * kernel_width + inc * kernel_height * kernel_width + keh * kernel_width + kew;
                        byte_idx = ele_idx * weight_n_bits / 8;
                        bit_idx = ele_idx * weight_n_bits % 8;
                        unsigned char weight_value = (weight[byte_idx] >> bit_idx) & ((1 << weight_n_bits) - 1);
                        if (bit_idx + weight_n_bits > 8)
                            weight_value |= (weight[byte_idx + 1] << (8 - bit_idx)) & ((1 << weight_n_bits) - 1);

                        // Dequantize the weight value
                        weight_value -= weight_offset;
                        weight_t _weight_value = (weight_t)weight_value;
                        float weight_value_f = weight_per_tensor ?
                            (_weight_value - weight_zero[0]) * weight_scale[0] :
                            (_weight_value - weight_zero[outc]) * weight_scale[outc];

                        // Compute the output value
                        output_value += input_value_f * weight_value_f;
                    }
                }
            }
        }

        // Store the output
        output[index] = output_value;
    }
}


/**
 * Quantized Conv2d function.
 * 
 * @param input The input tensor of 1D shape, which contains elements of unsigned char.
 * @param input_des The description of input tensor, which is a 1D tensor of [n_bits, signed, *shape].
 * @param input_scale The scale of input tensor, which is one element tensor (per_tensor) or 1D tensor (per_channel).
 * @param input_zero The zero of input tensor, which is one element tensor (per_tensor) or 1D tensor (per_channel).
 * @param weight The weight tensor of 1D shape, which contains elements of unsigned char.
 * @param weight_des The description of weight tensor, which is a 1D tensor of [n_bits, signed, *shape].
 * @param weight_scale The scale of weight tensor, which is one element tensor (per_tensor) or 1D tensor (per_channel).
 * @param weight_zero The zero of weight tensor, which is one element tensor (per_tensor) or 1D tensor (per_channel).
 * @param bias The bias tensor of shape (output_channel) or None.
 * @param stride The stride of convolution.
 * @param padding The padding of convolution.
 * @return The output tensor of shape (batch_size, output_channel, output_height, output_width).
 *         The output_height and output_width are calculated as:
 *         output_height = (input_height + 2 * padding - kernel_height) / stride + 1
 *         output_width = (input_width + 2 * padding - kernel_width) / stride + 1
*/
torch::Tensor quantconv2d(
    const torch::Tensor & input,
    const torch::Tensor & input_des,
    const torch::Tensor & input_scale,
    const torch::Tensor & input_zero,
    const torch::Tensor & weight,
    const torch::Tensor & weight_des,
    const torch::Tensor & weight_scale,
    const torch::Tensor & weight_zero,
    const c10::optional<torch::Tensor> & bias,
    const int stride,
    const int padding)
{
    // Check the input
    CHECK_INPUT(input);
    CHECK_INPUT(input_des);
    CHECK_INPUT(input_scale);
    CHECK_INPUT(input_zero);
    CHECK_INPUT(weight);
    CHECK_INPUT(weight_des);
    CHECK_INPUT(weight_scale);
    CHECK_INPUT(weight_zero);
    if (bias.has_value()){
        CHECK_INPUT(bias.value());
    }

    // Parse descriptions
    auto input_n_bits = input_des[0].item<int>();
    auto input_sign = input_des[1].item<bool>();
    auto input_shape = input_des.slice(0, 2, 6, 1);
    auto weight_n_bits = weight_des[0].item<int>();
    auto weight_sign = weight_des[1].item<bool>();
    auto weight_shape = weight_des.slice(0, 2, 6, 1);

    // Get the input size
    auto batch_size = input_shape[0].item<int>();
    auto input_channel = input_shape[1].item<int>();
    auto input_height = input_shape[2].item<int>();
    auto input_width = input_shape[3].item<int>();

    // Get the weight size
    auto output_channel = weight_shape[0].item<int>();
    auto kernel_height = weight_shape[2].item<int>();
    auto kernel_width = weight_shape[3].item<int>();

    // Get the output size
    auto output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    auto output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

    // Allocate output tensor
    auto output = torch::zeros({batch_size, output_channel, output_height, output_width}, input.options());
    output = output.to(torch::kFloat32);

    // Get the offset
    unsigned char input_offset = input_sign ? 1 << (input_n_bits - 1) : 0;
    unsigned char weight_offset = weight_sign ? 1 << (weight_n_bits - 1) : 0;

    // Get the number of threads and blocks
    auto threads = 1024;
    auto blocks = (batch_size * output_channel * output_height * output_width + threads - 1) / threads;

    // Get the input and weight data types
    torch::ScalarType input_type = input_sign ? torch::kInt8 : torch::kUInt8;
    torch::ScalarType weight_type = weight_sign ? torch::kInt8 : torch::kUInt8;

    // Compute output
    AT_DISPATCH_ALL_TYPES(input_type, "quantconv2d_cuda_1", [&] {
        using input_t = scalar_t;
        AT_DISPATCH_ALL_TYPES(weight_type, "quantconv2d_cuda_2", [&] {
            using weight_t = scalar_t;
            quantconv2d_cuda_kernel<input_t, weight_t><<<blocks, threads>>>(
                input.data_ptr<unsigned char>(),
                input_scale.data_ptr<float>(),
                input_zero.data_ptr<float>(),
                input_scale.numel() == 1,
                input_n_bits,
                input_offset,
                weight.data_ptr<unsigned char>(),
                weight_scale.data_ptr<float>(),
                weight_zero.data_ptr<float>(),
                weight_scale.numel() == 1,
                weight_n_bits,
                weight_offset,
                bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
                output.data_ptr<float>(),
                batch_size,
                input_channel,
                input_height,
                input_width,
                output_channel,
                kernel_height,
                kernel_width,
                output_height,
                output_width,
                stride,
                padding);
        });
    });

    return output;
}
