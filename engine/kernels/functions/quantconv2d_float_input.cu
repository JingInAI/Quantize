/**
 * @file quantconv2d_float_input.cu
 * @brief The CUDA kernel of quantconv2d_float_input.
 * @version 0.0.1
 * @date 2024-02-26
*/

#include <cuda.h>
#include <torch/extension.h>
#include <vector>
#include "funcs.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be a float tensor")

#define BLOCKSIZE 32


/**
 * The CUDA kernel of quantconv2d_float_input.
 * 
 * @param input The input tensor of shape (batch_size, input_channel, input_height, input_width), which is a float tensor.
 * @param weight The weight tensor of 1D shape, which contains elements of unsigned char.
 * @param weight_scale The scale of weight tensor, which is one element tensor (per_tensor) or 1D tensor (per_channel).
 * @param weight_zero The zero of weight tensor, which is one element tensor (per_tensor) or 1D tensor (per_channel).
 * @param weight_per_tensor Whether the weight tensor is per tensor or per channel.
 * @param weight_n_bits The number of bits of weight tensor.
 * @param weight_offset The offset of weight tensor.
 * @param bias The bias tensor of shape (output_channel) or None.
 * @param output The output tensor of shape (batch_size, output_channel, output_height, output_width).
 * @param batch_size The batch size of input tensor.
 * @param input_channel The input channel of input tensor.
 * @param input_height The input height of input tensor.
 * @param input_width The input width of input tensor.
 * @param output_channel The output channel of output tensor.
 * @param kernel_height The kernel height of weight tensor.
 * @param kernel_width The kernel width of weight tensor.
 * @param output_height The output height of output tensor.
 * @param output_width The output width of output tensor.
 * @param stride The stride of convolution.
 * @param padding The padding of convolution.
*/
template <typename weight_t>
__global__ void quantconv2d_float_input_cuda(
    const float * input,
    const unsigned char * weight,
    const float * weight_scale,
    const float * weight_zero,
    const bool weight_per_tensor,
    const int weight_n_bits,
    const unsigned char weight_offset,
    const float * bias,
    float * output,
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

        // Get the channel index
        auto output_index = index % (output_channel * output_height * output_width);
        auto outw = output_index % output_width;
        auto outh = (output_index / output_width) % output_height;
        auto outc = (output_index / (output_width * output_height)) % output_channel;

        // Compute the output
        float output_value = bias == nullptr ? 0 : bias[outc];

        #pragma unroll
        for (auto inc = 0; inc < input_channel; inc++){
            for (auto keh = 0; keh < kernel_height; keh++){
                for (auto kew = 0; kew < kernel_width; kew++){
                    int inh = outh * stride - padding + keh;
                    int inw = outw * stride - padding + kew;

                    if (inh >= 0 && inh < input_height && inw >= 0 && inw < input_width){
                        // Get the weight value
                        auto ele_idx = outc * input_channel * kernel_height * kernel_width + inc * kernel_height * kernel_width + keh * kernel_width + kew;
                        auto byte_idx = ele_idx * weight_n_bits / 8;
                        auto bit_idx = ele_idx * weight_n_bits % 8;
                        unsigned char weight_value = (weight[byte_idx] >> bit_idx) & ((1 << weight_n_bits) - 1);
                        if (bit_idx + weight_n_bits > 8)
                            weight_value |= (weight[byte_idx + 1] << (8 - bit_idx)) & ((1 << weight_n_bits) - 1);
                        
                        // Dequantize the weight value
                        weight_value -= weight_offset;
                        weight_t _weight_value = (weight_t)weight_value;
                        float weight_value_f = weight_per_tensor ?
                            (_weight_value - weight_zero[0]) * weight_scale[0] :
                            (_weight_value - weight_zero[outc]) * weight_scale[outc];
                        
                        // Get the input value
                        float input_value_f = input[batch * input_channel * input_height * input_width + inc * input_height * input_width + inh * input_width + inw];

                        // Compute the output value
                        output_value += input_value_f * weight_value_f;
                    }
                }
            }
        }

        // Store the output value
        output[index] = output_value;
    }
}


/**
 * Quantized Conv2d function with float input.
 * 
 * @param input The input tensor of shape (batch_size, input_channel, input_height, input_width), which is a float tensor.
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
torch::Tensor quantconv2d_float_input(
    const torch::Tensor & input,
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
    CHECK_FLOAT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(weight_des);
    CHECK_INPUT(weight_scale);
    CHECK_INPUT(weight_zero);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    // Parse the description of weight tensor
    auto weight_n_bits = weight_des[0].item<int>();
    auto weight_sign = weight_des[1].item<bool>();
    auto weight_shape = weight_des.slice(0, 2, 6, 1);

    // Get the input size
    auto batch_size = input.size(0);
    auto input_channel = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);

    // Get the weight size
    auto output_channel = weight_shape[0].item<int>();
    auto kernel_height = weight_shape[2].item<int>();
    auto kernel_width = weight_shape[3].item<int>();

    // Get the output size
    auto output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    auto output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

    // Allocate the output tensor
    auto output = torch::zeros({batch_size, output_channel, output_height, output_width}, input.options());

    // Get the offset
    unsigned char weight_offset = weight_sign ? 1 << (weight_n_bits - 1) : 0;

    // Get the number of threads and blocks
    auto threads = 1024;
    auto blocks = (batch_size * output_channel * output_height * output_width + threads - 1) / threads;

    // Get the input and weight data types
    torch::ScalarType weight_type = weight_sign ? torch::kInt8 : torch::kUInt8;

    // Launch the CUDA kernel
    AT_DISPATCH_ALL_TYPES(weight_type, "quantconv2d_float_input_cuda", [&] {
        quantconv2d_float_input_cuda<scalar_t><<<blocks, threads>>>(
            input.data_ptr<float>(),
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

    return output;
}
