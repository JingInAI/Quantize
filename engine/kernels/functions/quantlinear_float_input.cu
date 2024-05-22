/**
 * @file quantlinear_float_input.cu
 * @brief The kernel file for quantlinear_float_input.
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
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float tensor")

#define BLOCKSIZE 32


/**
 * The CUDA kernel of quantlinear_float_input.
 * 
 * @param input The input tensor of shape (batch_size, input_size), which is a float tensor.
 * @param weight The weight tensor of 1D shape, which contains elements of unsigned char.
 * @param weight_scale The scale of weight tensor, which is one element tensor (per_tensor) or 1D tensor (per_channel).
 * @param weight_zero The zero of weight tensor, which is one element tensor (per_tensor) or 1D tensor (per_channel).
 * @param weight_per_tensor Whether the weight tensor is per tensor or per channel.
 * @param weight_n_bits The number of bits of weight tensor.
 * @param weight_offset The offset of weight tensor.
 * @param bias The bias tensor of shape (output_size) or None.
 * @param output The output tensor of shape (batch_size, output_size).
 * @param batch_size The batch size of input tensor.
 * @param input_size The input size of input tensor.
 * @param output_size The output size of output tensor.
*/
template <typename weight_t>
__global__ void quantlinear_float_input_cuda(
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
    const int input_size,
    const int output_size)
{
    // Get the row and column
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    auto col = blockIdx.x * blockDim.x + threadIdx.x;

    // Allocate the shared memory
    __shared__ float input_shared[BLOCKSIZE][BLOCKSIZE];
    __shared__ float weight_shared[BLOCKSIZE][BLOCKSIZE];

    // Initialize the output tensor
    float output_value = 0.0;

    // Loop over the input size
    #pragma unroll
    for (auto i = 0; i < input_size; i += BLOCKSIZE){
        // Load the input tensor
        if (i + threadIdx.x < input_size && row < batch_size){
            input_shared[threadIdx.y][threadIdx.x] = input[row * input_size + i + threadIdx.x];
        }

        // Load the weight tensor
        if (i + threadIdx.y < input_size && col < output_size){
            auto ele_idx = col * input_size + i + threadIdx.y;
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
                (_weight_value - weight_zero[col]) * weight_scale[col];
            
            weight_shared[threadIdx.y][threadIdx.x] = weight_value_f;
        }

        __syncthreads();

        // Loop over the shared memory
        #pragma unroll
        for (auto j = 0; j < BLOCKSIZE; j++){
            output_value += input_shared[threadIdx.y][j] * weight_shared[j][threadIdx.x];
        }

        __syncthreads();
    }

    // Load the bias tensor
    if (row < batch_size && col < output_size){
        output[row * output_size + col] = output_value + (bias ? bias[col] : 0.0);
    }
}


/**
 * Quantized linear function with float input.
 * 
 * @param input The input tensor of shape (batch_size, input_size), which is a float tensor.
 * @param weight The weight tensor of 1D shape, which contains elements of unsigned char.
 * @param weight_des The description of weight tensor, which is a 1D tensor of [n_bits, signed, *shape].
 * @param weight_scale The scale of weight tensor, which is one element tensor (per_tensor) or 1D tensor (per_channel).
 * @param weight_zero The zero of weight tensor, which is one element tensor (per_tensor) or 1D tensor (per_channel).
 * @param bias The bias tensor of shape (output_size) or None.
 * @return The output tensor of shape (batch_size, output_size).
 *         The output is calculated as:
 *         output = input * weight^T + bias
*/
torch::Tensor quantlinear_float_input(
    const torch::Tensor & input,
    const torch::Tensor & weight,
    const torch::Tensor & weight_des,
    const torch::Tensor & weight_scale,
    const torch::Tensor & weight_zero,
    const c10::optional<torch::Tensor> & bias)
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
    auto weight_shape = weight_des.slice(0, 2, 4, 1);

    // Get the input size
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    
    // Get the output size
    auto output_size = weight_shape[0].item<int>();

    // Allocate the output tensor
    auto output = torch::zeros({batch_size, output_size}, input.options());

    // Get the offset
    unsigned char weight_offset = weight_sign ? 1 << (weight_n_bits - 1) : 0;

    // Get the threads and blocks
    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    dim3 blocks((output_size + BLOCKSIZE - 1) / BLOCKSIZE, (batch_size + BLOCKSIZE - 1) / BLOCKSIZE);

    // Get the type of weight tensor
    torch::ScalarType weight_type = weight_sign ? torch::kInt8 : torch::kUInt8;

    // Call the kernel function
    AT_DISPATCH_ALL_TYPES(weight_type, "quantlinear_float_input_cuda", [&] {
        quantlinear_float_input_cuda<scalar_t><<<blocks, threads>>>(
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
            input_size,
            output_size);
    });

    return output;
}
