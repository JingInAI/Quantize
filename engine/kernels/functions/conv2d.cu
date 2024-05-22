/**
 * @file conv2d.cu
 * @brief Conv2d function implementation.
 * @version 0.0.1
 * @date 2024-02-22
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
 * Conv2d CUDA kernel.
 *
 * @param input The input tensor of shape (batch_size, input_channel, input_height, input_width).
 * @param weight The weight tensor of shape (output_channel, input_channel, kernel_height, kernel_width).
 * @param bias The bias tensor of shape (output_channel).
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
template <typename scalar_t>
__global__ void conv2d_cuda_kernel_1(
    const scalar_t * __restrict__ input,
    const scalar_t * __restrict__ weight,
    const scalar_t * __restrict__ bias,
    scalar_t * __restrict__ output,
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

        // Get the output index
        auto output_index = index % (output_channel * output_height * output_width);
        auto outw = output_index % output_width;
        auto outh = (output_index / output_width) % output_height;
        auto outc = output_index / (output_height * output_width);

        // Compute the output
        scalar_t output_value = 0;
        if (bias != nullptr){
            output_value = bias[outc];
        }

        #pragma unroll
        for (auto inc = 0; inc < input_channel; inc++){
            for (auto keh = 0; keh < kernel_height; keh++){
                for (auto kew = 0; kew < kernel_width; kew++){
                    int inh = outh * stride + keh - padding;
                    int inw = outw * stride + kew - padding;
                    if (inh >= 0 && inh < input_height && inw >= 0 && inw < input_width){
                        output_value += input[batch * input_channel * input_height * input_width + inc * input_height * input_width + inh * input_width + inw] * weight[outc * input_channel * kernel_height * kernel_width + inc * kernel_height * kernel_width + keh * kernel_width + kew];
                    }
                }
            }
        }

        // Store the output
        output[index] = output_value;
    }
}


/**
 * Conv2d CUDA kernel.
 * 
 * @param input The input tensor of shape (batch_size, input_channel, input_height, input_width).
 * @param weight The weight tensor of shape (output_channel, input_channel, kernel_height, kernel_width).
 * @param bias The bias tensor of shape (output_channel).
 * @param output The output tensor of shape (batch_size, output_channel, output_height, output_width).
 * @param batch_size The batch size, input.size(0).
 * @param input_channel The input channel, input.size(1) and weight.size(1).
 * @param input_height The input height, input.size(2).
 * @param input_width The input width, input.size(3).
 * @param output_channel The output channel, weight.size(0).
 * @param kernel_height The kernel height, weight.size(2).
 * @param kernel_width The kernel width, weight.size(3).
 * @param output_height The output height, calculated as (input_height + 2 * padding - kernel_height) / stride + 1.
 * @param output_width The output width, calculated as (input_width + 2 * padding - kernel_width) / stride + 1.
 * @param stride The stride of convolution.
 * @param padding The padding of convolution.
*/
template <typename scalar_t>
__global__ void conv2d_cuda_kernel_2(
    const scalar_t * __restrict__ input,
    const scalar_t * __restrict__ weight,
    const scalar_t * __restrict__ bias,
    scalar_t * __restrict__ output,
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
    // Get the batch and channel of output channel
    auto index = blockIdx.z;
    auto batch = index / output_channel;
    auto outc = index % output_channel;

    // Get the row and col of output tensor
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    auto col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the index is valid
    if (row < output_height && col < output_height) {

        // Create shared memory
        // const int shared_height = BLOCKSIZE * stride + kernel_height - 1;
        // const int shared_width = BLOCKSIZE * stride + kernel_width - 1;
        // __shared__ scalar_t input_shared[BLOCKSIZE * stride + kernel_height - 1][BLOCKSIZE * stride + kernel_width - 1];
        // __shared__ scalar_t weight_shared[kernel_height][kernel_width];
        __shared__ scalar_t input_shared[2*BLOCKSIZE][2*BLOCKSIZE];
        __shared__ scalar_t weight_shared[BLOCKSIZE][BLOCKSIZE];

        // Load bias and initialize output
        scalar_t output_value = 0;
        if (bias != nullptr){
            output_value = bias[outc];
        }

        #pragma unroll
        for (auto inc = 0; inc < input_channel; inc++){
            // Load weight
            if (threadIdx.y < kernel_height && threadIdx.x < kernel_width){
                weight_shared[threadIdx.y][threadIdx.x] = weight[outc * input_channel * kernel_height * kernel_width + inc * kernel_height * kernel_width + threadIdx.y * kernel_width + threadIdx.x];
            }
            __syncthreads();

            // Load input
            #pragma unroll
            for (auto keh = 0; keh < stride && keh < kernel_height; keh++){
                for (auto kew = 0; kew < stride && kew < kernel_width; kew++){
                    int inh = row * stride + keh - padding;
                    int inw = col * stride + kew - padding;
                    if (inh >= 0 && inh < input_height && inw >= 0 && inw < input_width){
                        input_shared[threadIdx.y * stride + keh][threadIdx.x * stride + kew] =
                        input[batch * input_channel * input_height * input_width + inc * input_height * input_width + inh * input_width + inw];
                    } else {
                        input_shared[threadIdx.y * stride + keh][threadIdx.x * stride + kew] = 0;
                    }
                }
            }

            // Load the boundary
            if (threadIdx.y == BLOCKSIZE - 1 || threadIdx.x == BLOCKSIZE - 1){
                #pragma unroll
                for (auto keh = 0; keh < kernel_height - stride; keh++){
                    for (auto kew = 0; kew < kernel_width - stride; kew++){
                        int inh = row * stride + keh + stride - padding;
                        int inw = col * stride + kew + stride - padding;
                        if (inh >= 0 && inh < input_height && inw >= 0 && inw < input_width){
                            input_shared[threadIdx.y * stride + keh + stride][threadIdx.x * stride + kew + stride] =
                            input[batch * input_channel * input_height * input_width + inc * input_height * input_width + inh * input_width + inw];
                        } else {
                            input_shared[threadIdx.y * stride + keh + stride][threadIdx.x * stride + kew + stride] = 0;
                        }
                    }
                }
            }
            __syncthreads();

            // Compute output
            #pragma unroll
            for (auto keh = 0; keh < kernel_height; keh++){
                for (auto kew = 0; kew < kernel_width; kew++){
                    output_value += input_shared[threadIdx.y * stride + keh][threadIdx.x * stride + kew] * weight_shared[keh][kew];
                }
            }
            __syncthreads();
        }

        // Store the output
        output[batch * output_channel * output_height * output_width + outc * output_height * output_width + row * output_width + col] = output_value;
    }
}


/**
 * Conv2d function.
 * 
 * @param input The input tensor of shape (batch_size, input_channel, input_height, input_width).
 * @param weight The weight tensor of shape (output_channel, input_channel, kernel_height, kernel_width).
 * @param bias The bias tensor of shape (output_channel).
 * @param stride The stride of convolution.
 * @param padding The padding of convolution.
 * @param mode The mode of cuda kernel.
 * @return The output tensor of shape (batch_size, output_channel, output_height, output_width).
 *         The output_height and output_width are calculated as:
 *         output_height = (input_height + 2 * padding - kernel_height) / stride + 1
 *         output_width = (input_width + 2 * padding - kernel_width) / stride + 1
*/
torch::Tensor conv2d(
    const torch::Tensor & input,
    const torch::Tensor & weight,
    const c10::optional<torch::Tensor> & bias,
    const int stride,
    const int padding,
    const int mode)
{
    // Check the input
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.has_value()){
        CHECK_INPUT(bias.value());
    }

    // Get the input size
    auto batch_size = input.size(0);
    auto input_channel = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);

    // Get the weight size
    auto output_channel = weight.size(0);
    auto kernel_height = weight.size(2);
    auto kernel_width = weight.size(3);

    // Calculate the output size
    auto output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    auto output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

    // Create the output tensor
    auto output = torch::empty({batch_size, output_channel, output_height, output_width}, input.options());

    // Call the CUDA kernel
    if (mode == 0) {
        auto threads = 1024;
        auto blocks = (batch_size * output_channel * output_height * output_width + threads - 1) / threads;
        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "conv2d_cuda", ([&] {
            conv2d_cuda_kernel_1<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(),
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
        }));
    } else if (mode == 1) {
        dim3 threads(BLOCKSIZE, BLOCKSIZE);
        dim3 blocks((output_width + threads.x - 1) / threads.x, (output_height + threads.y - 1) / threads.y, batch_size * output_channel);
        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "conv2d_cuda", ([&] {
            conv2d_cuda_kernel_2<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(),
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
        }));
    }

    return output;
}
