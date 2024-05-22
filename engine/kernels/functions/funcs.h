#pragma once 
#include <torch/extension.h>

#include <vector>


/**
 * Linear function.
 *
 * @param input The input tensor.
 * @param weight The weight tensor.
 * @param bias The bias tensor.
 * @param mode The mode of cuda kernel.
 * @return The output tensor, defined as:
 *         output = input * weight^T + bias
 */
torch::Tensor linear(
    const torch::Tensor & input,
    const torch::Tensor & weight,
    const c10::optional<torch::Tensor> & bias={},
    const int mode=0);


/**
 * Quantized linear function.
 *
 * @param input The input tensor of shape (batch_size, input_size).
 * @param input_des The description of input tensor, which is a 1D tensor of size 4.
 * @param input_scale The scale of input tensor.
 * @param input_zero The zero of input tensor.
 * @param weight The weight tensor of shape (output_size, input_size).
 * @param weight_des The description of weight tensor, which is a 1D tensor of size 4.
 * @param weight_scale The scale of weight tensor.
 * @param weight_zero The zero of weight tensor.
 * @param bias The bias tensor of shape (output_size).
 */
torch::Tensor quantlinear(
    const torch::Tensor & input,
    const torch::Tensor & input_des,
    const torch::Tensor & input_scale,
    const torch::Tensor & input_zero,
    const torch::Tensor & weight,
    const torch::Tensor & weight_des,
    const torch::Tensor & weight_scale,
    const torch::Tensor & weight_zero,
    const c10::optional<torch::Tensor> & bias);


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
    const c10::optional<torch::Tensor> & bias);


/**
 * Conv2d function.
 * 
 * @param input The input tensor of shape (batch_size, input_channel, input_height, input_width).
 * @param weight The weight tensor of shape (output_channel, input_channel, kernel_height, kernel_width).
 * @param bias The bias tensor of shape (output_channel).
 * @param stride The stride of convolution.
 * @param padding The padding of convolution.
 * @param mode The mode of convolution.
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
    const int mode);


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
    const int padding);


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
    const int padding);
