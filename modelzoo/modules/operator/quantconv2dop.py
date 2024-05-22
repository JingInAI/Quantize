"""
QuantConv2dOp
version: 0.0.2
update: 2024-02-27
"""
import torch
from torch.autograd import Function
from torch.nn import functional as F

from engine import (
    quantconv2d,
    quantconv2d_float_input,
)


class QuantConv2dOp1(Function):
    """ Quantized 2D convolution operator. """
    @staticmethod
    def forward(ctx, input, input_des, input_scale, input_zero,
                weight, weight_des, weight_scale, weight_zero,
                bias, stride, padding):
        """ Forward function.
        Arg "input" and "weight" are quantized 1D tensors with the type of torch.uint8.
        Arg "input_des" and "weight_des" are the description of the quantization.
        Arg "input_scale" and "weight_scale" are the scale of the quantization.
        Arg "input_zero" and "weight_zero" are the zero point of the quantization.
        Arg "bias" is the bias tensor, which is the type of torch.float32.
        Arg "stride" is the stride of the convolution, which is an integer.
        Arg "padding" is the padding of the convolution, which is an integer.
        """
        return quantconv2d(input, input_des, input_scale, input_zero, 
                           weight, weight_des, weight_scale, weight_zero, 
                           bias, stride, padding)
    
    @staticmethod
    def symbolic(g, input, input_des, input_scale, input_zero,
                 weight, weight_des, weight_scale, weight_zero,
                 bias, stride, padding):
        return g.op("QuantConv2dOp1", input, input_des, input_scale, input_zero, 
                    weight, weight_des, weight_scale, weight_zero, 
                    bias, stride_i=stride, padding_i=padding)


class QuantConv2dOp2(Function):
    """ Quantized 2D convolution operator. """
    @staticmethod
    def forward(ctx, input, weight, weight_des, weight_scale, weight_zero,
                bias, stride, padding):
        """ Forward function.
        Arg "input" is a 4D tensor with the type of torch.float32.
        Arg "weight" is a quantized 1D tensor with the type of torch.uint8.
        Arg "weight_des" is the description of the quantization.
        Arg "weight_scale" is the scale of the quantization.
        Arg "weight_zero" is the zero point of the quantization.
        Arg "bias" is the bias tensor, which is the type of torch.float32.
        Arg "stride" is the stride of the convolution, which is an integer.
        Arg "padding" is the padding of the convolution, which is an integer.
        """
        return quantconv2d_float_input(input, weight, weight_des, weight_scale, 
                                       weight_zero, bias, stride, padding)
    
    @staticmethod
    def symbolic(g, input, weight, weight_des, weight_scale, weight_zero,
                 bias, stride, padding):
        return g.op("QuantConv2dOp2", input, weight, weight_des, weight_scale,
                    weight_zero, bias, stride_i=stride, padding_i=padding)


def quantconv2d_forward(input, weight, bias, stride, padding, dilation, groups):
    """ Forward function for QuantConv2d module. """
    # Prepare input and weight
    if isinstance(input, tuple) or isinstance(input, list):
        input, input_des, input_scale, input_zero = input[:4]
    if isinstance(weight, tuple) or isinstance(weight, list):
        weight, weight_des, weight_scale, weight_zero = weight[:4]

    # Compute Conv2d
    if input.dtype == torch.float32 and weight.dtype == torch.float32:
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    # Prepare stride and padding
    if isinstance(stride, tuple) or isinstance(stride, list):
        stride = stride[0]
    if isinstance(padding, tuple) or isinstance(padding, list):
        padding = padding[0]

    # Compute QuantConv2dOps
    if input.dtype == torch.uint8 and weight.dtype == torch.uint8:
        return QuantConv2dOp1.apply(input, input_des, input_scale, input_zero,
                                    weight, weight_des, weight_scale, weight_zero,
                                    bias, stride, padding)
    
    if input.dtype == torch.float32 and weight.dtype == torch.uint8:
        return QuantConv2dOp2.apply(input, weight, weight_des, weight_scale,
                                    weight_zero, bias, stride, padding)
    
    raise ValueError("Unsupported input and weight types.")
