"""
QuantLinearOp
version: 0.0.1
update: 2024-02-27
"""
import torch
from torch.autograd import Function
from torch.nn import functional as F

from engine import (
    quantlinear,
    quantlinear_float_input,
)


class QuantLinearOp1(Function):
    """ Quantized linear operator. """
    @staticmethod
    def forward(ctx, input, input_des, input_scale, input_zero,
                weight, weight_des, weight_scale, weight_zero, bias):
        """ Forward function.
        Arg "input" and "weight" are quantized 1D tensors with the type of torch.uint8.
        Arg "input_des" and "weight_des" are the description of the quantization.
        Arg "input_scale" and "weight_scale" are the scale of the quantization.
        Arg "input_zero" and "weight_zero" are the zero point of the quantization.
        Arg "bias" is the bias tensor, which is the type of torch.float32.
        """
        return quantlinear(input, input_des, input_scale, input_zero, 
                           weight, weight_des, weight_scale, weight_zero, bias)
    
    @staticmethod
    def symbolic(g, input, input_des, input_scale, input_zero,
                 weight, weight_des, weight_scale, weight_zero, bias):
        return g.op("QuantLinearOp1", input, input_des, input_scale, input_zero, 
                    weight, weight_des, weight_scale, weight_zero, bias)


class QuantLinearOp2(Function):
    """ Quantized linear operator. """
    @staticmethod
    def forward(ctx, input, weight, weight_des, weight_scale, weight_zero, bias):
        """ Forward function.
        Arg "input" is a 2D tensor with the type of torch.float32.
        Arg "weight" is a quantized 1D tensor with the type of torch.uint8.
        Arg "weight_des" is the description of the quantization.
        Arg "weight_scale" is the scale of the quantization.
        Arg "weight_zero" is the zero point of the quantization.
        Arg "bias" is the bias tensor, which is the type of torch.float32.
        """
        return quantlinear_float_input(input, weight, weight_des, weight_scale, 
                                       weight_zero, bias)
    
    @staticmethod
    def symbolic(g, input, weight, weight_des, weight_scale, weight_zero, bias):
        return g.op("QuantLinearOp2", input, weight, weight_des, weight_scale, 
                    weight_zero, bias)


def quantlinear_forward(input, weight, bias):
    """ Forward function for QuantLinear module. """
    # Prepare input and weight
    if isinstance(input, tuple) or isinstance(input, list):
        input, input_des, input_scale, input_zero = input[:4]
    if isinstance(weight, tuple) or isinstance(weight, list):
        weight, weight_des, weight_scale, weight_zero = weight[:4]

    # Compute Linear
    if input.dtype == torch.float32 and weight.dtype == torch.float32:
        return F.linear(input, weight, bias)
    
    if input.dtype == torch.uint8 and weight.dtype == torch.uint8:
        return QuantLinearOp1.apply(input, input_des, input_scale, input_zero,
                                    weight, weight_des, weight_scale, weight_zero,
                                    bias)
    
    if input.dtype == torch.float32 and weight.dtype == torch.uint8:
        return QuantLinearOp2.apply(input, weight, weight_des, weight_scale,
                                    weight_zero, bias)
    
    raise ValueError("Unsupported input and weight types.")
