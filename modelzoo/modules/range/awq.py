"""
AWQ range estimator
version: 0.0.1
update: 2024-04-17
"""
import torch
from torch import Tensor

from .minmax import MinMax


class AWQ(MinMax):
    """ AWQ range estimator.
        Activation-aware Weight Quantization (AWQ) is a range estimator that considers the activation statistics
        to scale the network weights and determine the quantization range by
        $Q(ws) \codt x/s = \Delta' Round(ws/\Delta) \cdot x/s$

    Args:
        n_bits (int): number of bits
        symmetric (bool): whether to use symmetric quantization
        signed (bool): whether to use signed quantization
        granularity (str): quantization granularity, channel or tensor
        q_group_size (int): quantization group size, Default: -1
        grid (int): number of grid points, Default: 20
        accumulate (bool): whether to accumulate the x_mean, Default: True.
            If True, update the x_mean by statistics of all input batches.
            If False, return the x_mean of current input batch.

    Returns:
        scale: quantization scale
        zero: quantization zero point
        qmin: quantization minimum value
        qmax: quantization maximum value
    
    """
    def __init__(
        self,
        n_bits: int,
        symmetric: bool,
        signed: bool,
        granularity: str,
        q_group_size: int = -1,
        grid: int = 20,
        accumulate: bool = True,
        **kwargs
    ):
        assert granularity == 'channel', 'AWQ only supports channel granularity'
        super().__init__(n_bits, symmetric, signed, granularity, 0.)
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.signed = signed
        self.granularity = granularity
        self.q_group_size = q_group_size
        self.grid = grid
        self.accumulate = accumulate
    
    def update(self, pre_act: Tensor, accumulate=True):
        """ update statistics of previous activations for calculating awq_scale
        Args:
            pre_act (torch.Tensor): input tensor of pre-activation, shape (N, C, ...)
            accumulate (bool): whether to accumulate the x_mean.
                If True, update the x_mean by statistics of all input batches.
                If False, return the x_mean of current input batch.
        Returns:
            torch.Tensor: updated x_mean
        """
        if pre_act.dim() <= 3:
            _x = pre_act.abs().view(-1, pre_act.shape[-1]).transpose(0, 1).contiguous()
        else:
            _x = pre_act.abs().transpose(0, 1).contiguous().view(pre_act.shape[1], -1)
        num_x = _x.shape[1]
        x_mean = _x.mean(1)
        
        if 'x_mean' not in self.__dict__ or not accumulate:
            self.x_mean = x_mean
            self.num_x = num_x
        else:
            self.x_mean = (self.x_mean * self.num_x + x_mean * num_x) / (self.num_x + num_x)
            self.num_x += num_x
        
        return self.x_mean

    @torch.no_grad()
    def __call__(self, flag: str, w: Tensor,
        pre_act: Tensor, func: callable, kwarg={}, **kwargs
    ):
        """ estimate the quantization range.
        Args:
            flag (str): flag for weight.
            w (torch.Tensor): input tensor of weights, shape (C, ...).
            pre_act (torch.Tensor): input tensor of pre-activation, shape (N, C, ...).
            func (callable): forward function of the module to simulate quantization,
                which takes w, pre_act and other arguments as input, e.g. torch.nn.functional.linear.
            kwarg (dict): other arguments for the forward function.
        """
        assert flag == 'weight', 'AWQ only supports weight quantization'

        if w.dim() == 2:
            awq_scale_shape = [1, -1]
            scale_shape = [-1, 1]
        elif w.dim() == 4:
            awq_scale_shape = [1, -1, 1, 1]
            scale_shape = [-1, 1, 1, 1]

        org_out = func(input=pre_act, weight=w, **kwarg)
        x_mean = self.update(pre_act, self.accumulate)

        best_error = float('inf')
        best_cfgs = None

        for ratio in range(self.grid):
            ratio = ratio * 1. / self.grid
            awq_scale = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            awq_scale = awq_scale / (awq_scale.max() * awq_scale.min()).sqrt()
            _w = w * awq_scale.view(*awq_scale_shape).to(w.device)

            org_w_shape = _w.shape
            if self.q_group_size > 0:
                assert org_w_shape[1] % self.q_group_size == 0, \
                    'Quantization group size should be divisible by the number of channels'
                _w = _w.reshape(-1, self.q_group_size, *_w.shape[2:]).flatten(1)
            wmin, wmax = self.range(_w, flag, accumulate=False)
            scale, zero, qmin, qmax = self.quantize(wmin, wmax)
            scale = scale.view(*scale_shape).to(w.device)
            zero = zero.view(*scale_shape).to(w.device)

            _w = ((_w/scale - zero).round().clamp(qmin, qmax) + zero).mul(scale)
            _w = _w.view(org_w_shape) / awq_scale.view(*awq_scale_shape).to(w.device)

            out = func(input=pre_act, weight=_w, **kwarg)
            loss = (org_out - out).float().pow(2).mean().item()

            if loss < best_error:
                best_error = loss
                best_cfgs = (scale, zero, qmin, qmax, awq_scale.view(*awq_scale_shape))

        return best_cfgs
