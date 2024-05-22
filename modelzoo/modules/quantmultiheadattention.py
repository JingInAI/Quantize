"""
Quantized torch.nn.MultiheadAttention
version: 0.0.5
update: 2024-04-19
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing import Optional, Tuple

from .quantizer import Quantizer
from .range import BiasCorrect
from engine import tpack, tunpack


class QuantMultiheadAttention(nn.MultiheadAttention):
    """ Quantized multi-head attention module

    Args:
        embed_dim: total dimension of the model
        num_heads: parallel attention heads
        dropout: a Dropout layer on attn_output_weights. Default: 0.0
        bias: add bias as module parameter. Default: True
        add_bias_kv: add bias to the key and value sequences at dim=0.
            Default: False
        add_zero_attn: add a new batch of zeros to the key and
            value sequences at dim=1. Default: False
        kdim: total number of features in key. Default: None
        vdim: total number of features in value. Default: None
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``
        w_setting: weight quantization setting
            n_bits: number of bits
            symmetric: whether to use symmetric quantization
            signed: whether to use signed quantization
            granularity: quantization granularity, channel or tensor
            range: quantization range, min_max
        a_setting: activation quantization setting
            n_bits: number of bits
            symmetric: whether to use symmetric quantization
            signed: whether to use signed quantization
            granularity: quantization granularity, channel or tensor
            range: quantization range, min_max
        bias_correct: whether to correct bias with calibration data
            momentum: momentum of expected mean, Default: 0.1
        device: device type
        dtype: data type

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size,
            S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length,
            S is the source sequence length.
        - Output:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
    
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        w_setting: dict = {},
        a_setting: dict = {},
        bias_correct: dict = {},
        device=None,
        dtype=None,
        **kwargs
    ):
        bias = kwargs['_parameters']['in_proj_bias'] is not None
        add_bias_kv = 'bias_k' in kwargs['_parameters'] and 'bias_v' in kwargs['_parameters']
        super().__init__(
            embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
            kdim, vdim, batch_first, device, dtype)

        self.q_quantizer = Quantizer(**a_setting, flag='activation', n_channels=embed_dim, dim=3)
        self.k_quantizer = Quantizer(**a_setting, flag='activation', n_channels=embed_dim, dim=3)
        self.v_quantizer = Quantizer(**a_setting, flag='activation', n_channels=embed_dim, dim=3)

        if not self._qkv_same_embed_dim:
            self.q_proj_weight.data = kwargs['_parameters']['q_proj_weight']
            self.k_proj_weight.data = kwargs['_parameters']['k_proj_weight']
            self.v_proj_weight.data = kwargs['_parameters']['v_proj_weight']
            weight_dim = self.q_proj_weight.dim()
        else:
            self.in_proj_weight.data = kwargs['_parameters']['in_proj_weight']
            weight_dim = self.in_proj_weight.dim()

        self.q_proj_quantizer = Quantizer(**w_setting, flag='weight', n_channels=embed_dim, dim=weight_dim)
        self.k_proj_quantizer = Quantizer(**w_setting, flag='weight', n_channels=embed_dim, dim=weight_dim)
        self.v_proj_quantizer = Quantizer(**w_setting, flag='weight', n_channels=embed_dim, dim=weight_dim)

        self.out_proj.weight.data = kwargs['_modules']['out_proj'].weight.data
        # TODO: add awq to out_proj layer
        w_setting['range'] = {'name': 'mse', 'maxshrink': 0.8, 'grid': 100}
        self.out_proj_quantizer = Quantizer(**w_setting, flag='weight', n_channels=embed_dim, dim=self.out_proj.weight.dim())

        if bias:
            self.in_proj_bias.data = kwargs['_parameters']['in_proj_bias']
            self.out_proj.bias.data = kwargs['_modules']['out_proj'].bias.data
        elif bias_correct:
            self.in_proj_bias = nn.Parameter(torch.zeros(3*embed_dim, device=device, dtype=dtype))
            self.out_proj.bias = nn.Parameter(torch.zeros(embed_dim, device=device, dtype=dtype))
        
        if bias_correct:
            self.q_corrector = BiasCorrect(**bias_correct)
            self.k_corrector = BiasCorrect(**bias_correct)
            self.v_corrector = BiasCorrect(**bias_correct)
            # TODO: add bias corrector to out_proj layer
            # self.out_corrector = BiasCorrect(**bias_correct)

        self.bias_correct = bias_correct
        self.calibrating = False
        self.packed = False

    def calibrate(self, query: Tensor, key: Tensor, value: Tensor):
        """ calibrate weight and activation quantizer.
        Args:
            query (Tensor): query tensor
            key (Tensor): key tensor
            value (Tensor): value tensor
        """
        self.q_quantizer.calibrate(query.detach().clone())
        self.k_quantizer.calibrate(key.detach().clone())
        self.v_quantizer.calibrate(value.detach().clone())

        if not self._qkv_same_embed_dim:
            w_q = self.q_proj_weight.detach().clone()
            w_k = self.k_proj_weight.detach().clone()
            w_v = self.v_proj_weight.detach().clone()
        else:
            w_q, w_k, w_v = self.in_proj_weight.detach().clone().chunk(3)
        
        self.q_proj_quantizer.calibrate(
            w_q, pre_act=query.detach().clone(), func=F.linear)
        self.k_proj_quantizer.calibrate(
            w_k, pre_act=key.detach().clone(), func=F.linear)
        self.v_proj_quantizer.calibrate(
            w_v, pre_act=value.detach().clone(), func=F.linear)

        self.out_proj_quantizer.calibrate(self.out_proj.weight.detach().clone())

        if self.bias_correct:
            self.q_corrector.calibrate(query.detach().clone())
            self.k_corrector.calibrate(key.detach().clone())
            self.v_corrector.calibrate(value.detach().clone())

    def pack(self, calibrated=True):
        """ Pack the quantized layer.
            Weights are quantized and add scale and zero_point.
        Args:
            calibrated (bool): the layer is calibrated or not
        """
        self.requires_grad_(False)
        self.q_quantizer.pack(None)
        self.k_quantizer.pack(None)
        self.v_quantizer.pack(None)

        if not self._qkv_same_embed_dim:
            w_q, w_k, w_v = self.q_proj_weight, self.k_proj_weight, self.v_proj_weight
        else:
            w_q, w_k, w_v = self.in_proj_weight.chunk(3)

        if self.bias_correct and calibrated:
            qw_q = self.q_proj_quantizer(w_q)
            qw_k = self.k_proj_quantizer(w_k)
            qw_v = self.v_proj_quantizer(w_v)
                        
            q_bias = self.q_corrector(None, w_q - qw_q, func=F.linear)
            k_bias = self.k_corrector(None, w_k - qw_k, func=F.linear)
            v_bias = self.v_corrector(None, w_v - qw_v, func=F.linear)

            self.in_proj_bias.data += torch.cat([q_bias.mean(0), k_bias.mean(0), v_bias.mean(0)], dim=0)
        self.q_corrector, self.k_corrector, self.v_corrector = None, None, None
        
        q_proj_weight, q_proj_scale, q_proj_zero = self.q_proj_quantizer.pack(w_q)
        k_proj_weight, k_proj_scale, k_proj_zero = self.k_proj_quantizer.pack(w_k)
        v_proj_weight, v_proj_scale, v_proj_zero = self.v_proj_quantizer.pack(w_v)
        self.q_proj_weight.data, q_proj_des = \
            tpack(q_proj_weight, self.q_proj_quantizer.n_bits, self.q_proj_quantizer.signed)
        self.k_proj_weight.data, k_proj_des = \
            tpack(k_proj_weight, self.k_proj_quantizer.n_bits, self.k_proj_quantizer.signed)
        self.v_proj_weight.data, v_proj_des = \
            tpack(v_proj_weight, self.v_proj_quantizer.n_bits, self.v_proj_quantizer.signed)
        self.register_buffer('q_proj_scale', q_proj_scale)
        self.register_buffer('q_proj_zero', q_proj_zero)
        self.register_buffer('q_proj_des', q_proj_des)
        self.register_buffer('k_proj_scale', k_proj_scale)
        self.register_buffer('k_proj_zero', k_proj_zero)
        self.register_buffer('k_proj_des', k_proj_des)
        self.register_buffer('v_proj_scale', v_proj_scale)
        self.register_buffer('v_proj_zero', v_proj_zero)
        self.register_buffer('v_proj_des', v_proj_des)
        self.q_proj_quantizer = None
        self.k_proj_quantizer = None
        self.v_proj_quantizer = None

        out_proj_weight, out_proj_scale, out_proj_zero = self.out_proj_quantizer.pack(self.out_proj.weight)
        self.out_proj.weight.data, out_proj_des = \
            tpack(out_proj_weight, self.out_proj_quantizer.n_bits, self.out_proj_quantizer.signed)
        self.register_buffer('out_proj_scale', out_proj_scale)
        self.register_buffer('out_proj_zero', out_proj_zero)
        self.register_buffer('out_proj_des', out_proj_des)
        self.out_proj_quantizer = None

        self.packed = True
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True, attn_mask: Optional[Tensor] = None, average_attn_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        if self.calibrating:
            self.calibrate(query, key, value)
        
        if not self.packed:
            query, key, value = \
                self.q_quantizer(query), self.k_quantizer(key), self.v_quantizer(value)

            if not self._qkv_same_embed_dim:
                w_q, w_k, w_v = self.q_proj_weight, self.k_proj_weight, self.v_proj_weight        
            else:
                w_q, w_k, w_v = self.in_proj_weight.chunk(3)

            qw_q, qw_k, qw_v = \
                self.q_proj_quantizer(w_q), self.k_proj_quantizer(w_k), self.v_proj_quantizer(w_v)

            in_proj_bias = self.in_proj_bias
            if self.bias_correct:
                q_bias = self.q_corrector(query, w_q - qw_q, func=F.linear)
                k_bias = self.k_corrector(key, w_k - qw_k, func=F.linear)
                v_bias = self.v_corrector(value, w_v - qw_v, func=F.linear)
                in_proj_bias = in_proj_bias + \
                    torch.cat([q_bias.mean(0), k_bias.mean(0), v_bias.mean(0)], dim=0)

            if not self._qkv_same_embed_dim:
                q_proj_weight, k_proj_weight, v_proj_weight = qw_q, qw_k, qw_v
                in_proj_weight = self.in_proj_weight
            else:
                in_proj_weight = torch.cat([qw_q, qw_k, qw_v], dim=0)
                q_proj_weight, k_proj_weight, v_proj_weight = \
                    self.q_proj_weight, self.k_proj_weight, self.v_proj_weight
            
            out_proj_weight = self.out_proj_quantizer(self.out_proj.weight)
            
        else:
            query, q_scale, q_zero = self.q_quantizer(query)
            key, k_scale, k_zero = self.k_quantizer(key)
            value, v_scale, v_zero = self.v_quantizer(value)
            query = (query + q_zero).mul_(q_scale)
            key = (key + k_zero).mul_(k_scale)
            value = (value + v_zero).mul_(v_scale)

            qw_q = (self.q_proj_weight + self.q_proj_zero).mul_(self.q_proj_scale)
            qw_k = (self.k_proj_weight + self.k_proj_zero).mul_(self.k_proj_scale)
            qw_v = (self.v_proj_weight + self.v_proj_zero).mul_(self.v_proj_scale)

            if not self._qkv_same_embed_dim:
                q_proj_weight, k_proj_weight, v_proj_weight = qw_q, qw_k, qw_v
                in_proj_weight = self.in_proj_weight
            else:
                in_proj_weight = torch.cat([qw_q, qw_k, qw_v], dim=0)
                q_proj_weight, k_proj_weight, v_proj_weight = \
                    self.q_proj_weight, self.k_proj_weight, self.v_proj_weight
            
            out_proj_weight = (self.out_proj.weight + self.out_proj_zero).mul_(self.out_proj_scale)
            in_proj_bias = self.in_proj_bias
        
        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif in_proj_weight is not None and query.dtype != in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif query.is_nested and key_padding_mask is not None:
            why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"
        elif self.num_heads % 2 == 1:
            why_not_fast_path = "num_heads is odd"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                in_proj_weight,
                in_proj_bias,
                out_proj_weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x is None or x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x is not None and x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    in_proj_weight,
                    in_proj_bias,
                    out_proj_weight,
                    self.out_proj.bias,
                    key_padding_mask if key_padding_mask is not None else attn_mask,
                    need_weights,
                    average_attn_weights,
                    1 if key_padding_mask is not None else 0 if attn_mask is not None else None)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                in_proj_weight, in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, out_proj_weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                in_proj_weight, in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, out_proj_weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def extra_repr(self) -> str:
        return super().extra_repr() + \
            (f', bias_correct=True' if self.bias_correct else '')
    
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, 
        strict, missing_keys, unexpected_keys, error_msgs
    ):
        device = self.out_proj.weight.device
        if prefix + 'out_proj_scale' in state_dict:
            self.pack(calibrated=False)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs)
        
        if prefix + 'out_proj_scale' in state_dict:
            self.q_proj_weight.data = tunpack(self.q_proj_weight, self.q_proj_des)
            self.k_proj_weight.data = tunpack(self.k_proj_weight, self.k_proj_des)
            self.v_proj_weight.data = tunpack(self.v_proj_weight, self.v_proj_des)
            
            def reload(*args, **kwargs):
                nn.Linear._load_from_state_dict(self.out_proj, *args, **kwargs)
                self.out_proj.weight.data = tunpack(self.out_proj.weight, self.out_proj_des)
            self.out_proj._load_from_state_dict = reload

        self.to(device)
