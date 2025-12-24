"""
RTN (Round-to-Nearest) 量化模块
提供最基础的量化方法，直接对权重进行舍入
"""
from .quantizer import RTNQuantizer, PerChannelRTNQuantizer, PerTensorRTNQuantizer
from .linear import RTNQuantizedLinear, W4Linear, W6Linear, W8Linear

__all__ = [
    'RTNQuantizer',
    'PerChannelRTNQuantizer', 
    'PerTensorRTNQuantizer',
    'RTNQuantizedLinear',
    'W4Linear',
    'W6Linear',
    'W8Linear'
]
