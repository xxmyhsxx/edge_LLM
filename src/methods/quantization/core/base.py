"""
量化器基类和基础接口
定义所有量化方法需要实现的核心接口
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional


class BaseQuantizer(ABC):
    """量化器基类"""
    
    def __init__(self, n_bits: int = 8, symmetric: bool = False, per_channel: bool = False):
        """
        Args:
            n_bits: 量化位宽 (4, 8, 16)
            symmetric: 是否使用对称量化
            per_channel: 是否逐通道量化
        """
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        
        # 量化参数
        self.scale = None
        self.zero_point = None
        self.qmin = None
        self.qmax = None
        
    @abstractmethod
    def calibrate(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        校准：计算量化参数 (scale, zero_point)
        
        Args:
            tensor: 待量化的张量
            
        Returns:
            包含 scale 和 zero_point 的字典
        """
        pass
    
    @abstractmethod
    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        量化：将浮点数转换为整数
        
        Args:
            tensor: 浮点张量
            
        Returns:
            量化后的整数张量
        """
        pass
    
    @abstractmethod
    def dequantize(self, q_tensor: torch.Tensor) -> torch.Tensor:
        """
        反量化：将整数转换为浮点数
        
        Args:
            q_tensor: 量化后的整数张量
            
        Returns:
            反量化后的浮点张量
        """
        pass
    
    def forward(self, tensor: torch.Tensor, calibrate: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            tensor: 输入张量
            calibrate: 是否进行校准
            
        Returns:
            量化/反量化后的张量（用于伪量化）
        """
        if calibrate or self.scale is None:
            self.calibrate(tensor)
        
        q_tensor = self.quantize(tensor)
        return self.dequantize(q_tensor)
    
    def get_state(self) -> Dict[str, Any]:
        """获取量化器状态（用于序列化）"""
        return {
            'n_bits': self.n_bits,
            'symmetric': self.symmetric,
            'per_channel': self.per_channel,
            'scale': self.scale,
            'zero_point': self.zero_point,
            'qmin': self.qmin,
            'qmax': self.qmax,
        }
    
    def load_state(self, state: Dict[str, Any]):
        """加载量化器状态"""
        self.n_bits = state['n_bits']
        self.symmetric = state['symmetric']
        self.per_channel = state['per_channel']
        self.scale = state['scale']
        self.zero_point = state['zero_point']
        self.qmin = state['qmin']
        self.qmax = state['qmax']


class QuantizationConfig:
    """量化配置类"""
    
    def __init__(self, 
                 method: str = "rtn",
                 n_bits: int = 8,
                 symmetric: bool = False,
                 per_channel: bool = True,
                 group_size: int = -1,
                 **kwargs):
        """
        Args:
            method: 量化方法 ("rtn", "awq", "smoothquant")
            n_bits: 量化位宽
            symmetric: 是否对称
            per_channel: 是否逐通道
            group_size: 分组大小 (-1 表示不分组)
        """
        self.method = method
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.group_size = group_size
        self.kwargs = kwargs
    
    def __repr__(self):
        return (f"QuantConfig(method={self.method}, n_bits={self.n_bits}, "
                f"symmetric={self.symmetric}, per_channel={self.per_channel}, "
                f"group_size={self.group_size})")


class QuantizedLinear(nn.Module):
    """量化线性层基类"""
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 bias: bool = True,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_config = quant_config or QuantizationConfig()
        
        # 注册缓冲区
        self.register_buffer('weight_q', None)
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_zero_point', None)
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def quantize_weight(self, weight: torch.Tensor):
        """量化权重"""
        raise NotImplementedError
    
    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, {self.quant_config}")
