"""
RTN (Round-to-Nearest) 量化器实现
最基础的量化方法，直接对权重进行舍入
"""
import torch
from typing import Dict, Any
from ..base import BaseQuantizer


class RTNQuantizer(BaseQuantizer):
    """RTN 量化器 - Round-to-Nearest"""
    
    def calibrate(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算量化参数"""
        # 确定量化范围
        if self.symmetric:
            # 对称量化: [-max, max]
            self.qmax = 2 ** (self.n_bits - 1) - 1
            self.qmin = -self.qmax
            if self.per_channel:
                # 逐通道
                if tensor.dim() == 2:
                    scale = tensor.abs().max(dim=1, keepdim=True)[0]
                elif tensor.dim() == 4:  # conv weight
                    scale = tensor.abs().flatten(start_dim=1).max(dim=1, keepdim=True)[0]
                else:
                    scale = tensor.abs().max()
            else:
                scale = tensor.abs().max()
            scale = scale.clamp(min=1e-5) / self.qmax
            zero_point = torch.tensor(0, dtype=torch.int32, device=tensor.device)
        else:
            # 非对称量化: [0, 2^n-1]
            self.qmax = 2 ** self.n_bits - 1
            self.qmin = 0
            if self.per_channel:
                if tensor.dim() == 2:
                    t_min = tensor.min(dim=1, keepdim=True)[0]
                    t_max = tensor.max(dim=1, keepdim=True)[0]
                elif tensor.dim() == 4:
                    t_min = tensor.flatten(start_dim=1).min(dim=1, keepdim=True)[0]
                    t_max = tensor.flatten(start_dim=1).max(dim=1, keepdim=True)[0]
                else:
                    t_min = tensor.min()
                    t_max = tensor.max()
            else:
                t_min = tensor.min()
                t_max = tensor.max()
            
            scale = (t_max - t_min).clamp(min=1e-5) / self.qmax
            zero_point = torch.round(-t_min / scale).clamp(self.qmin, self.qmax)
        
        self.scale = scale
        self.zero_point = zero_point
        
        return {'scale': self.scale, 'zero_point': self.zero_point}
    @torch.no_grad()
    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """量化到整数"""
        if self.scale is None:
            raise RuntimeError("Must calibrate before quantize")
        
        # 归一化
        q_tensor = tensor / self.scale
        
        # 添加 zero_point (非对称)
        if not self.symmetric:
            q_tensor = q_tensor + self.zero_point
        
        # 四舍五入并截断
        q_tensor = torch.round(q_tensor).clamp(self.qmin, self.qmax)
        
        # 转换为整数类型
        if self.n_bits == 8:
            dtype = torch.int8 if self.symmetric else torch.uint8
        elif self.n_bits == 4:
            dtype = torch.int8  # 4bit使用int8存储
        else:
            dtype = torch.int32
            
        return q_tensor.to(dtype)
    @torch.no_grad()
    # @torch.jit.script
    def dequantize(self, q_tensor: torch.Tensor) -> torch.Tensor:
        """反量化到浮点"""
        if self.scale is None:
            raise RuntimeError("Must calibrate before dequantize")
        
        result = q_tensor.to(torch.bfloat16)
        
        # 移除 zero_point (非对称)
        if not self.symmetric:
            result = result - self.zero_point.to(torch.bfloat16)
        
        # 缩放回原范围
        result = result * self.scale.to(torch.bfloat16)
        
        return result
    
    def fake_quantize(self, tensor: torch.Tensor, calibrate: bool = False) -> torch.Tensor:
        """
        伪量化 - 用于实验验证
        前向模拟量化效果，保持梯度
        """
        if calibrate or self.scale is None:
            self.calibrate(tensor)
        
        # 量化
        q_tensor = self.quantize(tensor)
        
        # 反量化
        dq_tensor = self.dequantize(q_tensor)
        
        return dq_tensor


class PerChannelRTNQuantizer(RTNQuantizer):
    """逐通道RTN量化器（默认启用）"""
    
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_dim: int = 0):
        super().__init__(n_bits=n_bits, symmetric=symmetric, per_channel=True)
        self.channel_dim = channel_dim

class PerTensorRTNQuantizer(RTNQuantizer):
    """逐张量RTN量化器"""
    
    def __init__(self, n_bits: int = 8, symmetric: bool = False):
        super().__init__(n_bits=n_bits, symmetric=symmetric, per_channel=False)
