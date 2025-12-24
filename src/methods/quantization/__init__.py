"""
量化模块统一接口
支持RTN、AWQ、SmoothQuant以及伪量化和基准测试
"""
from .core.base import QuantizationConfig, QuantizedLinear, BaseQuantizer
from .core.rtn import RTNQuantizer, PerChannelRTNQuantizer, PerTensorRTNQuantizer, W4Linear, W6Linear, W8Linear
from .core.awq import AWQQuantizer, AWQLinear, MixedBitsAWQQuantizer
from .core.smoothquant import SmoothQuantQuantizer, SmoothLinear, PerChannelSmoothQuant, PerTensorSmoothQuant

from .models.quant_model import (
    QuantModelWrapper, 
    ActivationCollector, 
    quantize_model
)

from .fake.fake_quantizer import (
    FakeQuantizer, 
    PerChannelFakeQuantizer, 
    QuantizationAwareTraining,
    FakeQuantTensor
)

from .benchmark.quant_benchmark import (
    QuantizationBenchmark,
    FakeQuantizationBenchmark,
    BenchmarkResult,
    compare_to_baseline
)

from .runtime.torch_backend import (
    TorchQuantBackend,
    QuantizedLinearTorch,
    fuse_quantized_layers
)

# 模块导出
__all__ = [
    # 配置和基类
    'QuantizationConfig',
    'QuantizedLinear',
    'BaseQuantizer',
    
    # 核心量化方法
    'RTNQuantizer',
    'PerChannelRTNQuantizer',
    'PerTensorRTNQuantizer',
    'W4Linear',
    'W6Linear', 
    'W8Linear',
    'AWQQuantizer',
    'AWQLinear',
    'MixedBitsAWQQuantizer',
    'SmoothQuantQuantizer',
    'SmoothLinear',
    'PerChannelSmoothQuant',
    'PerTensorSmoothQuant',
    
    # 模型包装
    'QuantModelWrapper',
    'ActivationCollector',
    'quantize_model',
    
    # 伪量化
    'FakeQuantizer',
    'PerChannelFakeQuantizer',
    'QuantizationAwareTraining',
    'FakeQuantTensor',
    
    # 基准测试
    'QuantizationBenchmark',
    'FakeQuantizationBenchmark',
    'BenchmarkResult',
    'compare_to_baseline',
    
    # 运行时
    'TorchQuantBackend',
    'QuantizedLinearTorch',
    'fuse_quantized_layers',
]


def quick_quantize(model, n_bits=8, method='rtn', symmetric=False, calibration_data=None):
    """
    快速量化函数
    
    Args:
        model: 待量化模型
        n_bits: 量化位宽 (4, 8)
        method: 方法 ('rtn', 'awq', 'smoothquant')
        symmetric: 是否对称量化
        calibration_data: 校准数据
    
    Returns:
        量化后的模型
    """
    config = QuantizationConfig(
        method=method,
        n_bits=n_bits,
        symmetric=symmetric,
        per_channel=True
    )
    
    activation_collector = None
    if method in ['awq', 'smoothquant']:
        activation_collector = ActivationCollector(max_samples=32)
    
    return quantize_model(model, config, calibration_data, activation_collector)


def benchmark_methods(model, methods, calibration_data=None, test_data=None):
    """
    批量基准测试
    
    Args:
        model: 模型
        methods: 方法配置列表
        calibration_data: 校准数据
        test_data: 测试数据
    
    Returns:
        基准测试结果列表
    """
    bench = QuantizationBenchmark(model, calibration_data, test_data)
    return bench.run_full_benchmark(methods)
