
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import gc
import sys
import json
import os
from typing import Dict, Any, Optional
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

# 确保可以正确导入同目录下的模块
try:
    from .quantizer import PerChannelRTNQuantizer, PerTensorRTNQuantizer
except ImportError:
    # 如果相对导入失败，使用绝对导入s
    from src.methods.quantization.core.rtn.quantizer import PerChannelRTNQuantizer, PerTensorRTNQuantizer


class W8Linear(nn.Module):
    """8-bit Quantized Linear Layer with PerChannelRTNQuantizer"""
    
    def __init__(self, in_features, out_features, quantizer=None, bias=True):
        super(W8Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 注册量化参数为buffer，不占用模型参数内存
        self.register_buffer(
            "weight",
            torch.zeros(
                self.out_features,
                self.in_features,
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer("scale", torch.zeros((self.out_features,1), dtype=torch.bfloat16, requires_grad=False),)
        self.register_buffer("zero_point", torch.tensor(0, dtype=torch.int32,requires_grad=False,),)
        # self.scale = torch.zeros((out_features,1), dtype=torch.bfloat16)
        # self.zero_point = torch.tensor(0, dtype=torch.int32)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter('bias', None)
        

    def to(self, *args, **kwargs):
        super(W8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.scale is not None:
            self.scale = self.scale.to(*args, **kwargs)
        if self.zero_point is not None:
            self.zero_point = self.zero_point.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self
    
    @staticmethod
    # @torch.jit.script
    def fused_quant_linear_forward(x, weight, scale, zero_point):
        """融合量化线性前向计算"""
        
        weight_dequantized = (weight - zero_point) * scale
        result = torch.matmul(x, weight_dequantized.transpose(0, 1))
        
        return result

    @torch.no_grad()
    # @torch.compile()
    def forward(self, input):
        # # 使用quantizer的反量化方法
        # weight_deq = self.dequantize(self.weight,self.scale,self.zero_point)
        
        result = self.fused_quant_linear_forward(input, self.weight.to(torch.bfloat16), self.scale.to(torch.bfloat16), self.zero_point.to(torch.bfloat16))
        if self.bias is not None:
            result = result + self.bias
        return result
    
    @classmethod
    @torch.no_grad()
    def from_float(cls, module, quantizer=None):
        """
        从浮点模型转换 - 使用Quantizer进行量化
        
        Args:
            module: 原始的torch.nn.Linear模块
            quantizer: 外部传入的Quantizer实例（必需）
        """
        assert isinstance(module, torch.nn.Linear)
        assert quantizer is not None, "quantizer parameter is required"
        
        # 创建新模块，传入quantizer
        new_module = cls(
            module.in_features,
            module.out_features,
            quantizer,
            bias=module.bias is not None,
        )
        
        # 使用quantizer进行校准和量化
        quantizer.calibrate(module.weight)
        new_module.weight.data = quantizer.quantize(module.weight)
        
        # 保存量化结果
        
        # new_module.weight.data = q_weight
        new_module.scale.data = quantizer.scale
        if quantizer.zero_point is not None:
            new_module.zero_point.data = quantizer.zero_point
            # print( quantizer.zero_point)
        else:
            pass
        # print(quantizer.zero_point)

        if module.bias is not None:
            new_module.bias.data = module.bias.to(torch.bfloat16)
        
        del module  # 释放原模块内存
        
        return new_module
    
    

    def __repr__(self):
        return f"W8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None})"


# 对单个Qwen3模型的layer进行量化替换
def quantize_layer(layer, weight_quant="channel", w_bit=8, zp=False, quantizer_config=None):
    """
    对单个Qwen3模型的layer进行量化替换
    
    Args:
        layer: 模型的单个层
        weight_quant: 量化模式 ("channel" 或 "tensor")
        w_bit: 量化位宽
        zp: 是否使用zero_point
        quantizer_config: quantizer配置字典（可选）
    """
    layer.to("cuda")

    if isinstance(layer, Qwen3DecoderLayer):
        # 根据配置创建quantizer类
        if weight_quant == "channel":
            quantizer_class = PerChannelRTNQuantizer
            quantizer_kwargs = {'channel_dim': 0}
        elif weight_quant == "tensor":
            quantizer_class = PerTensorRTNQuantizer
            quantizer_kwargs = {}
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")

        # 创建共享的quantizer实例（减少对象创建开销）
        quantizer_kwargs_full = {
            'n_bits': w_bit,
            'symmetric': not zp,
            **quantizer_kwargs,
            **(quantizer_config or {})
        }

        # 量化MLP层
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(layer.mlp, proj_name)
            quantizer = quantizer_class(**quantizer_kwargs_full)
            proj = W8Linear.from_float(
                    proj, 
                    quantizer=quantizer
                )
                
            setattr(layer.mlp, proj_name, proj)
        
        # 量化Attention层
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            proj = getattr(layer.self_attn, proj_name)
            quantizer = quantizer_class(**quantizer_kwargs_full)
            proj = W8Linear.from_float(
                proj, 
                quantizer=quantizer
            )
            setattr(layer.self_attn, proj_name, proj)
        
        # 内存优化：每完成一个layer就清理
        del quantizer
        gc.collect()
        torch.cuda.empty_cache()
    
    else:
        raise ValueError(f"目前不支持该模型类型: {type(layer)}")
    

def quantize_layers(layers, weight_quant="channel", w_bit=8, zp=False, quantizer_config=None):
    """
    对Qwen3模型的layers进行量化替换
    
    Args:
        layers: 模型的层列表
        weight_quant: 量化模式 ("channel" 或 "tensor")
        w_bit: 量化位宽
        zp: 是否使用zero_point
        quantizer_config: quantizer配置字典（可选）
    """
    for layer in tqdm(layers, desc="Quantizing Layers"):
        quantize_layer(
            layer,
            weight_quant=weight_quant,
            w_bit=w_bit,
            zp=zp,
            quantizer_config=quantizer_config
        )





if __name__ == "__main__":
    """
    InternVL3_5-4B-Instruct 模型量化示例
    使用 quantize_layers 函数
    """
    from transformers import AutoModel, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    import gc
    
    print("=" * 80)
    print("InternVL3_5-4B-Instruct 模型量化示例")
    print("=" * 80)
    
    # 1. 配置
    model_path = "/app/models/InternVL3_5-4B-Instruct"
    weight_quant = "channel"  # "channel" 或 "tensor"
    w_bit = 8
    zp = False  # False 表示非对称量化
    
    print(f"\n量化配置:")
    print(f"  模型路径: {model_path}")
    print(f"  量化模式: {weight_quant}")
    print(f"  量化位宽: {w_bit}-bit")
    print(f"  对称量化: {zp}")
    
    # 2. 加载原始模型
    print(f"\n1) 加载原始模型...")
    try:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().cuda()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        # 统计模型信息
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / (1024**2)
        
        print(f"   ✓ 模型加载完成")
        print(f"   总参数量: {total_params:,}")
        print(f"   参数内存: {param_size_mb:.2f} MB")
        
    except Exception as e:
        print(f"   ✗ 模型加载失败: {e}")
        exit(1)
    
    # 3. 查找 Qwen3DecoderLayer
    print(f"\n2) 查找量化层...")
    decoder_layers = []
    for name, module in model.named_modules():
        if isinstance(module, Qwen3DecoderLayer):
            decoder_layers.append(module)
    
    print(f"   找到 {len(decoder_layers)} 个 Qwen3DecoderLayer")
    
    # 4. 记录内存并开始量化
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / 1024**2
        print(f"   量化前GPU内存: {memory_before:.2f} MB")
    
    print(f"\n3) 使用 quantize_layers 进行量化...")
    
    try:
        # 使用 quantize_layers 函数批量量化
        quantize_layers(
            decoder_layers,
            weight_quant=weight_quant,
            w_bit=w_bit,
            zp=zp
        )
        print(f"   ✓ 量化完成！")
        
    except Exception as e:
        print(f"   ✗ 量化失败: {e}")
        exit(1)
    
    # 5. 统计结果
    if torch.cuda.is_available():
        memory_after = torch.cuda.memory_allocated() / 1024**2
        memory_saved = memory_before - memory_after
        print(f"   量化后GPU内存: {memory_after:.2f} MB")
        print(f"   内存节省: {memory_saved:.2f} MB ({memory_saved/memory_before*100:.1f}%)")
    
    # 6. 测试推理
    print(f"\n4) 测试推理...")
    test_text = "你好，请介绍一下InternVL模型的特点"
    
    try:
        inputs = tokenizer(test_text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            print(f"   输入: {test_text}")
            print(f"   生成中...")

            response = model.chat(
                tokenizer,
                None,
                test_text,
                generation_config = {"max_new_tokens": 100, "do_sample": False, "pad_token_id": tokenizer.eos_token_id}
                
            )

            print(f"   输出: {response}")
            print(f"   ✓ 推理成功！")
            
    except Exception as e:
        print(f"   ✗ 推理失败: {e}")
    

    print(f"\n" + "=" * 80)
    print("量化完成！")
    print("=" * 80)
    print(f"总结:")
    print(f"  - 原始模型: {total_params:,} 参数 ({param_size_mb:.2f} MB)")
    print(f"  - 量化配置: {w_bit}-bit {weight_quant} 量化")
    print(f"  - 处理层数: {len(decoder_layers)} 个 Qwen3DecoderLayer")
    if 'memory_saved' in locals():
        print(f"  - 内存节省: {memory_saved:.2f} MB")

