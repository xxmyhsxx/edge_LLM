
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
    # 如果相对导入失败，使用绝对导入
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
                dtype=torch.uint8,
                requires_grad=False,
            ),
        )
        self.register_buffer("scale", None)
        self.register_buffer("zero_point", None)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
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
    @torch.jit.script
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
        new_module.scale = quantizer.scale
        new_module.zero_point = quantizer.zero_point
        # print(quantizer.zero_point)

        if module.bias is not None:
            new_module.bias.data = module.bias.to(torch.bfloat16)
        
        # 保存反量化函数引用
        # new_module._dequantize_func = quantizer.dequantize
        del module  # 释放原模块内存
        
        return new_module
    
    

    def __repr__(self):
        return f"W8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None})"



def quantize_layers(layers, weight_quant="channel", w_bit=8, zp=False, quantizer_config=None):
    """
    对Qwen3模型的layers进行量化替换
    
    Args:
        layers: 模型的层列表
        weight_quant: 量化模式 ("channel" 或 "tensor")
        w_bit: 量化位宽
        zp: 是否使用zero_point
        quantizer_config: quantizer配置字典（可选）
    
    Returns:
        量化后的layers
    """
    # 根据配置创建quantizer类
    if weight_quant == "channel":
        quantizer_class = PerChannelRTNQuantizer
        quantizer_kwargs = {'channel_dim': 0}
    elif weight_quant == "tensor":
        quantizer_class = PerTensorRTNQuantizer
        quantizer_kwargs = {}
    else:
        raise ValueError(f"Invalid weight_quant: {weight_quant}")
    
    # 合并默认配置
    if quantizer_config is None:
        quantizer_config = {}
    
    for i in tqdm(range(len(layers)), desc="Quantizing Layers"):
        m = layers[i]
        m.to("cuda")
        
        if isinstance(m, Qwen3DecoderLayer):
            # 创建共享的quantizer实例（减少对象创建开销）
            quantizer_kwargs_full = {
                'n_bits': w_bit,
                'symmetric': not zp,
                **quantizer_kwargs,
                **quantizer_config
            }
            
            # 量化MLP层
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(m.mlp, proj_name)
                quantizer = quantizer_class(**quantizer_kwargs_full)
                proj = W8Linear.from_float(
                    proj, 
                    quantizer=quantizer
                )
                
                setattr(m.mlp, proj_name, proj)
            
            # 量化Attention层
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj = getattr(m.self_attn, proj_name)
                quantizer = quantizer_class(**quantizer_kwargs_full)
                proj = W8Linear.from_float(
                    proj, 
                    quantizer=quantizer
                )
                setattr(m.self_attn, proj_name, proj)
            
            # 内存优化：每完成一个layer就清理
            del quantizer
            gc.collect()
            torch.cuda.empty_cache()

        else:
            raise ValueError(f"目前不支持该模型类型: {type(m)}")
    
    return layers





def save_quantized_model(model, save_path, tokenizer=None, quantization_config=None, original_model_name=None):
    """
    以 HuggingFace 格式保存量化后的模型
    
    Args:
        model: 量化后的模型
        save_path: 保存路径
        tokenizer: 可选的tokenizer
        quantization_config: 量化配置字典
        original_model_name: 原始模型名称（可选）
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 1. 保存量化后的模型权重（兼容格式）
    model.save_pretrained(save_path)
    
    # 2. 如果模型没有 save_pretrained 方法，使用标准方式保存
    if not hasattr(model, 'save_pretrained') or not callable(getattr(model, 'save_pretrained', None)):
        # 保存为标准 PyTorch 格式
        torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        
        # 创建 HuggingFace 风格的配置文件
        config = {
            "model_type": "internvl",
            "torch_dtype": str(next(model.parameters()).dtype),
            "quantization_config": quantization_config or {
                "method": "RTN",
                "bits": 8,
                "group_size": -1,
                "symmetric": False,
            },
            "quantization_method": "RTN",
            "quantization_bits": 8,
            "original_model": original_model_name or "internvl",
        }
        
        # 保存配置
        import json
        with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 3. 保存 tokenizer（如果提供）
    if tokenizer is not None:
        try:
            tokenizer.save_pretrained(save_path)
            print(f"✓ Tokenizer 已保存为 HuggingFace 格式")
        except Exception as e:
            print(f"⚠ Tokenizer 保存失败: {e}")
    
    # 4. 保存量化信息摘要（额外信息）
    if quantization_config:
        quant_summary = {
            "quantization_method": "RTN",
            "bits": 8,
            "config": quantization_config,
            "model_stats": {
                "total_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_size_mb": sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / (1024**2),
            }
        }
        
        with open(os.path.join(save_path, "quantization_config.json"), "w", encoding="utf-8") as f:
            json.dump(quant_summary, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 量化模型已保存为 HuggingFace 格式: {save_path}")
    print(f"  可以使用 from_pretrained() 加载")


def load_quantized_model(save_path, device="auto"):
    """
    从 HuggingFace 格式加载量化模型
    
    Args:
        save_path: 保存路径
        device: 设备设置
    
    Returns:
        state_dict: 状态字典
        config: 配置信息
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"模型路径不存在: {save_path}")
    
    # 自动选择设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 尝试加载配置
    config = None
    config_file = os.path.join(save_path, "config.json")
    if os.path.exists(config_file):
        import json
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    
    # 尝试加载模型权重（支持多种格式）
    state_dict = None
    possible_files = [
        "pytorch_model.bin",
        "model.pt",
        "quantized_model.pt",
        "pytorch_model_safe.bin"
    ]
    
    for filename in possible_files:
        model_file = os.path.join(save_path, filename)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location=device)
            break
    
    if state_dict is None:
        raise FileNotFoundError(f"在 {save_path} 中未找到模型文件")
    
    print(f"✓ 从 HuggingFace 格式加载成功: {save_path}")
    print(f"  设备: {device}")
    print(f"  参数数量: {len(state_dict)}")
    
    return state_dict, config


# 增强的类方法（让 W8Linear 也能像 HuggingFace 模型一样保存/加载）
def add_huggingface_methods_to_w8linear():
    """
    为 W8Linear 类添加 HuggingFace 风格的 save_pretrained 和 from_pretrained 方法
    """
    def save_pretrained(self, save_directory):
        """保存 W8Linear 层（HuggingFace 风格）"""
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存权重
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # 保存配置
        config = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "has_bias": self.bias is not None,
            "class_type": "W8Linear",
            "quantization_config": {
                "weight_shape": list(self.weight.shape),
                "scale": self.scale.item() if self.scale is not None else None,
                "zero_point": self.zero_point.item() if self.zero_point is not None else None,
            }
        }
        
        import json
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ W8Linear 保存为 HuggingFace 格式: {save_directory}")
    
    def from_pretrained(cls, model_id_or_path, device="auto"):
        """从 HuggingFace 格式加载 W8Linear 层"""
        import json
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 检查路径
        if not os.path.exists(model_id_or_path):
            raise FileNotFoundError(f"路径不存在: {model_id_or_path}")
        
        # 加载配置
        config_file = os.path.join(model_id_or_path, "config.json")
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 创建实例
        instance = cls(
            in_features=config["in_features"],
            out_features=config["out_features"],
            quantizer=None,
            bias=config["has_bias"]
        )
        
        # 加载权重
        model_file = os.path.join(model_id_or_path, "pytorch_model.bin")
        state_dict = torch.load(model_file, map_location=device)
        instance.load_state_dict(state_dict)
        instance = instance.to(device)
        
        print(f"✓ W8Linear 从 HuggingFace 格式加载: {model_id_or_path}")
        return instance
    
    # 动态绑定方法
    W8Linear.save_pretrained = save_pretrained
    W8Linear.from_pretrained = classmethod(from_pretrained)
    W8Linear.push_to_hub = lambda self, repo_id, *args, **kwargs: print("push_to_hub 可以通过 save_pretrained + git lfs 实现")




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
    
    # 7. 保存量化模型
    print(f"\n5) 保存量化模型...")
    save_path = "./quantized_internvl_model"
    try:
        save_quantized_model(model, save_path, tokenizer)
        print(f"   ✓ 模型已保存到: {save_path}")
        
        # 显示保存的文件
        saved_files = os.listdir(save_path)
        print(f"   保存的文件: {saved_files}")
        
    except Exception as e:
        print(f"   ✗ 保存失败: {e}")
        exit(1)
    
    # 8. 测试加载量化模型
    print(f"\n6) 测试加载量化模型...")
    try:
        # 加载状态字典和配置
        loaded_state_dict, loaded_config = load_quantized_model(save_path)
        
        print(f"   ✓ 加载成功!")
        print(f"   配置信息: {loaded_config['model_type']}")
        print(f"   量化方法: {loaded_config['quantization_method']}")
        
        # 测试加载后的模型
        print(f"\n7) 测试加载后模型的推理...")
        
        # 创建测试用的量化模型（使用原始模型结构）
        test_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().cuda()
        
        # 应用量化后的权重
        test_model.load_state_dict(loaded_state_dict, strict=False)
        
        # 测试推理
        with torch.no_grad():
            response = test_model.chat(
                tokenizer,
                None,
                test_text,
                generation_config = {"max_new_tokens": 50, "do_sample": False, "pad_token_id": tokenizer.eos_token_id}
            )
            
            print(f"   加载后模型输出: {response}")
            print(f"   ✓ 加载后模型推理成功！")
        
    except Exception as e:
        print(f"   ✗ 加载或推理失败: {e}")
        exit(1)
    
    # 9. 总结
    print(f"\n" + "=" * 80)
    print("量化完成！")
    print("=" * 80)
    print(f"总结:")
    print(f"  - 原始模型: {total_params:,} 参数 ({param_size_mb:.2f} MB)")
    print(f"  - 量化配置: {w_bit}-bit {weight_quant} 量化")
    print(f"  - 处理层数: {len(decoder_layers)} 个 Qwen3DecoderLayer")
    if 'memory_saved' in locals():
        print(f"  - 内存节省: {memory_saved:.2f} MB")
    print(f"\n模型保存路径: {save_path}")
    print(f"\n使用方法:")
    print(f"  # 保存量化模型")
    print(f"  from src.methods.quantization.core.rtn.quantize_model import save_quantized_model")
    print(f"  save_quantized_model(quantized_model, './quantized_internvl', tokenizer)")
    print(f"  ")
    print(f"  # 加载量化模型")
    print(f"  from src.methods.quantization.core.rtn.quantize_model import load_quantized_model, create_quantized_model")
    print(f"  state_dict, config = load_quantized_model('./quantized_internvl')")
    print(f"  quantized_model = create_quantized_model(original_model, state_dict)")
    print(f"  ")
    print(f"  # 直接推理")
    print(f"  response = quantized_model.chat(tokenizer, None, '你的问题', generation_config={...})")
