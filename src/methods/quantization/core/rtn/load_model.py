import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer,AutoModel
import os
import json
from accelerate import init_empty_weights, load_checkpoint_and_dispatch # 可选，用于加速大模型结构初始化

try:
    from .quantize_model import W8Linear,PerChannelRTNQuantizer,quantize_layers
except ImportError:
    from src.methods.quantization.core.rtn.quantize_model import W8Linear,PerChannelRTNQuantizer,quantize_layers

def replace_linear_with_w8_for_loading(module, w_bit=8):
    """
    递归地将模型中的 Linear 层替换为 W8Linear (用于加载阶段)
    注意：这里不需要 Quantizer，因为我们只是搭建“空壳”结构用来接收权重
    """
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            # 获取原始 Linear 的特征
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None
            
            # 创建空的 W8Linear
            new_layer = W8Linear(
                in_features=in_features, 
                out_features=out_features, 
                quantizer=None, # 加载时不需要量化器
                bias=bias,
            )
            # 替换
            setattr(module, name, new_layer)
        else:
            # 递归处理子模块
            replace_linear_with_w8_for_loading(child, w_bit)

def load_quantized_model(model_path, device="cuda"):
    """
    加载自定义量化的模型
    """
    print(f"Loading quantized model from {model_path}...")
    # 1. 加载 Config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 2. 读取我们自己存的量化配置
    quant_config_path = os.path.join(model_path, "quant_config.json")
    if os.path.exists(quant_config_path):
        with open(quant_config_path, 'r') as f:
            quant_config = json.load(f)
    else:
        print("Warning: quant_config.json not found, assuming defaults.")
        quant_config = {"w_bit": 8} # 默认值
    
    print("1. Initializing empty model structure...")

    from transformers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME
    from transformers.modeling_utils import load_state_dict
    
    # 处理 safetensors 或 bin
    is_safetensors = False
    if os.path.exists(os.path.join(model_path, "model.safetensors.index.json")) or os.path.exists(os.path.join(model_path, "model.safetensors")):
        is_safetensors = True
        
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    replace_linear_with_w8_for_loading(model.language_model.model.layers, w_bit=quant_config.get('w_bit', 8))
    print("   ✓ Linear layers replaced with W8Linear.")
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    from transformers.modeling_utils import load_sharded_checkpoint
    
    if is_safetensors:
        from safetensors.torch import load_file
        # 如果是单文件
        if os.path.exists(os.path.join(model_path, "model.safetensors")):
            state_dict = load_file(os.path.join(model_path, "model.safetensors"))
            model.load_state_dict(state_dict, strict=True) # strict=True 验证是否完美匹配
        else:

            load_checkpoint_and_dispatch(model, model_path, device_map={"": "cuda:0"},no_split_module_classes=["Qwen3DecoderLayer"],)
    else:
        # PyTorch bin 文件
        if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
        else:
            load_checkpoint_and_dispatch(model, model_path, device_map="cuda",)

    # if hasattr(model, "hf_device_map"):
    #     print("模型分布情况 (hf_device_map):", model.hf_device_map)


    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_quantized_model("/app/edge_LLM/models/quantized_internvl", device="cuda")
    print("Model and tokenizer loaded successfully.")
    print(torch.cuda.memory_allocated() / 1024**2)
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
