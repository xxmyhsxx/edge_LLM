# 保存量化后的模型

import json
import os

def save_quantized_model(model, tokenizer, save_dir, quant_config):
    """
    保存量化后的模型
    Args:
        model: 已经完成量化替换的模型
        tokenizer: 分词器
        save_dir: 保存路径
        quant_config: 量化配置字典 (例如 {'w_bit': 8, 'weight_quant': 'channel'})
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Saving model to {save_dir}...")
    # 1. 保存模型权重和配置文件
    #由于你的W8Linear使用了register_buffer，save_pretrained会自动保存这些int8权重和scale
    model.save_pretrained(save_dir)
    
    # 2. 保存 tokenizer
    tokenizer.save_pretrained(save_dir)

    # 3. 保存量化特定的配置，以便加载时知道如何重构模型结构
    with open(os.path.join(save_dir, "quant_config.json"), "w") as f:
        json.dump(quant_config, f)
        
    print("Model saved successfully.")

# 使用示例：

# quant_config = {"w_bit": 8, "weight_quant": "channel", "zp": False}
# save_quantized_model(model, tokenizer, "./qwen_quantized_w8", quant_config)

if __name__ == "__main__":
    # 这里可以添加一个简单的测试，确保保存功能正常
    import os
    from safetensors.torch import load_file
    import torch

    # 你的模型路径
    model_path = "/app/edge_LLM/models/quantized_internvl"

    # 找到其中一个 safetensors 文件
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            file_path = os.path.join(model_path, file)
            print(f"Inspecting: {file_path}")
            
            # 加载并打印 keys
            state_dict = load_file(file_path)
            keys = list(state_dict.keys())
            
            # 检查是否有 scale
            has_scale = any("scale" in k for k in keys)
            has_weight = any("weight" in k for k in keys)
            
            print(f"Total keys: {len(keys)}")
            print(f"Has 'scale' keys? -> {has_scale}")
            
            # 打印前几个 key 看看长什么样
            print("Sample keys:", keys[:5])
            
            break