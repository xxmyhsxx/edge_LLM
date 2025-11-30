import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import os
import sys
import csv

# 路径 hack
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
if project_root not in sys.path:
    sys.path.append(project_root)

# 引用 Collector 和 Quant Utils
from src.methods.research.collector import InternVLCollector
from src.methods.quantization.quant_tensor import quantize_weight_per_channel_absmax

class SensitivityProbe:
    def __init__(self, model_path):
        # 借用 Collector 来加载模型和处理输入，但不跑 collect
        self.collector = InternVLCollector(model_path)
        self.model = self.collector.model
        self.device = self.model.device
        
    def _get_target_modules(self, layer):
        """获取一层中所有 Linear 模块 (Q,K,V,O, MLP)"""
        modules = []
        # Attention
        if hasattr(layer, 'self_attn'):
            modules.extend([
                layer.self_attn.q_proj, 
                layer.self_attn.k_proj, 
                layer.self_attn.v_proj, 
                layer.self_attn.o_proj
            ])
        # MLP
        if hasattr(layer, 'mlp'):
            modules.extend([
                layer.mlp.gate_proj, 
                layer.mlp.up_proj, 
                layer.mlp.down_proj
            ])
        return modules

    def run_benchmark(self, media_path, prompt, n_bits=4, output_dir="experiments/results/sensitivity"):
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 准备输入 (复用 Collector 的逻辑)
        # 这里我们需要手动触发一次 Collector 的预处理来拿 input tensor
        print(">>> [Sensitivity] Preparing inputs...")
        
        # 借用 collector 内部方法构造 input
        # 注意：这里我们稍微 hack 一下，模拟 collector.collect_activations 的前半部分
        from src.utils.media_loader import MediaLoader
        media_type = 'video' if 'mp4' in media_path else 'image'
        media_list = MediaLoader.load(media_path, media_type=media_type)
        pixel_values, num_patches_list = self.collector.pipeline.preprocess(media_list)
        
        # Prompt 构造 (简化版，直接复用 collector 代码块逻辑)
        if not hasattr(self.model, 'conv_template'):
            template = copy.deepcopy(self.model.conv_template)
        else:
            template = copy.deepcopy(self.model.conv_template)
            
        if '<image>' not in prompt:
            prefix = "".join([f"Frame {i+1}: <image>\n" for i in range(len(num_patches_list))]) if len(num_patches_list) > 1 else '<image>\n'
            prompt = prefix + prompt
            
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        
        IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = '<img>', '</img>', '<IMG_CONTEXT>'
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            
        model_inputs = self.collector.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        B = pixel_values.shape[0]
        image_flags = torch.ones((B, 1), dtype=torch.long, device=self.device)
        
        # 2. 运行 FP16 基准 (Baseline)
        print(">>> [Sensitivity] Running FP16 Baseline...")
        with torch.no_grad():
            base_out = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                output_hidden_states=True
            )
        # 取最后一层 hidden state 作为金标准
        baseline_final = base_out.hidden_states[-1].float()

        # 3. 逐层量化攻击
        layers = self.model.language_model.model.layers
        scores = []
        
        print(f"\n>>> [Sensitivity] Scanning {len(layers)} layers with W{n_bits} quantization...")
        print(f"{'Layer':<6} | {'MSE (1e-3)':<10}")
        print("-" * 20)
        
        for i, layer in enumerate(layers):
            # A. 锁定现场：备份权重
            targets = self._get_target_modules(layer)
            backups = [m.weight.data.clone() for m in targets]
            
            # B. 实施攻击：原位替换为量化权重
            for m in targets:
                # 调用你的量化函数 (Per-Channel, Asymmetric usually better but here use default)
                # 使用你提供的 quantize_weight_per_channel_absmax
                m.weight.data = quantize_weight_per_channel_absmax(m.weight.data, n_bits=n_bits, zero_point=True)
            
            # C. 观测结果：运行推理
            with torch.no_grad():
                q_out = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_flags=image_flags,
                    output_hidden_states=True
                )
            
            # D. 计算损失
            q_final = q_out.hidden_states[-1].float()
            mse = torch.nn.functional.mse_loss(baseline_final, q_final).item()
            
            # E. 恢复现场：回滚权重 (非常重要！)
            for m, w_orig in zip(targets, backups):
                m.weight.data = w_orig
            
            # 记录
            scores.append({"layer": i, "mse": mse})
            print(f"{i:<6} | {mse*1000:.4f}")

        # 4. 保存结果
        self._save_and_plot(scores, output_dir, n_bits)

    def _save_and_plot(self, scores, output_dir, n_bits):
        path = os.path.join(output_dir, f"sensitivity_w{n_bits}.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["layer", "mse"])
            writer.writeheader()
            writer.writerows(scores)
            
        layers = [s['layer'] for s in scores]
        mses = [s['mse'] for s in scores]
        
        plt.figure(figsize=(10, 6))
        plt.plot(layers, mses, 'r-o', linewidth=2, label=f'W{n_bits} Sensitivity')
        plt.title(f'Layer-wise Sensitivity (W{n_bits} Quantization)')
        plt.xlabel('Layer Index')
        plt.ylabel('Output MSE Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"sensitivity_w{n_bits}.png"))
        print(f">>> [Sensitivity] Results saved to {output_dir}")