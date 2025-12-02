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

from src.methods.research.collector import InternVLCollector
from src.methods.quantization.quant_tensor import quantize_weight_per_channel_absmax

class DecoupledSensitivityProbe:
    def __init__(self, model_path):
        self.collector = InternVLCollector(model_path)
        self.model = self.collector.model
        self.device = self.model.device
        
    def _get_target_modules(self, layer):
        modules = []
        if hasattr(layer, 'self_attn'):
            modules.extend([layer.self_attn.q_proj, layer.self_attn.k_proj, 
                            layer.self_attn.v_proj, layer.self_attn.o_proj])
        if hasattr(layer, 'mlp'):
            modules.extend([layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj])
        return modules

    def run_benchmark(self, media_path, prompt, n_bits=8, output_dir="/app/Edge-LMM-Optimizer/experiments/results/decoupled_sensitivity"):
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 准备输入 & 获取 Mask (复用 Collector 的精髓)
        print(">>> [DecoupledSensitivity] Preparing inputs & masks...")
        
        # 为了拿到 mask，我们需要手动跑一遍 prompt 构造逻辑 (或者提取 collector 的内部逻辑)
        # 这里简化处理：直接利用 collector.collect_activations 的副产品不太容易，我们重写一部分逻辑
        # 重点是拿到 input_ids 和 mask
        
        from src.utils.media_loader import MediaLoader
        media_type = 'video' if 'mp4' in media_path else 'image'
        media_list = MediaLoader.load(media_path, media_type=media_type)
        pixel_values, num_patches_list = self.collector.pipeline.preprocess(media_list)
        
        # 构造 Prompt
        template = copy.deepcopy(self.model.conv_template)
        if '<image>' not in prompt:
            prefix = "".join([f"Frame {i+1}: <image>\n" for i in range(len(num_patches_list))]) if len(num_patches_list) > 1 else '<image>\n'
            prompt = prefix + prompt
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        
        IMG_START, IMG_END, IMG_CTX = '<img>', '</img>', '<IMG_CONTEXT>'
        for num_patches in num_patches_list:
            query = query.replace('<image>', IMG_START + IMG_CTX * self.model.num_image_token * num_patches + IMG_END, 1)
            
        model_inputs = self.collector.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        B = pixel_values.shape[0]
        image_flags = torch.ones((B, 1), dtype=torch.long, device=self.device)
        
        # --- 生成 Mask ---
        input_ids_cpu = input_ids[0].cpu()
        vis_id = self.collector.img_context_token_id
        
        visual_mask = (input_ids_cpu == vis_id).to(self.device)
        text_mask = ~visual_mask # 简单起见，所有非视觉都算文本 (包含 system 和 user)
        
        # 2. FP16 Baseline
        print(">>> [DecoupledSensitivity] Running FP16 Baseline...")
        with torch.no_grad():
            base_out = self.model(
                pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask,
                image_flags=image_flags, output_hidden_states=True
            )
        base_final = base_out.hidden_states[-1] # [1, Seq, Dim]

        # 3. 逐层攻击
        layers = self.model.language_model.model.layers
        scores = []
        
        print(f"\n>>> [DecoupledSensitivity] Scanning {len(layers)} layers...")
        print(f"{'Lyr':<4} | {'MSE_Vis':<10} | {'MSE_Txt':<10} | {'Ratio(V/T)':<10}")
        print("-" * 45)
        
        for i, layer in enumerate(layers):
            targets = self._get_target_modules(layer)
            backups = [m.weight.data.clone() for m in targets]
            
            # Quantize
            for m in targets:
                m.weight.data = quantize_weight_per_channel_absmax(m.weight.data, n_bits=n_bits, zero_point=True)
            
            # Forward
            with torch.no_grad():
                q_out = self.model(
                    pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask,
                    image_flags=image_flags, output_hidden_states=True
                )
            
            # --- 核心：解耦误差计算 ---
            q_final = q_out.hidden_states[-1]
            
            # 使用 Mask 切分
            # diff shape: [1, Seq, Dim]
            diff = (base_final - q_final).pow(2) # Squared Error
            
            # Visual MSE: 只看 mask 为 True 的位置
            # mean over (Visual_Tokens, Dim)
            mse_vis = diff[:, visual_mask, :].mean().item()
            
            # Text MSE
            mse_txt = diff[:, text_mask, :].mean().item()

            # 计算它们的相对误差
            rel_mse_vis = mse_vis / (torch.mean(base_final[:, visual_mask, :] ** 2).item() + 1e-9)
            rel_mse_txt = mse_txt / (torch.mean(base_final[:, text_mask, :] ** 2).item() + 1e-9)
            
            # 恢复权重
            for m, w_orig in zip(targets, backups):
                m.weight.data = w_orig
            
            ratio = rel_mse_vis / (rel_mse_txt + 1e-9)
            scores.append({"layer": i, "mse_vis": rel_mse_vis, "mse_txt": rel_mse_txt, "ratio": ratio})
            print(f"{i:<4} | {rel_mse_vis*1000:.4f}     | {rel_mse_txt*1000:.4f}     | {ratio:.2f}")

        self._plot_results(scores, output_dir, n_bits)

    def _plot_results(self, scores, output_dir, n_bits):
        layers = [s['layer'] for s in scores]
        vis = [s['mse_vis'] for s in scores]
        txt = [s['mse_txt'] for s in scores]
        
        plt.figure(figsize=(10, 6))
        plt.plot(layers, vis, 'r-o', label='Visual MSE', linewidth=2)
        plt.plot(layers, txt, 'b-^', label='Text MSE', linewidth=2, alpha=0.7)
        
        plt.title(f'Decoupled Sensitivity Analysis (W{n_bits})')
        plt.xlabel('Layer')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "decoupled_sensitivity.png"))
        
        # 保存 CSV
        path = os.path.join(output_dir, "decoupled_sensitivity.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["layer", "mse_vis", "mse_txt", "ratio"])
            writer.writeheader()
            writer.writerows(scores)
        print(f">>> [DecoupledSensitivity] Saved to {output_dir}")


if __name__ == "__main__":
    model = "/app/models/InternVL3_5-4B-Instruct"
    image = "/app/eslm/test/Test/Video/001.mp4"
    probe = DecoupledSensitivityProbe(model)
    probe.run_benchmark(image, "Describe this.", n_bits=4)