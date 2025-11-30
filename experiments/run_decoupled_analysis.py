import argparse
import os
import sys
import torch

sys.path.append(os.getcwd())

from src.methods.research.collector import InternVLCollector
from src.methods.research.analyzer import DistributionAnalyzer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="experiments/results/flexible_analysis")
    args = parser.parse_args()

    # 1. 采集数据 (一次性)
    collector = InternVLCollector(args.model)
    # raw_data 结构: {layer: {'visual': ..., 'text_pre': ..., 'text_post': ...}}
    raw_data = collector.collect_activations(args.image, "Describe this in detail.")

    # 2. 准备数据字典 (把数据按模态拆分成独立的字典)
    # 格式: {layer_idx: Tensor}
    visual_dict = {k: v['visual'] for k, v in raw_data.items()}
    sys_text_dict = {k: v['text_pre'] for k, v in raw_data.items()}
    user_text_dict = {k: v['text_post'] for k, v in raw_data.items()}
    
    # 还可以组合：所有文本
    all_text_dict = {}
    for k, v in raw_data.items():
        if v['text_pre'].numel() > 0 and v['text_post'].numel() > 0:
            all_text_dict[k] = torch.cat([v['text_pre'], v['text_post']], dim=0)
        elif v['text_post'].numel() > 0:
            all_text_dict[k] = v['text_post']

    vis_half1_dict = {}
    vis_half2_dict = {}
    
    for k, v in visual_dict.items():
        # v shape: [N_vis, Dim]
        split_point = v.shape[0] // 2
        if split_point > 0:
            vis_half1_dict[k] = v[:split_point]
            vis_half2_dict[k] = v[split_point:]
        else:
            # 如果太短没法切，就复制一份 (仅防报错)
            vis_half1_dict[k] = v
            vis_half2_dict[k] = v

    # 3. 只有在这里才实例化分析器
    analyzer = DistributionAnalyzer(output_dir=args.output)

    # --- 实验 A: 视觉 vs 系统提示词 (验证噪音) ---
    analyzer.run_analysis(visual_dict, sys_text_dict, name_a="Visual", name_b="SystemPrompt")

    # --- 实验 B: 视觉 vs 用户指令 (核心论据) ---
    analyzer.run_analysis(visual_dict, user_text_dict, name_a="Visual", name_b="UserInstruct")
    
    # --- 实验 C: 视觉 vs 全部文本 (传统视角) ---
    analyzer.run_analysis(visual_dict, all_text_dict, name_a="Visual", name_b="AllText")

    analyzer.run_analysis(user_text_dict, sys_text_dict, name_a="text", name_b="text")

    analyzer.run_analysis(vis_half1_dict, vis_half2_dict, 
                          name_a="Visual_Half1", name_b="Visual_Half2")

