import argparse
import sys
import os
import torch

sys.path.append(os.getcwd())

from src.methods.research.layer_quant_analyzer import LayerQuantAnalyzer
from src.methods.research.analyzer import DistributionAnalyzer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--output", type=str, default="experiments/results/layer_sensitivity")
    args = parser.parse_args()

    # 1. 采集数据
    analyzer_tool = LayerQuantAnalyzer(args.model)
    # baseline_data: {'visual': T, 'text_post': T}
    # quant_results: {0: {'visual': T...}, 1: {...}}
    baseline_data, quant_results = analyzer_tool.collect_data(args.image, "Describe this.", n_bits=args.bits)

    if not baseline_data:
        print("Error collecting data.")
        exit()

    # 2. 构建对比流 (Stream Construction)
    # 我们要对比: "Baseline Visual" vs "Layer-i-Quantized Visual"
    
    layers = sorted(quant_results.keys())
    
    # 构建 Baseline 流 (每个层都一样，都是原始 FP16)
    stream_base_vis = {l: baseline_data['visual'] for l in layers}
    stream_base_txt = {l: baseline_data['text_post'] for l in layers}
    
    # 构建 Quantized 流
    stream_quant_vis = {l: quant_results[l]['visual'] for l in layers}
    stream_quant_txt = {l: quant_results[l]['text_post'] for l in layers}

    # 3. 复用 Analyzer 进行分析
    # Analyzer 不知道这是"敏感度分析"，它只知道在对比两个分布
    # 但结果的物理含义变了：
    #   - KL(Base || Quant): 量化导致的信息损失 (Sensitivity)
    #   - Gap(Base, Quant): 量化导致的几何漂移
    
    analyzer = DistributionAnalyzer(output_dir=args.output)
    
    print("\n>>> Analysis 1: Visual Sensitivity (FP16 vs Layer-Quant)")
    analyzer.run_analysis(stream_quant_vis, stream_quant_txt, name_a="quant-vis", name_b="quant-text")

    analyzer.run_analysis(stream_base_vis, stream_quant_vis, name_a="base-vis", name_b="quant-vis")

    analyzer.run_analysis(stream_base_txt, stream_quant_txt, name_a="base-txt", name_b="quant-txt")




    