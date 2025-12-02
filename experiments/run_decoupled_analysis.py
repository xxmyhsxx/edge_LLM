import argparse
import os
import sys
import torch

from src.methods.research.collector import InternVLCollector
from src.methods.research.analyzer import DistributionAnalyzer



def data(raw_data):
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
    return visual_dict,vis_half1_dict,vis_half2_dict,all_text_dict,sys_text_dict,user_text_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/app/models/InternVL3_5-4B-Instruct")
    parser.add_argument("--image", type=str,default="/app/eslm/test/Test/Video/001.mp4" )
    parser.add_argument("--output", type=str, default="/app/Edge-LMM-Optimizer/experiments/results/text1")
    args = parser.parse_args()

    # 1. 采集数据 (一次性)
    quant_policy = {}

    for i in range(36):  # InternVL-4B 有 36 层 (0-35)
        layer_idx = str(i)
    
    # 1. 异常点 (Outlier): Layer 6 在 W8 下依然有 MSE 尖峰，必须 FP16
        if i == 6:
            quant_policy[layer_idx] = 16
        
    # 2. 浅层保护区 (Shallow Protection): L0-L15 视觉特征脆弱，使用 W8
        elif i <= 15:
            quant_policy[layer_idx] = 4
        
    # 3. 输出保护区 (Head Protection): L32-L35 影响最终预测，使用 W8
        elif i >= 34:
            quant_policy[layer_idx] = 16
        
    # 4. 深层压缩区 (Deep Compression): L16-L31 模态正交且鲁棒，安全使用 W4
        else:
            quant_policy[layer_idx] = 4
    
    collector = InternVLCollector(args.model,quant_policy=quant_policy)
    # raw_data 结构: {layer: {'visual': ..., 'text_pre': ..., 'text_post': ...}}
    raw_data = collector.collect_activations(args.image, "Describe this in detail.")
    quantvisall,quantvis1,quantvis2,quanttxtall,quanttxtsys,quanttxtuser = data(raw_data)
    del collector



    collector1 = InternVLCollector(args.model)
    raw_data1 = collector1.collect_activations(args.image, "Describe this in detail.")
    visall,vis1,vis2,txtall,txtsys,txtuser = data(raw_data1)

    del collector1
    from src.methods.research.analyzer import CosineGapMetric,L2DistanceMetric,KLDivergenceMetric,JSDivergenceMetric,SNRMetric,L2RelativeErrorMetric,WassersteinMetric
    # 实例化分析器

    modality_metrics = [
    CosineGapMetric(),      # 看方向偏离
    L2DistanceMetric(),            # 看空间距离
    JSDivergenceMetric()    # 看分布重合度 (对称)
    ]

    analyzer = DistributionAnalyzer(output_dir=args.output,metrics=modality_metrics)
    


    # 实验1: 探究原始模型的分布情况
    analyzer.run_analysis(visall,txtall,name_a="visall",name_b="txtall")
    analyzer.run_analysis(visall,txtsys,name_a="visall",name_b="txtsys")
    analyzer.run_analysis(visall,txtuser,name_a="visall",name_b="txtuser")

    # 实验2: 探究量化模型的分布情况

    analyzer.run_analysis(quantvisall,quanttxtall,name_a="quantvisall",name_b="quanttxtall")
    analyzer.run_analysis(quantvisall,quanttxtsys,name_a="quantvisall",name_b="quanttxtsys")
    analyzer.run_analysis(quantvisall,quanttxtuser,name_a="quantvisall",name_b="quanttxtuser")

    # 实验3: 探究模型量化前后的分布改变
    quant_metrics = [
    SNRMetric(),             # 必须：看精度 (dB)
    WassersteinMetric(),     # 必须：看有没有发生 Outlier 截断
    L2RelativeErrorMetric(), # 辅助：看整体误差百分比
    CosineGapMetric()        # 辅助：看语义方向
    ]

    analyzer = DistributionAnalyzer(output_dir=args.output,metrics=quant_metrics)

    analyzer.run_analysis(visall,quantvisall,name_a="vis",name_b="quantvis")
    analyzer.run_analysis(txtall,quanttxtall,name_a="txt",name_b="quanttxt")
    analyzer.run_analysis(txtuser,quanttxtuser,name_a="txtuser",name_b="quant")


    





