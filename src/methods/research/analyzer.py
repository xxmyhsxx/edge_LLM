import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

class DistributionAnalyzer:
    """
    【通用分布分析器】
    职责：只负责数学计算和画图，完全不关心数据的来源和名称。
    接口：传入两个按层索引的 Tensor 字典，自动计算 Gap/JSD/KL 并绘图。
    """
    def __init__(self, output_dir="experiments/results/analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _compute_metrics_pair(self, p_tensor, q_tensor):
        """
        计算两个分布(Tensor)之间的所有核心指标
        P: 主分布 (如 Visual)
        Q: 参考分布 (如 Text)
        """
        if p_tensor.numel() == 0 or q_tensor.numel() == 0:
            return {"gap": float('nan'), "jsd": float('nan'), "kl": float('nan')}

        p_data = p_tensor.float()
        q_data = q_tensor.float()

        # 1. 几何中心 (Centroids)
        p_center = p_data.mean(dim=0)
        q_center = q_data.mean(dim=0)

        # --- Metric 1: Cosine Gap ---
        sim = F.cosine_similarity(p_center.unsqueeze(0), q_center.unsqueeze(0)).item()
        gap = 1.0 - sim

        # --- 转换为概率分布 (Softmax) ---
        # 加上 epsilon 防止数值问题
        eps = 1e-8
        p_prob = F.softmax(p_center, dim=-1).clamp(min=eps)
        q_prob = F.softmax(q_center, dim=-1).clamp(min=eps)
        
        # 重新归一化
        p_prob = p_prob / p_prob.sum()
        q_prob = q_prob / q_prob.sum()

        # --- Metric 2: KL Divergence (P || Q) ---
        # P是真实分布，Q是拟合分布。衡量用Q编码P损失多少信息。
        kl = F.kl_div(q_prob.log(), p_prob, reduction='sum').item()

        # --- Metric 3: JSD (Symmetric) ---
        m_prob = 0.5 * (p_prob + q_prob)
        jsd = 0.5 * (F.kl_div(m_prob.log(), p_prob, reduction='sum') + 
                     F.kl_div(m_prob.log(), q_prob, reduction='sum')).item()

        return {"gap": gap, "jsd": jsd, "kl": kl}

    def run_analysis(self, distributions_a, distributions_b, name_a="Dist_A", name_b="Dist_B"):
        """
        主入口：传入两个字典 {layer_idx: tensor}，自动对其分析
        """
        print(f">>> [Analyzer] Comparing {name_a} vs {name_b} ...")
        
        # 找出公共层
        layers = sorted(list(set(distributions_a.keys()) & set(distributions_b.keys())))
        results = []

        print(f"{'Lyr':<4} | {'Gap':<8} | {'JSD':<8} | {'KL':<8}")
        print("-" * 40)

        for idx in layers:
            tensor_a = distributions_a[idx]
            tensor_b = distributions_b[idx]
            
            metrics = self._compute_metrics_pair(tensor_a, tensor_b)
            metrics['layer'] = idx
            results.append(metrics)
            
            print(f"{idx:<4} | {metrics['gap']:.4f}   | {metrics['jsd']:.4f}   | {metrics['kl']:.4f}")

        # 保存与绘图
        file_suffix = f"{name_a}_vs_{name_b}"
        self._save_csv(results, file_suffix)
        self._plot_curves(results, name_a, name_b, file_suffix)
        
        return results

    def _save_csv(self, results, suffix):
        path = os.path.join(self.output_dir, f"metrics_{suffix}.csv")
        if not results: return
        keys = results[0].keys()
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f">>> [Analyzer] CSV saved: {path}")

    def _plot_curves(self, results, name_a, name_b, suffix):
        if not results: return
        layers = [r['layer'] for r in results]
        gaps = [r['gap'] for r in results]
        kls = [r['kl'] for r in results]
        jsds = [r['jsd'] for r in results]

        # 画在一张大图里，但是分三个子图，清晰明了
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # 1. Gap
        ax1.plot(layers, gaps, marker='o', color='tab:blue', linewidth=2)
        ax1.set_ylabel('Cosine Gap')
        ax1.set_title(f'Geometric Separation: {name_a} vs {name_b}')
        ax1.grid(True, alpha=0.3)

        # 2. JSD
        ax2.plot(layers, jsds, marker='s', color='tab:red', linewidth=2)
        ax2.axhline(0.6931, color='gray', linestyle='--', label='ln(2)')
        ax2.set_ylabel('JSD')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. KL
        ax3.plot(layers, kls, marker='^', color='tab:purple', linewidth=2)
        ax3.set_ylabel(f'KL({name_a}||{name_b})')
        ax3.set_xlabel('Layer Index')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"plot_{suffix}.png")
        plt.savefig(save_path)
        print(f">>> [Analyzer] Plot saved: {save_path}")