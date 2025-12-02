import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import os
from abc import ABC, abstractmethod

# ==========================================
# 1. 定义指标接口 (Strategy Base Class)
# ==========================================
class BaseMetric(ABC):
    """所有指标的基类，强制要求实现计算逻辑和名称"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """指标名称，用于CSV表头和图表标题"""
        pass

    @abstractmethod
    def calculate(self, p_tensor: torch.Tensor, q_tensor: torch.Tensor) -> float:
        """
        计算逻辑
        :param p_tensor: 主分布 Tensor (L, D)
        :param q_tensor: 参考分布 Tensor (L, D)
        :return: 标量值
        """
        pass

    def _get_probs(self, tensor, eps=1e-8):
        """辅助函数：计算中心点的 Softmax 概率分布 (复用逻辑)"""
        center = tensor.float().mean(dim=0)
        prob = F.softmax(center, dim=-1).clamp(min=eps)
        return prob / prob.sum()

    def _get_center(self, tensor):
        """辅助函数：获取几何中心"""
        return tensor.float().mean(dim=0)

# ==========================================
# 2. 具体指标实现 (Concrete Strategies)
# ==========================================

class CosineGapMetric(BaseMetric):
    @property
    def name(self):
        return "Cosine_Gap"

    def calculate(self, p_tensor, q_tensor):
        p_center = self._get_center(p_tensor)
        q_center = self._get_center(q_tensor)
        sim = F.cosine_similarity(p_center.unsqueeze(0), q_center.unsqueeze(0)).item()
        return 1.0 - sim

class KLDivergenceMetric(BaseMetric):
    @property
    def name(self):
        return "KL_Div"

    def calculate(self, p_tensor, q_tensor):
        # KL(P||Q): P是真实分布(target)，Q是拟合分布(input)
        p_prob = self._get_probs(p_tensor)
        q_prob = self._get_probs(q_tensor)
        # F.kl_div 期望输入是 log_prob
        return F.kl_div(q_prob.log(), p_prob, reduction='sum').item()

class JSDivergenceMetric(BaseMetric):
    @property
    def name(self):
        return "JSD"

    def calculate(self, p_tensor, q_tensor):
        p_prob = self._get_probs(p_tensor)
        q_prob = self._get_probs(q_tensor)
        m_prob = 0.5 * (p_prob + q_prob)
        
        loss_1 = F.kl_div(m_prob.log(), p_prob, reduction='sum')
        loss_2 = F.kl_div(m_prob.log(), q_prob, reduction='sum')
        return 0.5 * (loss_1 + loss_2).item()

# 你可以轻松添加新指标，例如 L2 距离，而不需要修改分析器代码
class L2DistanceMetric(BaseMetric):
    @property
    def name(self):
        return "L2_Dist"

    def calculate(self, p_tensor, q_tensor):
        p_center = self._get_center(p_tensor)
        q_center = self._get_center(q_tensor)
        return torch.dist(p_center, q_center, p=2).item()


class MSEMetric(BaseMetric):
    """
    【均方误差】(Mean Squared Error)
    衡量特征中心点(Centroid)的绝对距离。
    反映了量化/剪枝是否导致了特征在空间中的绝对漂移。
    """
    @property
    def name(self):
        return "MSE"

    def calculate(self, p_tensor, q_tensor):
        # 比较几何中心 (D,)
        p_c = self._get_center(p_tensor)
        q_c = self._get_center(q_tensor)
        return F.mse_loss(p_c, q_c).item()


class SNRMetric(BaseMetric):
    """
    【信噪比】(Signal-to-Noise Ratio) - 单位: dB
    量化中最关键的指标。衡量‘原始信号能量’与‘噪声(误差)能量’的比率。
    SNR 越高，说明量化/剪枝后的恢复质量越好。
    """
    @property
    def name(self):
        return "SNR(dB)"

    def calculate(self, p_tensor, q_tensor):
        p_c = self._get_center(p_tensor) # 原始信号 (视为 Ground Truth)
        q_c = self._get_center(q_tensor) # 有噪信号
        
        noise = p_c - q_c
        
        # 信号功率 (Signal Power)
        signal_power = torch.sum(p_c ** 2)
        # 噪声功率 (Noise Power)
        noise_power = torch.sum(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
            
        # SNR = 10 * log10(P_signal / P_noise)
        snr = 10 * torch.log10(signal_power / noise_power)
        return snr.item()


class L2RelativeErrorMetric(BaseMetric):
    """
    【L2 相对误差】(Relative Error)
    计算公式: ||X - Y||_2 / ||X||_2
    消除了数值量级的影响。比如第1层数值很大，第10层数值很小，MSE没法比，
    但相对误差可以在不同层之间公平比较。
    """
    @property
    def name(self):
        return "Rel_Err"

    def calculate(self, p_tensor, q_tensor):
        p_c = self._get_center(p_tensor)
        q_c = self._get_center(q_tensor)
        
        diff_norm = torch.norm(p_c - q_c, p=2)
        ref_norm = torch.norm(p_c, p=2)
        
        if ref_norm == 0:
            return 0.0
            
        return (diff_norm / ref_norm).item()


class WassersteinMetric(BaseMetric):
    """
    【1D 推土机距离】(Wasserstein Distance)
    将所有数值视为一个 1D 分布，衡量两个分布的形状差异。
    特别适合：观察量化是否改变了数值的直方图分布（如是否导致了极值削波）。
    
    *注意：为了处理剪枝后 token 数量不一致的情况，我们使用分位数(Quantile)对齐法。*
    """
    @property
    def name(self):
        return "W_Dist"

    def calculate(self, p_tensor, q_tensor):
        # 1. 展平所有数值
        p_flat = p_tensor.flatten().float()
        q_flat = q_tensor.flatten().float()
        
        # 2. 如果数据量太大，随机采样以加速 (可选，这里为了精度先全量)
        # 3. 排序 (计算 Wasserstein 的核心步骤)
        p_sorted, _ = torch.sort(p_flat)
        q_sorted, _ = torch.sort(q_flat)
        
        # 4. 处理长度不一致 (如 Token Pruning 导致 q 变短)
        # 使用插值法将短的序列拉长，或者将长的序列缩短。这里通过线性插值对齐到相同长度。
        if p_sorted.numel() != q_sorted.numel():
            target_len = min(p_sorted.numel(), q_sorted.numel())
            # 使用 linspace 采样分位数
            p_sorted = self._resample(p_sorted, target_len)
            q_sorted = self._resample(q_sorted, target_len)
            
        # 5. 计算 L1 距离 (对于 1D 分布，Wasserstein-1 就是排序后的 L1 差值均值)
        w_dist = torch.abs(p_sorted - q_sorted).mean()
        return w_dist.item()

    def _resample(self, tensor, target_len):
        """简单的线性插值重采样"""
        if tensor.numel() == target_len:
            return tensor
        # 构造 grid
        idx = torch.linspace(0, tensor.numel() - 1, target_len).to(tensor.device)
        # 简单的最近邻或线性插值 (这里用简单的取整索引近似，对于大量数据足够)
        return tensor[idx.long()]

# ==========================================
# 3. 各种各样的分析器 (The Context)
# ==========================================

class DistributionAnalyzer:
    def __init__(self, output_dir="experiments/results/analysis", metrics=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 注册指标：如果没传，默认使用标准的一套
        if metrics is None:
            self.metrics = [CosineGapMetric(), JSDivergenceMetric(), KLDivergenceMetric()]
        else:
            self.metrics = metrics

    def add_metric(self, metric: BaseMetric):
        """允许动态添加指标"""
        self.metrics.append(metric)

    def run_analysis(self, distributions_a, distributions_b, name_a="Dist_A", name_b="Dist_B"):
        print(f">>> [Analyzer] Comparing {name_a} vs {name_b} with {len(self.metrics)} metrics...")
        
        layers = sorted(list(set(distributions_a.keys()) & set(distributions_b.keys())))
        results = []

        # 动态打印表头
        header = f"{'Lyr':<4} | " + " | ".join([f"{m.name:<10}" for m in self.metrics])
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        for idx in layers:
            tensor_a = distributions_a[idx]
            tensor_b = distributions_b[idx]
            
            # 跳过空数据
            if tensor_a.numel() == 0 or tensor_b.numel() == 0:
                continue

            row_data = {'layer': idx}
            
            # --- 核心改动：循环调用注册的 metrics ---
            log_str = f"{idx:<4} | "
            for metric in self.metrics:
                val = metric.calculate(tensor_a, tensor_b)
                row_data[metric.name] = val
                log_str += f"{val:<10.4f} | "
            
            print(log_str)
            results.append(row_data)

        # 保存与绘图
        file_suffix = f"{name_a}_vs_{name_b}"
        self._save_csv(results, file_suffix)
        self._plot_dynamic_curves(results, name_a, name_b, file_suffix)
        
        return results

    def _save_csv(self, results, suffix):
        if not results: return
        path = os.path.join(self.output_dir, f"metrics_{suffix}.csv")
        keys = results[0].keys()
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f">>> [Analyzer] CSV saved: {path}")

    def _plot_dynamic_curves(self, results, name_a, name_b, suffix):
        """完全动态的绘图逻辑，根据指标数量自动调整布局"""
        if not results: return
        
        num_metrics = len(self.metrics)
        layers = [r['layer'] for r in results]
        
        # 动态创建子图：例如 3个指标就是 (3,1)，4个指标就是 (4,1)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 3 * num_metrics), sharex=True)
        if num_metrics == 1: axes = [axes] # 兼容只有一个指标的情况

        # 颜色池，防止颜色不够用
        colors = ['tab:blue', 'tab:red', 'tab:purple', 'tab:green', 'tab:orange', 'tab:cyan']

        for i, metric in enumerate(self.metrics):
            ax = axes[i]
            values = [r[metric.name] for r in results]
            color = colors[i % len(colors)]
            
            ax.plot(layers, values, marker='o', color=color, linewidth=2, label=metric.name)
            ax.set_ylabel(metric.name)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            # 特殊处理：如果是 JSD，画一条 ln(2) 的参考线
            if metric.name == "JSD":
                ax.axhline(0.6931, color='gray', linestyle='--', alpha=0.5, label='ln(2)')

        axes[-1].set_xlabel('Layer Index')
        axes[0].set_title(f'Distribution Analysis: {name_a} vs {name_b}')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"plot_{suffix}.png")
        plt.savefig(save_path)
        print(f">>> [Analyzer] Plot saved: {save_path}")



