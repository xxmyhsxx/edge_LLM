# src/utils/metrics.py
import time
import torch
import contextlib
import gc
import numpy as np


class PerformanceMonitor:
    """
    全能性能监控器：同时测量 耗时(Latency) 和 显存(VRAM)
    """

    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.latency = 0
        self.peak_memory_gb = 0
        self.start_memory_gb = 0

    @contextlib.contextmanager
    def track(self, device="cuda"):
        """
        用法:
        with monitor.track():
            model.generate(...)
        """
        # 1. 清理现场 (可选，为了测得更准)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 2. 记录开始状态
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 等待之前的 GPU 任务结束
            self.start_memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)

        self.start_time = time.perf_counter()

        yield  # 执行原本的代码

        # 3. 记录结束状态
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 等待当前的 GPU 任务结束
            self.peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        self.end_time = time.perf_counter()
        self.latency = self.end_time - self.start_time

    def report(self):
        """返回字典格式的报告"""
        return {
            "Latency (s)": round(self.latency, 4),
            "Peak Memory (GB)": round(self.peak_memory_gb, 2),
            "Memory Growth (GB)": round(self.peak_memory_gb - self.start_memory_gb, 2)
        }


class TokenUtils:
    """
    Token 工具箱：计算吞吐量
    """

    @staticmethod
    def count_tokens(text, tokenizer=None):
        """
        计算 Token 数。
        如果有 tokenizer 就用 tokenizer 算（准）；
        如果没有，就按经验值估算（快）。
        """
        if tokenizer:
            return len(tokenizer.encode(text))
        else:
            # 经验公式：平均 1 个 token ≈ 0.75 个英文单词 或 0.6 个汉字
            # 这里简单粗暴按字符数/3 估算
            return max(1, int(len(text) / 3))

    @staticmethod
    def calc_throughput(num_tokens, latency_seconds):
        if latency_seconds <= 0: return 0.0
        return num_tokens / latency_seconds