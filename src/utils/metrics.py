import time
import torch
import gc
import contextlib

class PerformanceMonitor:
    """
    性能监控器：用于测量代码块的 执行时间(Latency) 和 显存峰值(Peak Memory)。
    专为 PyTorch/Jetson 优化，包含 CUDA 同步逻辑。
    """
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.peak_memory = 0
        self.start_memory = 0

    @contextlib.contextmanager
    def track(self, device="cuda"):
        """
        上下文管理器，用于包裹需要测试的代码块。
        """
        # 1. 环境清理 (确保测试准确，不受之前残留影响)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize() # 等待 GPU 完成所有旧任务
        
        gc.collect()

        # 2. 记录开始状态
        self.start_time = time.perf_counter()
        if torch.cuda.is_available():
            self.start_memory = torch.cuda.memory_allocated()

        # --- 执行被包裹的代码 ---
        yield 
        # ----------------------

        # 3. 记录结束状态
        if torch.cuda.is_available():
            torch.cuda.synchronize() # 等待 GPU 完成当前任务 (关键!)
            self.peak_memory = torch.cuda.max_memory_allocated()
        
        self.end_time = time.perf_counter()

    @property
    def latency(self):
        """返回秒数"""
        return self.end_time - self.start_time

    @property
    def peak_memory_gb(self):
        """返回峰值显存 (GB)"""
        return self.peak_memory / (1024**3)

    def get_report(self):
        return {
            "latency_sec": round(self.latency, 4),
            "peak_memory_gb": round(self.peak_memory_gb, 2)
        }

class TPSCalculator:
    """
    计算 Tokens Per Second (吞吐量) 的辅助工具
    """
    @staticmethod
    def calculate(output_text, latency, tokenizer=None):
        if latency <= 0: return 0.0
        
        # 如果有 tokenizer，计算准确的 token 数
        if tokenizer:
            num_tokens = len(tokenizer.encode(output_text))
        else:
            # 否则按字符估算 (粗略：1 token ≈ 3 chars)
            num_tokens = len(output_text) // 3
            
        return num_tokens / latency