# tests/check_env.py
import torch
import sys
import os


def check_environment():
    print("=" * 30)
    print("🔍 环境体检报告")
    print("=" * 30)

    # 1. Python & PyTorch 版本
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")

    # 2. CUDA 检查
    if torch.cuda.is_available():
        print(f"CUDA Available: ✅ Yes")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

        # 3. 显存检查 (关键!)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Total GPU Memory: {total_mem:.2f} GB")

        # 尝试分配一个小 Tensor 测试
        try:
            x = torch.ones(1).cuda()
            print("Tensor Allocation: ✅ Success")
        except Exception as e:
            print(f"Tensor Allocation: ❌ Failed ({e})")
    else:
        print("CUDA Available: ❌ No (你在用 CPU 跑吗?)")


if __name__ == "__main__":
    check_environment()