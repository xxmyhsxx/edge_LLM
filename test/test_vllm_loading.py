# tests/test_vllm_loading.py
import sys
import os

# --- 关键：把项目根目录加入 Python 路径，这样才能 import src ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backends.vllm_backend.vllm_runner import VLLMRunner
import yaml


def test_vllm():
    print(">>> 正在测试 vLLM 后端加载...")

    # 1. 模拟一个配置 (或者读取真实 yaml)
    config = {
        'hardware': {
            'gpu_memory_utilization': 0.6,  # 保守一点
            'max_model_len': 1024,
            'swap_space': 0,
            'enforce_eager': True
        }
    }

    # 2. 指定模型路径 (请修改为你实际的路径)
    model_path = "/app/models/InternVL3_5-8B-AWQ-4bit"

    try:
        runner = VLLMRunner(config)
        runner.load(model_path)
        print("\n✅ vLLM 模型加载成功！没有 OOM！")

        # 简单生成测试
        output = runner.generate("Hello")
        print(f"生成结果: {output}")

    except Exception as e:
        print(f"\n❌ 加载失败: {e}")


if __name__ == "__main__":
    test_vllm()