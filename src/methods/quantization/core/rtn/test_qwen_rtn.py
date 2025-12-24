"""
Qwen RTN 量化测试
使用 HuggingFace Qwen 模型测试 RTN 量化效果
"""
import sys
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from methods.quantization.core.rtn import W4Linear, W6Linear, W8Linear
from methods.quantization.core.base import QuantizationConfig
from methods.quantization.models.quant_model import QuantModelWrapper


class QwenRTNTester:
    """Qwen RTN 量化测试器"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.test_data = None
        
    def load_model(self):
        """加载 Qwen 模型"""
        print("=" * 60)
        print("加载 Qwen 模型...")
        print("=" * 60)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        print(f"✅ 模型加载完成: {self.model_path}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
        
    def generate_test_data(self):
        """生成测试数据"""
        print("\n生成测试数据...")
        
        prompts = [
            "今天天气很好，我们",
            "人工智能是",
            "Python编程语言的特点包括",
            "机器学习的主要应用有",
            "深度学习模型训练需要"
        ]
        
        self.test_data = prompts
        print(f"✅ 生成 {len(prompts)} 个测试提示")
        
    def get_model_size(self, model):
        """计算模型大小"""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size / (1024 * 1024)  # MB
    
    def test_original_model(self):
        """测试原始模型"""
        print("\n" + "=" * 60)
        print("测试原始模型")
        print("=" * 60)
        
        # 内存使用
        memory_mb = self.get_model_size(self.model)
        
        # 推理速度测试
        start_time = time.time()
        total_tokens = 0
        
        for prompt in self.test_data[:3]:  # 测试3个
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                total_tokens += outputs.shape[1]
        
        inference_time = time.time() - start_time
        tokens_per_second = total_tokens / inference_time
        
        # 生成质量测试
        test_prompt = "人工智能是"
        inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"内存使用: {memory_mb:.2f} MB")
        print(f"推理速度: {tokens_per_second:.2f} tokens/s")
        print(f"生成示例: {generated}")
        
        return {
            'memory_mb': memory_mb,
            'tokens_per_second': tokens_per_second,
            'generated': generated
        }
    
    def test_quantized_model(self, n_bits):
        """测试量化模型"""
        print("\n" + "=" * 60)
        print(f"测试 {n_bits}-bit 量化模型")
        print("=" * 60)
        
        # 创建量化配置
        config = QuantizationConfig(method='rtn', n_bits=n_bits, symmetric=False, per_channel=True)
        
        # 创建量化包装器
        wrapper = QuantModelWrapper(self.model, config)
        
        # 执行量化
        print("开始量化...")
        wrapper.quantize()
        
        # 量化后内存使用
        memory_mb = self.get_model_size(self.model)
        
        # 推理速度测试
        start_time = time.time()
        total_tokens = 0
        
        for prompt in self.test_data[:3]:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                total_tokens += outputs.shape[1]
        
        inference_time = time.time() - start_time
        tokens_per_second = total_tokens / inference_time
        
        # 生成质量测试
        test_prompt = "人工智能是"
        inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"内存使用: {memory_mb:.2f} MB")
        print(f"推理速度: {tokens_per_second:.2f} tokens/s")
        print(f"生成示例: {generated}")
        
        # 计算压缩比
        original_memory = wrapper.get_quantization_stats()['original_size_mb']
        compression_ratio = original_memory / memory_mb if memory_mb > 0 else 0
        
        print(f"压缩比: {compression_ratio:.2f}x")
        
        return {
            'memory_mb': memory_mb,
            'tokens_per_second': tokens_per_second,
            'generated': generated,
            'compression_ratio': compression_ratio
        }
    
    def run_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 80)
        print("Qwen RTN 量化测试开始")
        print("=" * 80)
        
        # 加载模型
        self.load_model()
        
        # 生成测试数据
        self.generate_test_data()
        
        # 测试原始模型
        original_results = self.test_original_model()
        
        # 测试不同位宽的量化模型
        results = {}
        for n_bits in [4, 6, 8]:
            results[n_bits] = self.test_quantized_model(n_bits)
        
        # 结果对比
        print("\n" + "=" * 80)
        print("测试结果汇总")
        print("=" * 80)
        
        print(f"{'模型类型':<12} {'内存(MB)':<12} {'速度(tokens/s)':<15} {'压缩比':<10} {'生成质量'}")
        print("-" * 100)
        
        print(f"{'原始':<12} {original_results['memory_mb']:<12.2f} {original_results['tokens_per_second']:<15.2f} {'1.00x':<10} {'原始'}")
        
        for n_bits, result in results.items():
            print(f"{'W' + str(n_bits):<12} {result['memory_mb']:<12.2f} {result['tokens_per_second']:<15.2f} {result['compression_ratio']:<10.2f} {'W' + str(n_bits)}")
        
        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)


def main():
    """主函数"""
    # 模型路径 - 用户需要修改
    MODEL_PATH = "Qwen/Qwen2-7B"  # 请修改为您的模型路径
    
    print(f"模型路径: {MODEL_PATH}")
    print("请确认模型路径正确，然后按回车继续...")
    input()
    
    try:
        tester = QwenRTNTester(MODEL_PATH)
        tester.run_tests()
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
