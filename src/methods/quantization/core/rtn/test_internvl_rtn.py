"""
Test script for INT4/INT8/INT6 quantization of InternVL models using RTN (Round-to-Nearest) method.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers import AutoModel, AutoTokenizer

# 修正导入路径
from src.methods.quantization.core.base import QuantizationConfig
from src.methods.quantization.models.quant_model import QuantModelWrapper


class InternVLRTNTester:
    """RTN quantization tester for InternVL models."""
    
    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVL-Chat-V1-5",
        quant_bits: int = 4,
        group_size: int = 64,
        device: str = "cuda"
    ):
        """
        Initialize RTN tester for InternVL.
        
        Args:
            model_path: Path to the InternVL model
            quant_bits: Quantization bits (4, 6, or 8)
            group_size: Group size for per-group quantization
            device: Device to run on
        """
        self.model_path = model_path
        self.quant_bits = quant_bits
        self.group_size = group_size
        self.device = device
        
        # 修正：创建正确的量化配置（使用n_bits而不是bits）
        self.quant_config = QuantizationConfig(
            method="rtn",
            n_bits=quant_bits,
            group_size=group_size,
            symmetric=True,  # 使用symmetric而不是sym
            per_channel=True
        )
        
        self.model = None
        self.tokenizer = None
        self.quant_model = None
        
    def load_model(self):
        """Load InternVL model with proper configuration."""
        try:
            print(f"加载InternVL模型: {self.model_path}")
            
            # 修正：使用后端代码的正确方式
            # 使用bfloat16并直接移动到GPU，避免中间状态
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval().cuda()
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 确保模型在评估模式
            self.model.eval()
            
            print(f"模型加载成功，设备: {next(self.model.parameters()).device}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def quantize_model(self):
        """Apply RTN quantization to the model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("开始模型量化...")
        
        try:
            # 修正：QuantModelWrapper只需要model和quant_config
            
            self.model = QuantModelWrapper(
                model=self.model,
                quant_config=self.quant_config
            )
            self.quant_model = self.model
            
            # 应用量化（不需要校准数据，RTN直接基于权重）
            self.quant_model.quantize()
            
            print("模型量化完成")
            
        except Exception as e:
            print(f"量化失败: {e}")
            raise
    
    def generate_text(
        self,
        question: str,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text response using quantized model.
        
        Args:
            question: Input question
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text response
        """
        if self.quant_model is None:
            raise ValueError("Quantized model not available. Call quantize_model() first.")
        
        try:
            # 对于纯文本测试，直接使用模型的generate方法
            # 构建输入
            inputs = self.tokenizer(question, return_tensors="pt")
            
            # 移动到正确设备
            if hasattr(inputs, 'input_ids') and self.model is not None:
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 生成配置
            generate_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
            }
            
            # 安全地设置token ID
            if self.tokenizer is not None:
                pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
                eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
                
                if pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = int(pad_token_id)
                if eos_token_id is not None:
                    generate_kwargs["eos_token_id"] = int(eos_token_id)
            
            # 执行生成
            if self.quant_model is not None and hasattr(self.quant_model, 'generate'):
                with torch.no_grad():
                    outputs = self.quant_model.generate(**inputs, **generate_kwargs)
                
                # 解码输出 - 确保类型安全
                if self.tokenizer is not None and hasattr(self.tokenizer, 'decode'):
                    if isinstance(outputs, torch.Tensor):
                        response = self.tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
                    else:
                        response = str(outputs)
                else:
                    response = str(outputs[0].tolist())
            else:
                response = f"模型不支持生成: {type(self.quant_model)}"
            
            return response
            
        except Exception as e:
            print(f"生成失败: {e}")
            return f"生成错误: {e}"
    
    def calculate_model_size(self) -> Dict[str, float]:
        """Calculate model memory usage."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        # 修正：避免重复计算，只计算一次
        if not hasattr(self, '_memory_cache'):
            total_params = sum(p.numel() for p in self.model.parameters())
            total_bytes = total_params * 2  # FP16
            total_gb = total_bytes / (1024**3)
            
            # 量化后大小（近似）
            quant_ratio = self.quant_bits / 16
            quant_gb = total_gb * quant_ratio
            
            self._memory_cache = {
                "original_size_gb": round(total_gb, 3),
                "quantized_size_gb": round(quant_gb, 3),
                "compression_ratio": round(quant_ratio, 2)
            }
        
        return self._memory_cache
    
    def evaluate_performance(self, test_questions: List[str]) -> Dict:
        """Evaluate quantized model performance."""
        results = {
            "avg_latency_ms": 0,
            "success_rate": 0,
            "total_questions": len(test_questions),
            "successful_answers": 0
        }
        
        if not test_questions:
            return results
        
        import time
        
        latencies = []
        for i, question in enumerate(test_questions):
            try:
                start_time = time.time()
                response = self.generate_text(question)
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                if response and not response.startswith("生成错误"):
                    results["successful_answers"] += 1
                    
                print(f"问题 {i+1}/{len(test_questions)}: {latency_ms:.1f}ms")
                
            except Exception as e:
                print(f"问题 {i+1} 失败: {e}")
        
        if latencies:
            results["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)
        
        results["success_rate"] = round(
            results["successful_answers"] / results["total_questions"] * 100, 2
        )
        
        return results
    
    def run_full_test(
        self,
        test_questions: Optional[List[str]] = None
    ) -> Dict:
        """
        Run complete RTN quantization test pipeline.
        
        Args:
            test_questions: Optional list of test questions. If None, uses defaults.
            
        Returns:
            Test results dictionary
        """
        print("=" * 60)
        print("InternVL RTN量化测试开始")
        print("=" * 60)
        
        # 使用默认测试问题（如果未提供）
        if test_questions is None:
            test_questions = [
                "请介绍一下InternVL模型的特点",
                "什么是RTN量化方法？",
                "请解释神经网络的基本原理"
            ]
        
        results = {}
        
        try:
            # 1. 加载模型
            print("\n1. 加载模型...")
            self.load_model()
            
            # 2. 量化模型
            print("\n2. 量化模型...")
            self.quantize_model()
            
            # 3. 计算内存使用
            print("\n3. 计算内存使用...")
            memory_info = self.calculate_model_size()
            print(f"原始大小: {memory_info['original_size_gb']}GB")
            print(f"量化后: {memory_info['quantized_size_gb']}GB")
            print(f"压缩比: {memory_info['compression_ratio']}")
            
            # 5. 性能评估
            print(f"\n5. 性能评估 (测试问题: {len(test_questions)}个)...")
            performance = self.evaluate_performance(test_questions)
            print(f"成功率: {performance['success_rate']}%")
            print(f"平均延迟: {performance['avg_latency_ms']}ms")
            
            # 6. 单条测试
            print("\n6. 单条测试生成...")
            test_question = test_questions[0]
            print(f"测试问题: {test_question}")
            response = self.generate_text(test_question, max_new_tokens=50)
            print(f"生成回答: {response}")
            
            # 组合结果
            results.update({
                "quantization_config": {
                    "method": "rtn",
                    "bits": self.quant_bits,
                    "group_size": self.group_size
                },
                "memory_info": memory_info,
                "performance": performance,
                "test_response": response,
                "status": "success"
            })
            
            print("\n" + "=" * 60)
            print("测试完成!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n测试失败: {e}")
            results = {
                "status": "failed",
                "error": str(e)
            }
        
        return results


def main():
    """Main execution function."""
    # 配置参数
    model_path = os.getenv("INTERNVL_MODEL_PATH", "/app/models/InternVL3_5-4B-Instruct")
    quant_bits = 4
    group_size = 64
    
    # 创建测试器
    tester = InternVLRTNTester(
        model_path=model_path,
        quant_bits=quant_bits,
        group_size=group_size,
        device="cuda"
    )
    
    # 运行完整测试
    results = tester.run_full_test()
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("最终测试结果")
    print("=" * 60)
    print(f"状态: {results.get('status', 'unknown')}")
    
    if results.get("status") == "success":
        print(f"量化配置: {results['quantization_config']}")
        print(f"内存优化: {results['memory_info']}")
        print(f"性能表现: {results['performance']}")
    else:
        print(f"错误信息: {results.get('error', '未知错误')}")


if __name__ == "__main__":
    main()
