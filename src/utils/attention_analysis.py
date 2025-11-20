import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io


class AttentionAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_attention_scores(self, inputs, layer_idx=-1, aggregation="mean"):
        """
        运行一次前向传播，获取指定层的注意力得分。

        Args:
            inputs (dict): 模型输入 (input_ids, pixel_values 等)
            layer_idx (int): 获取哪一层的注意力 (-1 表示最后一层, None 表示所有层)
            aggregation (str): 多头注意力的聚合方式 'mean' (平均) or 'max' (最大值)

        Returns:
            dict: {
                'attention_map': numpy array [Seq_Len, Seq_Len],
                'tokens': list[str] (解码后的token列表)
            }
        """
        print(">>> [Analyzer] Extracting attention maps...")

        # 1. 强制模型输出 Attention (这是关键)
        # 使用 torch.no_grad 节省显存
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True
            )

        # outputs.attentions 是一个 tuple，包含每一层的 [Batch, Heads, Seq, Seq]
        # 我们取最后一层 (通常语义最丰富)
        if layer_idx is not None:
            # shape: [Batch, Heads, Seq, Seq]
            attn_tensor = outputs.attentions[layer_idx]
        else:
            # 如果需要所有层，这里内存可能会爆，建议只取一层
            # 这里简单平均所有层
            attn_tensor = torch.stack(outputs.attentions).mean(dim=0)

        # 2. 处理 Batch 维度 (假设 Batch=1)
        # shape: [Heads, Seq, Seq]
        attn_tensor = attn_tensor[0]

        # 3. 聚合多头 (Multi-Head Aggregation)
        if aggregation == "mean":
            # shape: [Seq, Seq]
            attn_matrix = attn_tensor.mean(dim=0)
        elif aggregation == "max":
            attn_matrix = attn_tensor.max(dim=0)[0]

        # 转为 Numpy 用于绘图
        attn_matrix = attn_matrix.float().cpu().numpy()

        # 4. 获取对应的 Token 文本
        input_ids = inputs['input_ids'][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        return {
            "matrix": attn_matrix,
            "tokens": tokens
        }

    def plot_heatmap(self, attn_data, save_path="attention_heatmap.png", title="Attention Map"):
        """
        绘制注意力热力图
        """
        matrix = attn_data['matrix']
        # tokens = attn_data['tokens'] # 如果 token 太多，画在轴上会看不清

        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, cmap="viridis", vmin=0, vmax=matrix.max())
        plt.title(title)
        plt.xlabel("Key Token Index")
        plt.ylabel("Query Token Index")

        # 只在 token 数较少时显示 label，否则太乱
        if len(matrix) < 50:
            # 简单的解码处理，去掉特殊字符
            labels = [t.replace('Ġ', '') for t in attn_data['tokens']]
            plt.xticks(np.arange(len(labels)), labels, rotation=90)
            plt.yticks(np.arange(len(labels)), labels, rotation=0)

        plt.savefig(save_path)
        print(f">>> [Analyzer] Heatmap saved to {save_path}")
        plt.close()

    def analyze_token_importance(self, attn_data, top_k=20):
        """
        计算 Token 重要性并标记
        原理：计算 Attention 矩阵的“列和” (Column Sum)。
        如果一列的和很大，说明很多其他 Token 都在关注这个 Token -> 它很重要。
        """
        matrix = attn_data['matrix']
        tokens = attn_data['tokens']

        # 1. 计算重要性得分 (被关注的总和)
        # axis=0 (sum over queries) -> 得到每个 Key 被关注的程度
        importance_scores = matrix.sum(axis=0)

        # 2. 排序
        sorted_indices = np.argsort(importance_scores)[::-1]  # 降序

        print(f"\n{'=' * 10} Top {top_k} Important Tokens {'=' * 10}")
        print(f"{'Rank':<5} | {'Index':<6} | {'Score':<8} | {'Token Content'}")
        print("-" * 45)

        important_indices = []
        for rank, idx in enumerate(sorted_indices[:top_k]):
            token_text = tokens[idx].replace('Ġ', '')
            score = importance_scores[idx]
            print(f"{rank + 1:<5} | {idx:<6} | {score:.4f}   | {token_text}")
            important_indices.append(idx)

        # 3. 计算占比 (例如：前 20% 的 token 占了多少注意力权重)
        total_attention = importance_scores.sum()
        top_k_attention = importance_scores[sorted_indices[:top_k]].sum()
        ratio = (top_k_attention / total_attention) * 100

        print("-" * 45)
        print(f"Top {top_k} tokens hold {ratio:.2f}% of total attention.")

        return important_indices, importance_scores

    def plot_importance_curve(self, scores, save_path="token_importance.png"):
        """
        绘制 Token 重要性分布曲线 (用来寻找剪枝阈值)
        """
        sorted_scores = np.sort(scores)[::-1]
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_scores)
        plt.title("Token Importance Distribution")
        plt.xlabel("Token Rank")
        plt.ylabel("Importance Score (Sum of Attn)")
        plt.grid(True)
        plt.savefig(save_path)
        print(f">>> [Analyzer] Importance curve saved to {save_path}")
        plt.close()