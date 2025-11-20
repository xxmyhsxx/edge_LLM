# tests/debug_pruning.py
import torch
import torch.nn.functional as F


def simulate_attention_pruning():
    print(">>> 模拟 Attention Score 剪枝逻辑...")

    # 1. 造假数据: 假设有 1 个 Batch, 4 个 Head, 序列长度 10
    # Attention Map shape: [B, H, Seq_Len, Seq_Len]
    B, H, N = 1, 4, 10
    attn_weights = torch.rand(B, H, N, N)

    # 2. 模拟计算 Token 重要性 (比如把 attention column 求和)
    # shape: [B, H, N]
    importance_score = attn_weights.sum(dim=-2)

    # 对 Heads 取平均，得到每个 Token 的全局重要性
    # shape: [B, N]
    token_importance = importance_score.mean(dim=1)

    print(f"Token 重要性分数:\n{token_importance}")

    # 3. 测试 Top-K 选取
    keep_ratio = 0.5
    k = int(N * keep_ratio)
    topk_values, topk_indices = torch.topk(token_importance, k, dim=-1)

    print(f"保留的 Token 索引 (Top {k}):\n{topk_indices}")

    # 这里可以验证你的逻辑对不对，不需要启动庞大的 vLLM


if __name__ == "__main__":
    simulate_attention_pruning()