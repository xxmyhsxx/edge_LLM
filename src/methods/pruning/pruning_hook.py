# src/methods/pruning/pruner_hook.py
import torch
from transformers.models.llama.modeling_llama import LlamaAttention


class PruningHook:
    def __init__(self, tokenizer, image_patch_mapper, pruning_config):
        self.tokenizer = tokenizer
        self.image_patch_mapper = image_patch_mapper  # 用于可视化
        self.pruning_config = pruning_config

        self.removed_token_ids = []  # 记录被删除的文本 token 索引
        self.removed_patch_indices = []  # 记录被删除的图片 patch 索引 (全局索引)

        self.num_text_tokens = 0  # 记录本次推理的文本 token 数量
        self.num_image_patches = 0  # 记录本次推理的图片 patch 数量
        self.num_patches_list = []  # InternVL 切片后，每个 448 块有多少个 patch

        self.reset_records()

    def reset_records(self):
        """重置记录，在每次推理前调用"""
        self.removed_token_ids = []
        self.removed_patch_indices = []
        self.num_text_tokens = 0
        self.num_image_patches = 0
        self.num_patches_list = []

    def _get_token_type(self, token_idx):
        """判断一个 token 是文本还是图片 Patch"""
        if token_idx < self.num_image_patches:
            return "image_patch"
        elif token_idx < self.num_image_patches + self.num_text_tokens:
            return "text_token"
        return "unknown"

    def __call__(self, module, inputs, output):
        """
        前向传播 Hook 函数。
        此函数会在每个注册它的 Attention 模块执行 forward 之后被调用。

        Args:
            module: 当前的 Attention 模块 (e.g., LlamaAttention)
            inputs: 传递给 module 的输入
            output: module 的输出 (tuple: (hidden_states, attn_weights, past_key_value))
        """
        # 1. 提取 Attention 输出 (通常是 output[1] 或 output.attentions)
        # 注意：这里的索引可能需要根据 InternVL 的具体模型结构调整
        # Llama-based 模型通常 outputs 是 (hidden_states, past_key_value, attentions)
        # 或者 outputs[1] 是 attention_weights

        # 假设 output[1] 是 attention_weights: [B, H, Seq_len, Seq_len]
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attn_weights = output[1]
        elif hasattr(output, 'attentions') and output.attentions is not None:
            attn_weights = output.attentions
        else:
            # 如果 Attention weights 无法从 output 中直接获取，需要更深的修改
            # 或者确保 model(**inputs, output_attentions=True)
            print("Warning: Could not extract attention weights from module output.")
            return output  # 不进行剪枝

        # 为了简化，我们假设attn_weights是 [B, H, Seq, Seq]
        # [B, H, Q_Seq, K_Seq]

        # 2. 计算每个 Token 的重要性 (比如，被 Query 的次数)
        # 聚合 Head 和 Batch 维度
        importance_scores = attn_weights.mean(dim=(0, 1)).sum(dim=0)  # [K_Seq]

        # 3. 决定要删除哪些 Token
        seq_len = importance_scores.shape[0]
        num_tokens_to_keep = int(seq_len * (1 - self.pruning_config['ratio']))

        # 确保至少保留一个 Token
        if num_tokens_to_keep < 1: num_tokens_to_keep = 1

        _, topk_indices = torch.topk(importance_scores, num_tokens_to_keep, largest=True)

        # 创建一个 Mask
        pruning_mask = torch.zeros(seq_len, dtype=torch.bool, device=importance_scores.device)
        pruning_mask[topk_indices] = True  # True 表示保留

        removed_indices = torch.nonzero(~pruning_mask).squeeze(1).tolist()

        # 4. 记录被删除的 Token
        for idx in removed_indices:
            token_type = self._get_token_type(idx)
            if token_type == "text_token":
                self.removed_token_ids.append(idx)
            elif token_type == "image_patch":
                self.removed_patch_indices.append(idx)

        # 5. 应用剪枝 (修改 output 的 hidden_states)
        # 注意：这部分复杂，因为它需要修改 module 的原始输出
        # 通常需要修改 `output[0]` (hidden_states)
        # 并且要保证 Key, Value Cache 也相应剪枝
        # 为了简化，这里只是演示了如何获取信息，实际剪枝逻辑需要更深入地修改

        # 假设 output[0] 是 hidden_states: [B, Seq_len, Hidden_dim]
        # modified_hidden_states = output[0][:, pruning_mask, :]
        # return (modified_hidden_states,) + output[1:]

        return output  # 暂时不修改输出，只记录信息