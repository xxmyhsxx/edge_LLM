# src/methods/pruning/token_pruner.py
import torch.nn as nn
from src.methods.pruning.pruner_hook import PruningHook
from src.utils.image_patch_mapper import ImagePatchMapper
from src.models.base import BaseModelWrapper  # 用于获取通用层


class TokenPruner:
    def __init__(self, tokenizer, image_patch_mapper: ImagePatchMapper, pruning_config):
        self.tokenizer = tokenizer
        self.image_patch_mapper = image_patch_mapper
        self.pruning_config = pruning_config  # 例如 {'ratio': 0.5, 'layer_to_prune': 10}
        self.pruning_hook_instance = PruningHook(tokenizer, image_patch_mapper, pruning_config)
        self.handles = []  # 存储 hook 的句柄，方便移除

    def apply(self, model_wrapper: BaseModelWrapper, inputs_metadata):
        """
        将剪枝逻辑应用到模型。
        Args:
            model_wrapper (BaseModelWrapper): 被包装的模型实例 (InternVLPipeline.model)
            inputs_metadata (dict): 包含 num_text_tokens, num_image_patches, num_patches_list 等信息
        """
        print(f">>> [Pruner] Applying Token Pruning (Ratio: {self.pruning_config['ratio']})...")

        # 清空之前的记录
        self.pruning_hook_instance.reset_records()

        # 更新 Hook 实例的 Token 边界信息
        self.pruning_hook_instance.num_text_tokens = inputs_metadata['num_text_tokens']
        self.pruning_hook_instance.num_image_patches = inputs_metadata['num_image_patches']
        self.pruning_hook_instance.num_patches_list = inputs_metadata['num_patches_list']

        # 1. 获取要注册 Hook 的层 (例如所有 Attention 层)
        layers = model_wrapper.get_layers()
        target_layer = layers[self.pruning_config['layer_to_prune']]

        # 2. 注册 Hook
        # 假设 InternVL 的 Attention 模块是 LlamaAttention
        # 你需要根据 InternVL 实际结构调整：可能是 target_layer.self_attn
        attn_module = model_wrapper.get_attention(target_layer)

        # 移除旧的 Hook
        self.remove_hooks()

        # 注册新的 Hook
        # register_forward_hook 在模块前向传播后被调用
        handle = attn_module.register_forward_hook(self.pruning_hook_instance)
        self.handles.append(handle)

        print(f"    Hook registered on layer {self.pruning_config['layer_to_prune']} "
              f"Attention module: {type(attn_module)}")

    def remove_hooks(self):
        """移除所有注册的 Hook"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def get_pruning_results(self):
        """获取剪枝结果，用于可视化"""
        return {
            "removed_token_ids": self.pruning_hook_instance.removed_token_ids,
            "removed_patch_indices": self.pruning_hook_instance.removed_patch_indices,
            "num_text_tokens": self.pruning_hook_instance.num_text_tokens,
            "num_image_patches": self.pruning_hook_instance.num_image_patches,
            "num_patches_list": self.pruning_hook_instance.num_patches_list,
        }