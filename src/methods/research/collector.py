import torch
import copy
import os
import sys

# 自动添加项目根目录到 path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.backends.torch_backend.internvl import InternPyTorchBackend
from src.pipelines.intervl_pipeline import InternVLPipeline
from src.utils.media_loader import MediaLoader

class InternVLCollector:
    """
    【数据采集器 - 对照实验版】
    功能：
    1. 自动过滤特殊 Token。
    2. 将文本切分为 Pre-Image (系统提示词) 和 Post-Image (用户指令)，用于对照分析。
    """
    def __init__(self, model_path):
        print(f">>> [Collector] Loading model from {model_path} ...")
        self.backend = InternPyTorchBackend(model_path).load()
        self.pipeline = InternVLPipeline(self.backend)
        self.model = self.backend.model
        self.tokenizer = self.backend.tokenizer
        self.device = self.model.device

        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        
        if self.img_context_token_id is None:
            raise ValueError("Tokenizer does not contain <IMG_CONTEXT> token!")
            
        self.raw_outputs = {}
        self.hooks = []

    def _get_hook(self, layer_idx):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.raw_outputs[layer_idx] = hidden.detach().cpu()
        return hook

    def collect_activations(self, media_path, prompt):
        # 1. 加载与预处理
        ext = os.path.splitext(media_path)[-1].lower()
        media_type = 'video' if ext in ['.mp4', '.avi', '.mov', '.mkv'] else 'image'
        print(f">>> [Collector] Processing {media_type}: {media_path}")
        
        media_list = MediaLoader.load(media_path, media_type=media_type)
        if not media_list: raise ValueError("Media load failed")
        
        pixel_values, num_patches_list = self.pipeline.preprocess(media_list)
        
        # 2. 构造 Prompt
        if not hasattr(self.model, 'conv_template'):
            template = copy.deepcopy(self.model.conv_template)
        else:
            template = copy.deepcopy(self.model.conv_template)
            
        if '<image>' not in prompt:
            if len(num_patches_list) > 1:
                prefix = "".join([f"Frame {i+1}: <image>\n" for i in range(len(num_patches_list))])
                prompt = prefix + prompt
            else:
                prompt = '<image>\n' + prompt

        template.system_message = self.model.system_message
        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        # 3. 展开 <image>
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        # 4. Tokenize & Mask Generation
        model_inputs = self.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        
        input_ids_cpu = input_ids[0].cpu()
        seq_len = input_ids_cpu.shape[0]
        indices = torch.arange(seq_len)

        # A. 视觉掩码
        visual_mask = (input_ids_cpu == self.img_context_token_id)
        
        # 找到视觉区域的边界
        if visual_mask.sum() > 0:
            vis_indices = torch.where(visual_mask)[0]
            vis_start = vis_indices.min().item()
            vis_end = vis_indices.max().item()
        else:
            vis_start = seq_len
            vis_end = seq_len

        # B. 噪音掩码
        noise_ids = set(self.tokenizer.all_special_ids)
        extra_tokens = ['<img>', '</img>', '<|im_start|>', '<|im_end|>', '\n']
        for t in extra_tokens:
            tid = self.tokenizer.convert_tokens_to_ids(t)
            if isinstance(tid, int): noise_ids.add(tid)
        
        noise_mask = torch.zeros_like(visual_mask, dtype=torch.bool)
        for nid in noise_ids:
            noise_mask |= (input_ids_cpu == nid)

        # C. 文本切分 (对照实验核心)
        # Pre-Text: 视觉区域之前，且不是噪音
        text_pre_mask = (indices < vis_start) & (~noise_mask) & (~visual_mask)
        
        # Post-Text: 视觉区域之后，且不是噪音
        text_post_mask = (indices > vis_end) & (~noise_mask) & (~visual_mask)
        
        print(f">>> [Collector] Token Split Stats:")
        print(f"    - Visual Tokens : {visual_mask.sum().item()}")
        print(f"    - Pre-Text (Sys): {text_pre_mask.sum().item()} (System Prompt / Template)")
        print(f"    - Post-Text(Usr): {text_post_mask.sum().item()} (User Instruction)")
        
        # 5. 推理
        B = pixel_values.shape[0]
        image_flags = torch.ones((B, 1), dtype=torch.long, device=self.device)

        self.raw_outputs = {}
        self.hooks = []
        
        layers = self.model.language_model.model.layers
        for i, layer in enumerate(layers):
            self.hooks.append(layer.register_forward_hook(self._get_hook(i)))

        with torch.no_grad():
            self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags
            )
            
        for h in self.hooks: h.remove()
        self.hooks = []

        # 6. 数据提取
        cleaned_data = {}
        for layer_idx, hidden in self.raw_outputs.items():
            seq_feat = hidden[0]
            
            if seq_feat.shape[0] != visual_mask.shape[0]: continue
                
            # 分别提取三部分
            vis_tokens = seq_feat[visual_mask]
            pre_tokens = seq_feat[text_pre_mask]
            post_tokens = seq_feat[text_post_mask]
            
            if vis_tokens.numel() == 0: continue
            
            cleaned_data[layer_idx] = {
                "visual": vis_tokens,
                "text_pre": pre_tokens,
                "text_post": post_tokens
            }
            
        return cleaned_data