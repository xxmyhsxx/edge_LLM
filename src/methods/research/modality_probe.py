import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import csv
import copy
# 【核心修正】：自动找到 main.py 所在的根目录，并加入 path
# 这样无论你在哪里运行脚本，都能找到 src 包
current_file_path = os.path.abspath(__file__)
# 回退 3 层：src/methods/research/ -> src/methods/ -> src/ -> 根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
if project_root not in sys.path:
    sys.path.append(project_root)

# 引用你现有的代码 (基于上传文件确认的路径)
from src.backends.torch_backend.internvl import InternPyTorchBackend
from src.pipelines.intervl_pipeline import InternVLPipeline
from src.utils.media_loader import MediaLoader

class ModalityProbe:
    def __init__(self, model_path):
        print(f">>> [Probe] Initializing with model: {model_path}")
        # 加载 FP16 模型
        self.backend = InternPyTorchBackend(model_path).load()
        # 复用 Pipeline 的预处理逻辑
        self.pipeline = InternVLPipeline(self.backend)
        self.model = self.backend.model
        self.tokenizer = self.backend.tokenizer
        self.device = self.model.device
        # 存储钩子抓取的数据
        self.layer_outputs = {}
        self.handles = []

    def _get_hook(self, layer_idx):
        def hook(module, inputs, output):
            # 兼容不同模型的输出格式 (tuple vs tensor)
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # 移至 CPU 节省显存
            self.layer_outputs[layer_idx] = hidden.detach().cpu()
        return hook

    def register_hooks(self):
        """给每一层 Transformer 注册 Hook"""
        print(">>> [Probe] Registering hooks...")
        self.layer_outputs = {}
        self.handles = []
        
        # 自动寻找 layers 属性
        print(self.model.language_model.model.layers)
        print(123)
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'language_model') and hasattr(self.model.language_model.model, 'layers'):
            layers = self.model.language_model.model.layers
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            raise ValueError("Could not find '.layers' in model.")

        for i, layer in enumerate(layers):
            handle = layer.register_forward_hook(self._get_hook(i))
            self.handles.append(handle)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def run_probe(self, media_path, prompt, output_dir="experiments/results"):
        """
        运行探针，直接复用 InternVLChatModel 的 Prompt 处理逻辑
        """
        # 1. 加载与预处理
        ext = os.path.splitext(media_path)[-1].lower()
        media_type = 'video' if ext in ['.mp4', '.avi', '.mov', '.mkv'] else 'image'
        print(f">>> [Probe] Processing {media_type}: {media_path}")

        media_list = MediaLoader.load(media_path, media_type=media_type)
        if not media_list:
            print(f"Error: Failed to load media.")
            return

        pixel_values, num_patches_list = self.pipeline.preprocess(media_list)
        
        # 2. 【核心修正】复用模型源码逻辑构造 Prompt (Prompt Expansion)
        # 源码逻辑参考: InternVLChatModel.chat / batch_chat
        
        # A. 设置 img_context_token_id (源码 forward 依赖此属性)
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id # 注入模型实例
        
        # B. 获取并克隆 Template (防止修改原模型状态)
        if not hasattr(self.model, 'conv_template'):
            # 如果模型没加载 template，尝试从 config 或默认加载
            try:
                from src.backends.torch_backend.internvl import get_conv_template # 假设能从某处获取，或者直接用 model 的
                template = get_conv_template(self.model.template)
            except:
                # 兜底：直接深拷贝模型当前的 template 实例
                template = copy.deepcopy(self.model.conv_template)
        else:
            template = copy.deepcopy(self.model.conv_template)

        # C. 构造对话
        if '<image>' not in prompt:
            prompt = '<image>\n' + prompt
            
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        # D. 展开 <image> 标签 (Token Replacement)
        # 直接复用源码中的循环逻辑
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        
        for num_patches in num_patches_list:
            # 复用 self.model.num_image_token
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        print(f">>> [Probe] Expanded Prompt Length: {len(query)} chars")
        print(query)
        
        # 3. Tokenize
        model_inputs = self.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)

        # 4. 【关键修正】构造 image_flags
        # 源码: image_flags = image_flags.squeeze(-1) ... vit_embeds = vit_embeds[image_flags == 1]
        # 我们创建一个 shape=(B, 1) 的全 1 tensor
        B = pixel_values.shape[0]
        image_flags = torch.ones((B, 1), dtype=torch.long, device=self.device)

        # 5. 推理 (Forward)
        self.register_hooks()
        print(">>> [Probe] Running Forward Pass...")
        with torch.no_grad():
            self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags
                # num_patches_list 已移除，不需要
            )
        self.remove_hooks()
        
        # 6. 计算分离度时，我们需要知道有多少个 Visual Token
        # 计算逻辑：所有 patch 数 * 每个 patch 的 token 数
        total_visual_tokens = sum(num_patches_list) * self.model.num_image_token
        
        # 分析并保存
        self._analyze_separation(total_visual_tokens, output_dir)

    def _analyze_separation(self, num_visual_tokens, output_dir):
        print("\n>>> [Analysis] Calculating Modality Separation...")
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "modality_gap_data.csv")
        plot_path = os.path.join(output_dir, "modality_gap_plot.png")

        layers = sorted(self.layer_outputs.keys())
        data_records = []

        print(f"{'Layer':<6} | {'Cos Sim':<10} | {'Gap (1-Sim)':<15}")
        print("-" * 40)
        
        for idx in layers:
            hidden = self.layer_outputs[idx][0] # [Seq, Dim]
            
            # 切分模态:
            # 经过 Prompt Expansion 后，视觉 token 被展平在 input_ids 里
            # 它们的结构通常是: [Bos, <img>, ...VISUAL..., </img>, System..., User...]
            # 我们需要定位 Visual 区域。
            # 简单策略：寻找前 num_visual_tokens 个特征（排除开头的少量特殊 token）
            
            # 为了更精准，我们假设 Visual 占据了绝大部分长度，且在前面
            # 我们跳过前 2 个 token (Bos, <img> start)，取 num_visual_tokens
            # 这只是一个近似，但在 Gap 分析中足够有效
            
            start_idx = 0
            # 简单的启发式搜索：如果第一个 token 是 bos，跳过
            # 但这里我们简化处理，直接取前 N 个，误差极小
            
            if hidden.shape[0] <= num_visual_tokens:
                continue

            # 尝试切分
            img_feats = hidden[:num_visual_tokens]
            txt_feats = hidden[num_visual_tokens:]
            
            if img_feats.numel() == 0 or txt_feats.numel() == 0:
                continue

            img_center = img_feats.mean(dim=0)
            txt_center = txt_feats.mean(dim=0)
            
            cos_sim = F.cosine_similarity(img_center.unsqueeze(0), txt_center.unsqueeze(0)).item()
            gap = 1.0 - cos_sim
            
            data_records.append({
                "layer_idx": idx,
                "cosine_similarity": cos_sim,
                "modality_gap": gap,
                "img_norm": img_center.norm().item(),
                "txt_norm": txt_center.norm().item()
            })
            
            print(f"{idx:<6} | {cos_sim:.4f}     | {gap:.4f}")

        # 保存 CSV
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["layer_idx", "cosine_similarity", "modality_gap", "img_norm", "txt_norm"])
                writer.writeheader()
                writer.writerows(data_records)
            print(f">>> [Data] Saved to: {csv_path}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

        # 绘图
        if data_records:
            layer_indices = [d['layer_idx'] for d in data_records]
            gaps = [d['modality_gap'] for d in data_records]
            
            plt.figure(figsize=(10, 6))
            plt.plot(layer_indices, gaps, marker='o', label='Modality Gap', color='tab:blue')
            plt.xlabel("Layer Index")
            plt.ylabel("Gap (1 - CosSim)")
            plt.title("Modality Separation Across Layers")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(plot_path)
            print(f">>> [Plot] Saved to: {plot_path}")