import torch
import copy
import os
import sys
import gc
import torch.nn.functional as F
# 自动添加项目根目录到 path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.backends.torch_backend.internvl import InternPyTorchBackend
from src.pipelines.intervl_pipeline import InternVLPipeline
from src.utils.media_loader import MediaLoader
from src.methods.quantization.quant_tensor import quantize_weight_per_channel_absmax




def data(raw_data):
    visual_dict = {k: v['visual'] for k, v in raw_data.items()}
    sys_text_dict = {k: v['text_pre'] for k, v in raw_data.items()}
    user_text_dict = {k: v['text_post'] for k, v in raw_data.items()}
    
    # 还可以组合：所有文本
    all_text_dict = {}
    for k, v in raw_data.items():
        if v['text_pre'].numel() > 0 and v['text_post'].numel() > 0:
            all_text_dict[k] = torch.cat([v['text_pre'], v['text_post']], dim=0)
        elif v['text_post'].numel() > 0:
            all_text_dict[k] = v['text_post']

    vis_half1_dict = {}
    vis_half2_dict = {}
    
    for k, v in visual_dict.items():
        # v shape: [N_vis, Dim]
        split_point = v.shape[0] // 2
        if split_point > 0:
            vis_half1_dict[k] = v[:split_point]
            vis_half2_dict[k] = v[split_point:]
        else:
            # 如果太短没法切，就复制一份 (仅防报错)
            vis_half1_dict[k] = v
            vis_half2_dict[k] = v
    return visual_dict,vis_half1_dict,vis_half2_dict,all_text_dict,sys_text_dict,user_text_dict


class InternVLCollector:
    """
    【数据采集器 - 父类】(修复设备不匹配版)
    职责：负责模型加载、Prompt处理、Mask生成和基础推理。
    """
    def __init__(self, model_path, quant_policy=None):
        print(f">>> [Collector] Loading model from {model_path} ...")
        self.backend = InternPyTorchBackend(model_path).load()
        self.pipeline = InternVLPipeline(self.backend)
        self.model = self.backend.model
        self.tokenizer = self.backend.tokenizer
        self.device = self.model.device

        # 初始化 Token ID
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        
        if self.img_context_token_id is None:
            raise ValueError("Tokenizer does not contain <IMG_CONTEXT> token!")
        

        self.layer_backups = {}
        self.raw_outputs = {}
        self.hooks = []
            
        # 初始化量化 (如果需要)
        if quant_policy is not None:
            self._apply_quantization(quant_policy)
            
        self.clean_memory()

    def _apply_quantization(self, policy):
        """对模型应用量化策略"""
        print(f">>> [Collector] Applying Initial Quantization Policy...")
        if hasattr(self.model, 'language_model'):
            layers = self.model.language_model.model.layers
        elif hasattr(self.model, 'model'):
            layers = self.model.model.layers
        else:
            layers = self.model.layers
            
        for i, layer in enumerate(layers):
            if isinstance(policy, int): bits = policy
            elif isinstance(policy, dict): bits = policy.get(i, policy.get(str(i), 4))
            else: bits = 4
            
            if bits >= 16: continue
            # print(bits)
            
            target_modules = self._get_layer_modules(layer)
            for m in target_modules:
                m.weight.data = quantize_weight_per_channel_absmax(m.weight.data, n_bits=bits, zero_point=True)

    def _get_layer_modules(self, layer):
        """获取一层中的 Linear 模块"""
        modules = []
        if hasattr(layer, 'self_attn'):
            modules.extend([layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.self_attn.o_proj])
        if hasattr(layer, 'mlp'):
            modules.extend([layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj])
        return modules

    def _get_hook(self, layer_idx):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.raw_outputs[layer_idx] = hidden.detach().cpu()
        return hook

    def prepare_inputs(self, media_path, prompt):
        """
        只负责准备输入数据和 Mask，不运行推理。
        """
        # 1. 加载数据
        ext = os.path.splitext(media_path)[-1].lower()
        media_type = 'video' if ext in ['.mp4', '.avi', '.mov', '.mkv'] else 'image'
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

        IMG_START, IMG_END, IMG_CTX = '<img>', '</img>', '<IMG_CONTEXT>'
        for num_patches in num_patches_list:
            image_tokens = IMG_START + IMG_CTX * self.model.num_image_token * num_patches + IMG_END
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = self.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        
        # 3. Mask 生成
        input_ids_cpu = input_ids[0].cpu()
        visual_mask = (input_ids_cpu == self.img_context_token_id)
        
        noise_ids = set(self.tokenizer.all_special_ids)
        for t in ['<img>', '</img>', '<|im_start|>', '<|im_end|>', '\n']:
            tid = self.tokenizer.convert_tokens_to_ids(t)
            if isinstance(tid, int): noise_ids.add(tid)
        noise_mask = torch.zeros_like(visual_mask, dtype=torch.bool)
        for nid in noise_ids: noise_mask |= (input_ids_cpu == nid)
        
        # 文本切分
        indices = torch.arange(len(input_ids_cpu))
        vis_indices = torch.where(visual_mask)[0]
        if len(vis_indices) > 0:
            vis_start, vis_end = vis_indices.min(), vis_indices.max()
        else:
            vis_start, vis_end = 0, 0
            
        text_pre_mask = (indices < vis_start) & (~noise_mask) & (~visual_mask)
        text_post_mask = (indices > vis_end) & (~noise_mask) & (~visual_mask)
        
        B = pixel_values.shape[0]
        image_flags = torch.ones((B, 1), dtype=torch.long, device=self.device)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_flags": image_flags,
            "masks": {
                "visual": visual_mask.to(self.device),
                "text_pre": text_pre_mask.to(self.device),
                "text_post": text_post_mask.to(self.device)
            }
        }

    def split_output(self, hidden_state, masks):
        """
        【辅助工具】根据 Mask 切分输出
        关键修复：将 Mask 移动到与 hidden_state 相同的设备 (CPU)
        """
        seq = hidden_state[0] # [Seq, Dim] (通常在 CPU)
        target_device = seq.device
        
        return {
            "visual": seq[masks['visual'].to(target_device)].detach().cpu(),
            "text_pre": seq[masks['text_pre'].to(target_device)].detach().cpu(),
            "text_post": seq[masks['text_post'].to(target_device)].detach().cpu()
        }

    def collect_activations(self, media_path, prompt):
        """父类默认行为：收集所有层的输出"""
        print(f">>> [Collector] Processing {media_path}...")
        inputs = self.prepare_inputs(media_path, prompt)
        
        self.raw_outputs = {}
        self.hooks = []
        
        # 适配 layers
        if hasattr(self.model, 'language_model'):
            layers = self.model.language_model.model.layers
        elif hasattr(self.model, 'model'):
            layers = self.model.model.layers
        else:
            layers = self.model.layers

        for i, layer in enumerate(layers):
            self.hooks.append(layer.register_forward_hook(self._get_hook(i)))

        with torch.no_grad():
            self.model(
                pixel_values=inputs['pixel_values'],
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                image_flags=inputs['image_flags']
            )
            
        for h in self.hooks: h.remove()
        self.hooks = []

        cleaned_data = {}
        for layer_idx, hidden in self.raw_outputs.items():
            if hidden[0].shape[0] != inputs['masks']['visual'].shape[0]: continue
            cleaned_data[layer_idx] = self.split_output(hidden, inputs['masks'])
            
        return cleaned_data
    
    @property
    def layers(self):
        """
        [新增] 动态获取层列表，不占用 self.layers 空间
        """
        if hasattr(self.model, 'language_model'):
            return self.model.language_model.model.layers
        elif hasattr(self.model, 'model'):
            return self.model.model.layers
        return self.model.layers

    def clean_memory(self):
        """[新增] 强制清理显存"""
        self.raw_outputs = {}
        gc.collect()
        torch.cuda.empty_cache()

    def backup_layer(self, layer_idx):
        """[新增] 备份某层的原始权重到 CPU"""
        if layer_idx in self.layer_backups: return
        
        layer = self.layers[layer_idx]
        # 直接存成 list，避免修改 _get_layer_modules 的返回结构
        backup_tensors = []
        for module in self._get_layer_modules(layer):
            backup_tensors.append(module.weight.data.clone().cpu())
            
        self.layer_backups[layer_idx] = backup_tensors

    def restore_layer(self, layer_idx):
        """[新增] 恢复权重 (原地 copy_，无显存峰值)"""
        if layer_idx not in self.layer_backups: return
        
        layer = self.layers[layer_idx]
        backup_tensors = self.layer_backups[layer_idx]
        modules = self._get_layer_modules(layer)
        
        # 对应恢复
        for module, backup_w in zip(modules, backup_tensors):
            module.weight.data.copy_(backup_w)
        del self.layer_backups[layer_idx]
        gc.collect()

    def quantize_specific_layer(self, layer_idx, bits=4, backup=True):
        """[新增] 量化指定单层"""
        if bits >= 16: return
        if backup: self.backup_layer(layer_idx)
        
        layer = self.layers[layer_idx]
        for module in self._get_layer_modules(layer):
            with torch.no_grad():
                w = module.weight.data
                qw = quantize_weight_per_channel_absmax(w, n_bits=bits, zero_point=True)
                module.weight.data = qw
                del w, qw # 显式释放

    def run_layer_sensitivity_analysis(self, media_path, prompt, test_bits=4):
        """
        自动化敏感度分析 
        """

        print(f"\n>>> [Analysis] Starting Sensitivity Analysis (W{test_bits})...")
        self.clean_memory()
        
        # 1. 预加载输入 (避免循环重复申请显存)
        print("  - Pre-loading inputs to GPU...")
        inputs = self.prepare_inputs(media_path, prompt)
        
        # 2. 跑基准
        print("  - Running Baseline...")
        baseline_outputs = self.collect_activations(media_path,prompt)
        visall,vis1,vis2,txtall,txtsys,txtuser = data(baseline_outputs)



        # last_layer_idx = max(baseline_outputs.keys())
        
        # # 提取 Visual 特征作为对比基准
        # baseline_final = baseline_outputs[last_layer_idx].get('visual')
        # if baseline_final is None or baseline_final.numel() == 0:
        #     baseline_final = baseline_outputs[last_layer_idx].get('text_post')
        
        # # 移至 CPU 并清理
        # baseline_final = baseline_final.clone().detach()
        del baseline_outputs
        
        results = []
        num_layers = len(self.layers)
        from src.methods.research.analyzer import CosineGapMetric,SNRMetric,L2RelativeErrorMetric,WassersteinMetric,DistributionAnalyzer


        # 3. 循环测试
        for i in range(num_layers):

            quant_metrics = [
            SNRMetric(),             # 必须：看精度 (dB)
            WassersteinMetric(),     # 必须：看有没有发生 Outlier 截断
            L2RelativeErrorMetric(), # 辅助：看整体误差百分比
            CosineGapMetric()        # 辅助：看语义方向
            ]

            analyzer = DistributionAnalyzer(output_dir=f'/app/Edge-LMM-Optimizer/experiments/results/layers/{i+1}',metrics=quant_metrics)
            print(f"  - Testing Layer {i}/{num_layers}...", end="\r")
            
            try:
                # 量化
                self.quantize_specific_layer(i, bits=test_bits, backup=True)
                
                # 推理 (复用 inputs)
                current_outputs = self.collect_activations(media_path,prompt)

                quantvisall,quantvis1,quantvis2,quanttxtall,quanttxtsys,quanttxtuser = data(current_outputs)

                # analyzer.run_analysis(visall,quantvisall,name_a="vis",name_b="quantvis")
                # analyzer.run_analysis(txtall,quanttxtall,name_a="txt",name_b="quanttxt")
                # analyzer.run_analysis(txtuser,quanttxtuser,name_a="txtuser",name_b="quant")
                results1 = analyzer.run_analysis(visall, quantvisall, name_a="vis", name_b="quantvis")
                results2 = analyzer.run_analysis(txtall, quanttxtall, name_a="txt", name_b="quanttxt")
                results3 = analyzer.run_analysis(txtuser, quanttxtuser, name_a="txtuser", name_b="quant")

                analyzer.plot_multi_analysis(
                        [results1, results2, results3],
                              labels=["VIS", "TEXT", "USER"])

                del current_outputs

            except Exception as e:
                print(f"\n[Error] Layer {i}: {e}")
                # results.append({"layer": i, "mse": -1})

            finally:
                # 恢复 & 清理
                self.restore_layer(i)
                self.clean_memory()

        del inputs
        self.clean_memory()
        print(f"\n>>> [Analysis] Done.")
        return results


    
    

if __name__ == "__main__":
    # 配置你的模型路径 (请修改为实际路径)
    # 示例: /app/models/InternVL3_5-4B-Instruct
    MODEL_PATH = "/app/models/InternVL3_5-4B-Instruct" 
    MEDIA_FILE = "/app/eslm/test/Test/Video/001.mp4"
    PROMPT = "Please describe this image in detail."

    print("=== Testing InternVLCollector Sensitivity Analysis ===")

    if os.path.exists(MODEL_PATH) and os.path.exists(MEDIA_FILE):
        try:
            # 1. 初始化 (默认 FP16/BF16，不预先量化)
            collector = InternVLCollector(MODEL_PATH, quant_policy=None)
            
            # 2. 运行敏感度分析 (测试 Int4 量化对每一层的影响)
            # 这会自动遍历每一层：量化 -> 测MSE -> 回滚
            report = collector.run_layer_sensitivity_analysis(
                media_path=MEDIA_FILE, 
                prompt=PROMPT, 
                test_bits=4
            )
            
            # 3. 打印报表
            print("\n=== Sensitivity Report (W4) ===")
            print(f"{'Layer':<6} | {'MSE Loss':<12}")
            print("-" * 20)
            for item in report:
                mse_val = item['mse']
                mse_str = f"{mse_val:.6f}" if mse_val != float('inf') else "INF"
                print(f"{item['layer']:<6} | {mse_str:<12}")
                
        except Exception as e:
            print(f"[Test Failed] {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Path not found. Please check:\nModel: {MODEL_PATH}\nMedia: {MEDIA_FILE}")