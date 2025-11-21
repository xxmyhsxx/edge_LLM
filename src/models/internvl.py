import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from .base import BaseModelWrapper


class InternVLAdapter(BaseModelWrapper):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model_path):
        super().__init__(model_path)
        self.tokenizer = None
        self.model = None

    def load_model(self, device='cuda'):
        print(f">>> [Adapter] Loading InternVL from {self.model_path}")
        
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        self.model.to(device)
        return self.model, self.tokenizer

    # --- 以下是你原本代码中的预处理逻辑，封装在此 ---
    def _build_transform(self, input_size):
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        # ... (把你的 _dynamic_preprocess 代码完整粘贴到这里) ...
        # 为节省篇幅，此处省略具体实现，逻辑与你提供的一致
        pass

    def preprocess(self, media_list, input_size=448, max_num=12):
        """
        标准接口：输入 PIL List -> 输出 pixel_values, num_patches_list
        """
        transform = self._build_transform(input_size)
        pixel_values_list = []
        num_patches_list = []
        
        for img in media_list:
            # 如果是视频帧，建议 max_num 强制为 1 以省显存
            current_max_num = 1 if len(media_list) > 1 else max_num
            
            processed_imgs = self._dynamic_preprocess(
                img, image_size=input_size, use_thumbnail=True, max_num=current_max_num
            )
            pixel_values = torch.stack([transform(tile) for tile in processed_imgs])
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            
        pixel_values = torch.cat(pixel_values_list).to(torch.bfloat16).cuda()
        return {"pixel_values": pixel_values, "num_patches_list": num_patches_list}

    def get_prompt_template(self, prompt, media_type, num_frames=1):
        if media_type == 'video':
            prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(num_frames)])
            return prefix + prompt
        elif media_type == 'image':
            return "<image>\n" + prompt
        return prompt