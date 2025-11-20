import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import os

# 引入我们之前定义的基类和工具
from src.pipelines.base_pipeline import BasePipeline
from src.utils.video import VideoProcessor


class InternVLPipeline(BasePipeline):
    # InternVL 官方推荐的 ImageNet 归一化参数
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)
        print(f">>> [InternVL Pipeline] Loading model from: {model_path}")

        # 1. 显存清理
        torch.cuda.empty_cache()

        # 2. 加载模型
        # Jetson Orin 推荐使用 bfloat16，性能好且不溢出
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda"  # 自动分配到 GPU
        ).eval()

        # 3. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # 4. 初始化视频处理器
        self.video_processor = VideoProcessor(max_frames=8)

        # 5. 默认生成参数
        self.generation_config = dict(max_new_tokens=1024, do_sample=False)

    def _build_transform(self, input_size):
        """构建图像预处理变换"""
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """InternVL 核心逻辑：寻找最佳切片比例"""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """
        InternVL 核心逻辑：动态分辨率切片。
        将一张大图切成多个 448x448 的小块。
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) > 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def _process_image_list(self, images, input_size=448, max_num=12):
        """
        处理图片列表，转换为 Tensor。
        Args:
            images: List[PIL.Image]
            max_num: 每张图片允许的最大切片数 (视频帧建议设为1以省显存)
        """
        transform = self._build_transform(input_size=input_size)
        pixel_values_list = []
        num_patches_list = []

        for img in images:
            # 动态切片
            processed_imgs = self._dynamic_preprocess(
                img,
                image_size=input_size,
                use_thumbnail=True,
                max_num=max_num
            )
            # 转换为 Tensor 并堆叠
            pixel_values = torch.stack([transform(tile) for tile in processed_imgs])
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])

        # 拼接所有图片的 Tensor
        pixel_values = torch.cat(pixel_values_list).to(torch.bfloat16).cuda()
        return pixel_values, num_patches_list

    def chat(self, prompt, media_path=None, media_type='text', history=None, **kwargs):
        """
        统一推理接口
        """
        pixel_values = None
        num_patches_list = None
        final_prompt = prompt

        # ============================
        # 1. 媒体数据预处理
        # ============================
        images = []

        if media_type == 'text':
            # 纯文本不需要处理 pixel_values
            pass

        elif media_type == 'image':
            # 单图
            if not os.path.exists(media_path): raise FileNotFoundError(media_path)
            images = [Image.open(media_path).convert('RGB')]
            pixel_values, num_patches_list = self._process_image_list(images, max_num=kwargs.get('max_num', 12))

        elif media_type == 'multi-image':
            # 多图 (media_path 应该是一个 list)
            if not isinstance(media_path, list): raise ValueError("media_path must be a list for multi-image")
            images = [Image.open(p).convert('RGB') for p in media_path]
            pixel_values, num_patches_list = self._process_image_list(images, max_num=kwargs.get('max_num', 12))

        elif media_type == 'video':
            # 视频 (使用 VideoProcessor)
            num_frames = kwargs.get('num_frames', 8)
            print(f">>> Extracting {num_frames} frames from video...")
            images = self.video_processor.extract_frames(media_path, num_frames=num_frames)

            # 【关键优化】视频帧处理时，强制 max_num=1
            # 否则 8帧 x 12切片 = 96 个 patch，Jetson 16G 必爆
            pixel_values, num_patches_list = self._process_image_list(images, max_num=1)

            # 自动添加视频前缀 prompt
            video_prefix = self.video_processor.create_video_prompt(len(images))
            final_prompt = video_prefix + prompt

        else:
            raise ValueError(f"Unsupported media_type: {media_type}")

        # ============================
        # 2. 调用模型推理
        # ============================
        print(">>> Starting Inference...")

        response, history = self.model.chat(
            self.tokenizer,
            pixel_values,
            final_prompt,
            self.generation_config,
            num_patches_list=num_patches_list,
            history=history,
            return_history=True
        )

        return response