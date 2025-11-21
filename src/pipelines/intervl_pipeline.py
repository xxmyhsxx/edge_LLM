import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from src.pipelines.base_pipeline import BasePipeline
from src.utils.media_loader import MediaLoader
from src.utils.metrics import PerformanceMonitor, TPSCalculator

class InternVLPipeline(BasePipeline):
    def __init__(self, backend):
        super().__init__(backend)
        # 确保后端已加载
        if self.backend.model is None:
            self.backend.load()
            
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.monitor = PerformanceMonitor()

    def _build_transform(self, input_size):
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """找到最接近目标宽高比的比例。"""
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
        """根据图像的宽高比动态地将其分割成多个图块。"""
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


    def preprocess(self, media_list):
        """
        将 PIL Image 列表转换为 Tensor (pixel_values)
        """
        input_size = 448
        transform = self._build_transform(input_size)
        pixel_values_list = []
        num_patches_list = []
        
        for img in media_list:
            max_num = 1 if len(media_list) > 1 else 12
            patches = self._dynamic_preprocess(img, max_num=max_num)
            pixels = torch.stack([transform(p) for p in patches])
            pixel_values_list.append(pixels)
            num_patches_list.append(pixels.shape[0])
            
        pixel_values = torch.cat(pixel_values_list).to(torch.bfloat16).cuda()
        return pixel_values, num_patches_list
    
    def run(self, prompt, media_path=None, media_type='text',stream=False, **kwargs):
        print("Inference"+"-----"*12)
        
        # 1. 加载媒体
        media_list = []
        if media_path:
            media_list = MediaLoader.load(media_path, media_type, kwargs.get('frames', 8))

        # 2. 预处理数据 (Pipeline 的核心职责)
        pixel_values = None
        num_patches_list = []
        
        if media_list:
            pixel_values, num_patches_list = self.preprocess(media_list)
            
            # 构造 Prompt
            prefix = "".join([f"视频帧<{i+1}>: <image>\n" for i in range(len(media_list))])
            prompt = prefix + prompt
        
        if stream:
            # 调用后端获取生成器
            streamer = self.backend.generate(
                prompt, 
                pixel_values, 
                num_patches_list=num_patches_list,
                stream=True,
                
            )
            
            # 定义生成器函数，逐步 yield 内容
            # 流式模式下，暂时无法简单计算 TPS 和 Peak Memory，直接返回纯文本流
            def generator():
                for new_text in streamer:
                    yield new_text
            
            return generator()

        
        else:
            with self.monitor.track():
                response = self.backend.generate(
                    prompt, 
                    pixel_values,
                    num_patches_list=num_patches_list,
                    stream=False,
                    
                )
            tps = TPSCalculator.calculate(response, self.monitor.latency)
            return {
                "text": response,
                "stats": {
                    **self.monitor.get_report(), # 包含 latency 和 peak_memory
                    "tps": f"{tps:.2f} tok/s"
                }
            }
        
        
        
        
    
    

    
    
if __name__ =="__main__":
    # 测试
    from src.backends.torch_backend.internvl import PyTorchBackend
    model = PyTorchBackend("/app/models/InternVL3_5-4B-Instruct")
    internvl = InternVLPipeline(backend=model)
    kwargs = {"frames":8}
    answer = internvl.run("请详细描述视频内容：","/app/eslm/test/Test/Video/001.mp4","video",stream=True,**kwargs)
    print("Bot: ", end="", flush=True)
    for chunk in answer:
        print(chunk, end="", flush=True)
    print("\n\n>>> 结束")
    
