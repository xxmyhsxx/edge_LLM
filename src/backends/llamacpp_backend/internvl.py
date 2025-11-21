import os
import base64
from io import BytesIO
from ..base import BaseEngine

# 尝试导入，防止未安装报错
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError:
    Llama = None

class LlamaCppBackend(BaseEngine):
    def load(self):
        if Llama is None:
            raise ImportError("Please install llama-cpp-python with CUDA support first.")

        print(f">>> [Backend: LlamaCpp] Loading GGUF model: {self.model_path}...")
        
        # 1. 读取配置 (从 configs/jetson_orin.yaml 或默认值)
        # n_gpu_layers=-1 表示将所有层加载到 GPU (Jetson 上必须这样才快)
        n_gpu_layers = self.config.get('n_gpu_layers', -1) 
        n_ctx = self.config.get('max_model_len', 2048)
        
        # 2. 多模态支持 (Vision Projector)
        # 如果是多模态模型，通常需要指定 mmproj (clip) 文件的路径
        # 假设 config 中有一个 'mmproj_path' 字段
        chat_handler = None
        mmproj_path = self.config.get('mmproj_path')
        
        if mmproj_path and os.path.exists(mmproj_path):
            print(f"    Loading MMProj: {mmproj_path}")
            # 注意：这里使用 Llava15 处理器作为通用多模态处理器
            # 对于 InternVL GGUF，请确认其兼容性，通常需要专门的 handler 或 clip 模型
            chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
        
        # 3. 初始化 Llama
        self.model = Llama(
            model_path=self.model_path,
            chat_handler=chat_handler,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False  # 设为 True 可以看到底层的 CUDA log
        )
        print(f">>> [Backend: LlamaCpp] Load finished.")
        return self

    def _pil_to_base64(self, image):
        """辅助函数：将 PIL Image 转为 Base64"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def generate(self, prompt, pixel_values=None, num_patches_list=None, media_list=None, stream=False, **kwargs):
        """
        统一生成接口。
        
        Args:
            prompt (str): 文本提示
            pixel_values: (被忽略) PyTorch Tensor
            num_patches_list: (被忽略)
            media_list (list[PIL.Image]): 原始图片列表 【关键数据】
            stream (bool): 是否流式
        """
        if self.model is None:
            self.load()

        # 1. 构建消息 (Messages)
        content = []
        
        # 如果有多媒体数据，先处理图片
        if media_list:
            for img in media_list:
                # llama-cpp-python 接收 image_url 格式的 base64
                b64_img = self._pil_to_base64(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": b64_img}
                })
        
        # 添加文本
        # 注意：InternVL 的 Prompt 可能包含 <image> 占位符，这里简单清洗一下
        # 让 chat_handler 去自动处理图文位置
        clean_prompt = prompt.replace("<image>", "").replace("Image-1:", "").strip()
        content.append({"type": "text", "text": clean_prompt})

        messages = [
            {"role": "user", "content": content}
        ]

        # 2. 提取参数
        max_tokens = kwargs.get('max_new_tokens', 512)
        temperature = kwargs.get('temperature', 0.7)

        # 3. 调用底层推理
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream
        )

        # 4. 处理输出 (流式 vs 普通)
        if stream:
            # 返回一个生成器，适配 Pipeline 的 yield 逻辑
            def generator():
                for chunk in response:
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        yield delta['content']
            return generator()
        else:
            # 直接返回文本
            return response['choices'][0]['message']['content']