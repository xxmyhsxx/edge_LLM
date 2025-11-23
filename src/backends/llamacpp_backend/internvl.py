import base64
import io
from openai import OpenAI

class InternLlamaCppBackend:
    def __init__(self, base_url="http://localhost:8089/v1", api_key="sk-no-key-required"):
        """
        初始化 LlamaCpp 后端
        :param base_url: llama-server 的地址
        :param api_key: 占位符 key，server 模式通常忽略
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        # model 属性用于 pipeline 检查是否已加载，随便设一个非 None 值
        self.model = "remote-model" 
        self.base_url = base_url

    def load(self):
        """
        Server 模式下不需要本地加载权重，这里仅做连接测试
        """
        print(f">>> [LlamaCppBackend] Connecting to server at {self.base_url}...")
        try:
            # 尝试列出模型以测试连接
            self.client.models.list()
            print(">>> [LlamaCppBackend] Connection established successfully.")
        except Exception as e:
            print(f">>> [LlamaCppBackend] Warning: Connection failed. Ensure llama-server is running. Error: {e}")

    def _image_to_base64(self, image):
        """将 PIL Image 转换为 Base64 字符串"""
        buffered = io.BytesIO()
        # 统一转为 JPEG 以减少体积，保持 RGB 模式
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate(self, prompt, media_list=None, stream=False, **kwargs):
        """
        生成回复
        :param prompt: 文本提示词
        :param media_list: 图片列表 (PIL Image List)
        :param stream: 是否流式输出
        """
        
        # 1. 构建消息内容
        content = []
        
        # 处理文本部分
        if prompt:
            content.append({"type": "text", "text": prompt})

        # 处理图像/视频帧部分
        if media_list:
            for i, img in enumerate(media_list):
                b64_str = self._image_to_base64(img)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_str}"
                    }
                })

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        # 2. 发送请求
        try:
            response = self.client.chat.completions.create(
                model="default-model", # Server 启动时已指定模型，此处名称不重要
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.0),
                stream=stream
            )

            # 3. 处理返回结果
            if stream:
                # 返回生成器
                return self._stream_generator(response)
            else:
                # 返回完整字符串
                return response.choices[0].message.content

        except Exception as e:
            return f"Error during generation: {str(e)}"

    def _stream_generator(self, response_stream):
        """处理流式响应的辅助生成器"""
        for chunk in response_stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content