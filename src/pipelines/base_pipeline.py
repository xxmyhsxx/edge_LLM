from abc import ABC, abstractmethod

class BasePipeline(ABC):
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path

    @abstractmethod
    def chat(self, prompt, media_path=None, media_type='text', **kwargs):
        """
        所有 Pipeline 必须实现这个接口
        输入: 提示词, 媒体路径, 媒体类型
        输出: 文本回复
        """
        pass