from abc import ABC, abstractmethod

class BaseModelWrapper(ABC):
    def __init__(self, model_path):
        self.model_path = model_path

    @abstractmethod
    def get_layers(self):
        """返回模型的所有 Transformer 层列表"""
        pass

    @abstractmethod
    def get_attention(self, layer):
        """给定某一层，返回它的 Attention 模块"""
        pass

    @abstractmethod
    def get_mlp(self, layer):
        """给定某一层，返回它的 MLP 模块"""
        pass

    @abstractmethod
    def preprocess(self, media_list):
        """
        将 List[PIL.Image] 转换为该模型需要的 Tensor (pixel_values)
        同时返回其他元数据 (如 num_patches_list)
        """
        pass
    
    @abstractmethod
    def get_prompt_template(self, prompt, media_type, num_frames=1):
        """处理 Prompt 格式 (加 <image> 标签等)"""
        pass