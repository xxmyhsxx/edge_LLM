from abc import ABC, abstractmethod

class BaseModelWrapper(ABC):
    def __init__(self, model):
        self.model = model

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