from abc import ABC, abstractmethod

class BasePipeline(ABC):
    def __init__(self, backend):
        """
        Dependency Injection: Pipeline 依赖 Backend
        """
        self.backend = backend

    @abstractmethod
    def preprocess(self, media_list):
        """处理图片/视频"""
        pass

    @abstractmethod
    def run(self, prompt, media_path=None, **kwargs):
        """端到端运行"""
        pass