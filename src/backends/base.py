from abc import ABC, abstractmethod

class BaseEngine(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def load(self, model_path):
        pass

    @abstractmethod
    def generate(self, prompt, image_path=None, max_tokens=50):
        pass