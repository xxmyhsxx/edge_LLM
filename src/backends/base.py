from abc import ABC, abstractmethod

class BaseEngine(ABC):
    def __init__(self, model_path,config=None):
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        

    @abstractmethod
    def load(self, model_path):
        pass

    @abstractmethod
    def generate(self, prompt, image_path=None, max_tokens=50):
        pass