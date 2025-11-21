import torch
from transformers import AutoModel, AutoTokenizer
from ..base import BaseEngine

class PyTorchBackend(BaseEngine):
    

    def load(self):
        print(f">>> [Backend: PyTorch] Loading {self.model_path}...")
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().cuda()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        print(f">>> [Backend: PyTorch] Loading finish!!!")
        return self

    def generate(self, prompt, pixel_values, eos_token_id=None, **kwargs):
        # 纯粹的 forward 调用，不包含任何预处理逻辑
        generation_config = dict(
            max_new_tokens=kwargs.get('max_new_tokens', 1024),
            do_sample=False,
            eos_token_id=eos_token_id or self.tokenizer.eos_token_id
        )
        
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt, 
            generation_config,
            **kwargs # 传递 num_patches_list 等额外参数
        )
        return response