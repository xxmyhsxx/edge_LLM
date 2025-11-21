import torch
from threading import Thread
from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer 
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

    def generate(self, prompt, pixel_values, num_patches_list=None, eos_token_id=None, stream=False, **kwargs):
        generation_config = dict(
            max_new_tokens=kwargs.get('max_new_tokens', 1024),
            do_sample=False,
            eos_token_id=eos_token_id or self.tokenizer.eos_token_id
        )

        
        if stream:
            
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
            # 注意：InternVL 的 chat 底层调用 generate，通常会透传 generation_config
            generation_config['streamer'] = streamer

            # 3. 准备参数
            chat_kwargs = dict(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=generation_config,
                num_patches_list=num_patches_list, # 确保传入这个参数
                **kwargs 
            )

            # 4. 在新线程中启动 model.chat (因为它是阻塞的)
            thread = Thread(target=self.model.chat, kwargs=chat_kwargs)
            thread.start()
            # 5. 返回 streamer (生成器)
            return streamer

        else:
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt, 
                generation_config,
                num_patches_list=num_patches_list, # 补充缺失的参数传递
                **kwargs 
            )
            return response