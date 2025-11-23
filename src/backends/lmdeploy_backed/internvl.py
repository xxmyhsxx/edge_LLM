import time
from src.backends.base import BaseEngine

try:
    from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
    from lmdeploy.vl.utils import encode_image_base64
except ImportError:
    pipeline = None
    encode_image_base64 = None


import numpy as np
from lmdeploy import pipeline, GenerationConfig
from decord import VideoReader, cpu
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64
from PIL import Image




def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    imgs = []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        imgs.append(img)
    return imgs


def load_prompts(prompt,imgs,num_segments=8):
    question = ''
    for i in range(len(imgs)):
        question = question + f'Frame{i+1}: {IMAGE_TOKEN}\n'

    question += prompt
    content = [{'type': 'text', 'text': question}]
    for img in imgs:
        content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': 1, 'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}})
    messages = [dict(role='user', content=content)]
    return messages


class InternLMDeployBackend(BaseEngine):
    def load(self):
        if pipeline is None:
            raise ImportError("Please install lmdeploy first: pip install lmdeploy")

        print(f">>> [Backend: LMDeploy] Loading {self.model_path}...")

        cache_max_entry_count = self.config.get('cache_max_entry_count', 0.2) 
        session_len = self.config.get('session_len', 8192) 
        max_prefill_token_num = self.config.get('max_prefill_token_num', 4096)
        
        print(f"    -> Cache Ratio: {cache_max_entry_count}")
        print(f"    -> Session Len: {session_len}")

        backend_config = PytorchEngineConfig(
            cache_max_entry_count=cache_max_entry_count,
            session_len=session_len,
            max_prefill_token_num=max_prefill_token_num,
            dtype='float16',
            enable_prefix_caching=False
        )
        
        start_load = time.perf_counter()
        self.pipe = pipeline(self.model_path, backend_config=backend_config)
        end_load = time.perf_counter()
        print(f">>> [Backend: LMDeploy] ✅ Model loaded! Time: {end_load - start_load:.2f}s")
        
        return self

    def generate(self, prompt, media_list=None, stream=False, **kwargs):
        """
        使用官方推荐的 Messages 格式进行推理
        """
        if not hasattr(self, 'pipe'):
            self.load()
        if media_list:
            num_segments = kwargs.get('frames', 8)
            messages = load_prompts(prompt, media_list, num_segments=num_segments)
        else:
            messages = [dict(role='user', content=prompt)]

        # 2. 配置生成参数
        gen_config = GenerationConfig(
            max_new_tokens=kwargs.get('max_new_tokens', 1024),
            top_p=0.0,
            temperature=kwargs.get('temperature', 0.0),
            do_sample=kwargs.get('do_sample', False)
        )

        # 3. 推理
        if stream:
            return self._stream_wrapper(messages, gen_config)
        else:
            # pipe 返回的是 Response 列表
            response = self.pipe(messages, gen_config=gen_config)
            return response.text

    def _stream_wrapper(self, messages, gen_config):
        """流式输出包装器"""
        for output in self.pipe.stream_infer(messages, gen_config=gen_config):
            if hasattr(output, 'text'):
                yield output.text 
            elif hasattr(output, 'content'):
                yield output.content
