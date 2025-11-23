import sys
import os
import time
# 确保能导入 src 目录
sys.path.append(os.getcwd())

from src.backends.llamacpp_backend.internvl import InternLlamaCppBackend

from src.pipelines.intervl_pipeline import InternVLPipeline # 假设文件名叫 pipeline.py

if __name__ == "__main__":

    SERVER_URL = "http://localhost:8089/v1"
    
    print(f"Initializing LlamaCpp Backend connecting to {SERVER_URL}...")
    backend = InternLlamaCppBackend(base_url=SERVER_URL)
    # 2. 初始化 Pipeline
    internvl = InternVLPipeline(backend=backend)
    
    # 3. 准备测试参数
    video_path = "/app/eslm/test/Test/Video/006.mp4" 
    kwargs = {"frames": 4} 
    prompt = "请问有没有发生自然灾害，是什么灾害，请回答是什么灾害(20字以内):"
    start_time = time.time()

    try:
        result = internvl.run(
            prompt=prompt,
            media_path=video_path,
            media_type="video",
            stream=False,
            **kwargs
        )
        
        # Pipeline 的 run 方法在非流式下返回字典 {"text": ..., "stats": ...}
        print(f"Bot Reply: {result['text']}")
        print(f"Stats: {result['stats']}")
        
    except Exception as e:
        print(f"\n非流式测试出错: {e}")

    # --- 测试 A: 流式输出 (Streaming) ---
    try:
        answer_generator = internvl.run(
            prompt=prompt,
            media_path=video_path,
            media_type="video",
            stream=True,
            **kwargs
        )
        
        print("Bot: ", end="", flush=True)
        for chunk in answer_generator:
            print(chunk, end="", flush=True)
        print("\n")
        
    except Exception as e:
        print(f"\n流式测试出错: {e}")
    print(time.time()-start_time)

    # --- 测试 B: 非流式输出 (Non-Streaming) ---
    
