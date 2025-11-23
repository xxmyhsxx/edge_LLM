from src.backends.lmdeploy_backed.internvl import InternLMDeployBackend
from src.backends.llamacpp_backend.internvl import InternLlamaCppBackend
from src.backends.torch_backend.internvl import InternPyTorchBackend
from src.pipelines.intervl_pipeline import InternVLPipeline
import time
import gc

if __name__ =="__main__":
    kwargs = {"frames":4}

    model = InternPyTorchBackend("/app/models/InternVL3_5-4B-Instruct") 
    internvl = InternVLPipeline(backend=model)
    start_time = time.time()
    answer = internvl.run("请详细描述视频内容：","/app/eslm/test/Test/Video/001.mp4","video",stream=False,**kwargs)
    print(time.time()-start_time)
    print(answer)
    del model,internvl
    gc.collect()

    print("-----"*12)
    model = InternLMDeployBackend("/app/models/InternVL3_5-4B-Instruct") 
    internvl = InternVLPipeline(backend=model)
    print(time.sleep(5))
    start_time = time.time()
    answer = internvl.run("请详细描述视频内容：","/app/eslm/test/Test/Video/001.mp4","video",stream=False,**kwargs)
    print(time.time()-start_time)
    print(answer)



    # model = InternLlamaCppBackend() 
    # internvl = InternVLPipeline(backend=model)
    # time.sleep(5)
    # start_time = time.time()
    # answer = internvl.run("请详细描述视频内容：","/app/eslm/test/Test/Video/001.mp4","video",stream=False,**kwargs)
    # print(time.time()-start_time)
    # print(answer)