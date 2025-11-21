### Edge-LMM-Optimizer ⚡

Edge-LMM-Optimizer 是一个专为边缘计算设备（特别是 NVIDIA Jetson Orin NX 16GB）设计的多模态大模型（LMM）推理优化与实验框架。

本项目旨在解决大参数模型（如 InterVL-8B）在显存受限设备上的部署难题。通过 后端 (Backends)、业务流水线 (Pipelines) 和 优化算法 (Methods) 的解耦设计，它既能作为高效的推理引擎，也能作为科研算法（如剪枝、量化）的验证平台。

#### 🌟 核心特性 (Key Features)

极致显存适配: 针对 Jetson 16GB 统一内存架构深度优化，默认集成防 OOM 策略（如动态分辨率限制、Eager Mode）。

##### 多后端支持:

PyTorch: 适合科研与算法验证，支持 Hook 模型层、插入剪枝 Mask。

vLLM: (开发中) 适合高性能生产部署，支持 AWQ/GPTQ 量化。

llama.cpp: (开发中) 支持 GGUF 格式，进一步降低显存需求。

流式推理 (Streaming): 支持像 ChatGPT 一样的打字机效果，提升边缘端用户体验。

算法验证套件: 内置 Attention 热力图分析、Token 剪枝可视化工具。

#### 📂 项目结构概览 (Project Structure)
‘’‘
Edge-LMM-Optimizer/
├── assets/                 # 测试资源（图片、Prompt 模板等）
├── configs/                # 配置文件（硬件限制、推理参数）
├── src/                    # 【核心源码】
│   ├── backends/           # 推理引擎封装 (负责加载模型、底层推理)
│   │   ├── base.py         # 标准接口定义
│   │   ├── torch_backend/  # PyTorch 后端实现
│   │   ├── vllm_backend/   # vLLM 后端实现
│   │   └── lmdeploy_backend/ # LMDeploy 后端实现
│   │
│   ├── pipelines/          # 业务流水线 (负责预处理、Prompt拼接、流式封装)
│   │   ├── base_pipeline.py
│   │   └── intervl_pipeline.py # InternVL 专用流水线
│   │
│   ├── methods/            # 优化算法库 (你的科研核心)
│   │   ├── pruning/        # 剪枝算法 (Token Pruning)
│   │   └── quantization/   # 量化脚本
│   │
│   ├── models/             # 模型适配层 (屏蔽不同模型的结构差异)
│   │   └── base.py         # 定义获取 Attention 层的标准接口
│   │
│   └── utils/              # 通用工具
│       ├── media_loader.py # 统一媒体加载器 (Image/Video -> PIL)
│       ├── video.py        # 视频抽帧工具
│       ├── metrics.py      # 显存与TPS监控
│       └── attention_analysis.py # 注意力可视化工具
│
├── test/                   # 单元测试与调试脚本 (Playground)
├── benchmark.py            # 自动化跑分脚本
├── main.py                 # 统一 CLI 入口
└── requirements.txt        # 项目依赖
’‘’

#### 🧩 核心模块详解

##### 1. 推理后端 (src/backends)

后端只负责“怎么跑”。它加载模型权重，接收处理好的 Tensor 或 Prompt，返回结果。

PyTorchBackend: 目前的主力后端。支持流式输出 (stream=True)，内部使用多线程运行 model.chat。

##### 2. 业务流水线 (src/pipelines)

流水线负责“怎么处理数据”。

InternVLPipeline:

自动判断输入是单图、多图还是视频。

集成 VideoProcessor 进行视频抽帧。

执行 InternVL 特有的 dynamic_preprocess (动态切片)。

流式支持: 当 stream=True 时，返回一个 Python 生成器 (generator)。

##### 3. 优化算法 (src/methods)

pruning/token_pruner.py: 管理剪枝生命周期。

pruning/pruning_hook.py: 实际注入模型的 Hook，负责在推理时计算 Token 重要性并生成 Mask。

##### 🛠️ 快速上手 (Quick Start)

1. 环境准备

确保 JetPack 环境就绪 (推荐 JetPack 6.0+)，安装基础依赖：

pip install -r requirements.txt
如果使用 vLLM 或 llama.cpp，请参考各自的官方文档进行编译安装


2. 运行基准测试 (Benchmark)

测试模型在当前硬件上的速度 (TPS) 和显存占用。

python benchmark.py --model_path /path/to/InternVL-Chat-V1-5


3. 启动推理 (CLI)

普通对话 (文本/图片)

python main.py \
    --model_path /path/to/model \
    --prompt "Describe this image." \
    --image assets/dog.jpg


视频理解 (流式输出)

# --stream 参数开启流式输出
python main.py \
    --model_path /path/to/model \
    --video assets/demo.mp4 \
    --prompt "What is happening in this video?" \
    --stream


🧪 开发与调试 (Development)

如何添加新的剪枝算法？

在 src/methods/pruning/ 下新建算法文件。

编写逻辑：接收 Attention Map，计算 Score，生成 Mask。

在 test/debug_pruning.py 中使用伪造数据验证数学逻辑。

如何进行注意力分析？

我们提供了可视化的分析工具，可以生成热力图并保存。

在 test/test_attention.py 中
from src.utils.attention_analysis import AttentionAnalyzer

analyzer = AttentionAnalyzer(model, tokenizer)
获取最后一层的注意力
attn_data = analyzer.get_attention_scores(inputs, layer_idx=-1)
绘制热力图
analyzer.plot_heatmap(attn_data, save_path="heatmap.png")


📝 常见问题 (FAQ)

Q: 为什么视频推理显存占用很高？

A: InternVL 默认会对每一帧进行动态切片（一张图切成 ~12 个 Patch）。对于视频，我们在 InternVLPipeline 中默认限制了 max_num=1，即每帧只作为一个 Patch 处理，以节省显存。

Q: 流式输出卡住不动？

A: 请检查 TextIteratorStreamer 是否正确初始化，以及 model.chat 是否在独立的线程中运行。参考 src/backends/torch_backend/internvl.py 中的实现。