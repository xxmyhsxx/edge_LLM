import sys
import os
import torch

# 添加根目录路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.intervl_pipeline import InternVLPipeline
from src.utils.attention_analysis import AttentionAnalyzer


def test_attention_flow():
    model_path = "/app/models/InterVL-Chat-V1-5"  # 修改为你的路径

    # 1. 初始化 Pipeline
    pipeline = InternVLPipeline(model_path)

    # 2. 准备数据 (手动构建 input，模拟 pipeline 内部逻辑)
    prompt = "Describe this image."
    image_path = "assets/dog.jpg"  # 确保有一张测试图

    # 利用 pipeline 的 helper 函数处理图片
    pixel_values, _ = pipeline._process_image_list(
        [pipeline._load_image_from_file(image_path)],  # 假设你有这个 helper，或者用 PIL打开
        max_num=1  # 测试时为了看清 attention，图片切片少一点
    )

    # 利用 model 的 tokenizer 处理文本
    # 注意：这里简写了，实际需要加 <image> 等特殊 token
    final_prompt = f"<img>{prompt}"  # 简化的 prompt
    input_ids = pipeline.tokenizer(final_prompt, return_tensors='pt').input_ids.cuda()

    # 构造模型输入字典
    inputs = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        # "image_flags": torch.tensor([1]).cuda() # 部分版本需要
    }

    # 3. 初始化分析器
    analyzer = AttentionAnalyzer(pipeline.model, pipeline.tokenizer)

    # 4. 提取注意力 (取最后一层)
    attn_data = analyzer.get_attention_scores(inputs, layer_idx=-1)

    # 5. 可视化热力图
    analyzer.plot_heatmap(attn_data, save_path="tests/attn_heatmap.png")

    # 6. 分析重要 Token
    analyzer.analyze_token_importance(attn_data, top_k=10)

    # 7. 绘制分布曲线
    analyzer.plot_importance_curve(
        analyzer.analyze_token_importance(attn_data, top_k=50)[1],
        save_path="tests/attn_distribution.png"
    )


if __name__ == "__main__":
    # 注意：为了能运行 pipeline 的私有方法，这里是演示逻辑
    # 实际上建议在 pipeline 里加一个 `preprocess_inputs` 公有方法
    try:
        from PIL import Image

        # 简单的 mock 运行，如果报错可能是 prompt 格式问题
        # 这里主要是为了展示 Analyzer 的用法
        pass
    except Exception as e:
        print(e)