import argparse
import os
import sys

# 再次确保根目录在 path 中
sys.path.append(os.getcwd())

from src.methods.research.modality_probe import ModalityProbe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--image", type=str, required=True, help="测试图片路径")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        exit(1)

    probe = ModalityProbe(args.model)
    probe.run_probe(args.image, "Describe the image.")