# src/utils/video.py
import cv2
import numpy as np
from PIL import Image
import os


class VideoProcessor:
    """
    视频处理工具：负责将视频文件转换为一系列的 PIL Image 帧
    """

    def __init__(self, max_frames=8):
        self.max_frames = max_frames

    def extract_frames(self, video_path, num_frames=None):
        """
        从视频中均匀抽取指定数量的帧。

        Args:
            video_path (str): 视频文件路径
            num_frames (int): 需要抽取的帧数。如果不填，则使用初始化时的默认值。

        Returns:
            list[PIL.Image]: 抽取出的帧列表
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        target_frames = num_frames if num_frames else self.max_frames

        # 使用 OpenCV 读取
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            raise ValueError("Video has 0 frames.")

        # 计算均匀采样的索引
        # 例如总共100帧，取8帧 -> [0, 14, 28, 42, 57, 71, 85, 99]
        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

        frames = []
        for idx in indices:
            # 跳转到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # OpenCV 默认是 BGR，转为 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            else:
                print(f"Warning: Failed to read frame at index {idx}")

        cap.release()
        return frames

    @staticmethod
    def create_video_prompt(num_frames):
        """生成视频对应的 Prompt 前缀"""
        # InternVL 推荐格式: Frame 1: <image>\nFrame 2: <image>\n...
        return "".join([f"Frame {i + 1}: <image>\n" for i in range(num_frames)])

