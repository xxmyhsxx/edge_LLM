import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Union

class MediaLoader:
    """
    多媒体加载工具类。
    负责将本地图片/视频路径加载为统一的 PIL.Image 列表格式，
    供后续的 Processor 进行处理。
    """

    @staticmethod
    def load(path: Union[str, List[str]], media_type: str = 'image', num_frames: int = 2) -> List[Image.Image]:
        """
        通用加载入口。
        
        Args:
            path: 文件路径 (str) 或 文件路径列表 (List[str])
            media_type: 'image', 'multi-image', 'video'
            num_frames: 如果是视频，需要抽取的帧数
            
        Returns:
            List[PIL.Image]: 加载好的图片对象列表
        """
        if media_type == 'image':
            if isinstance(path, list):
                raise ValueError("For media_type='image', path must be a string, not a list.")
            return MediaLoader.load_image(path)
            
        elif media_type == 'multi-image':
            if isinstance(path, str):
                path = [path] # 兼容单个字符串
            return MediaLoader.load_multi_images(path)
            
        elif media_type == 'video':
            if isinstance(path, list):
                path = path[0] # 视频通常只处理一个路径
            return MediaLoader.load_video(path, num_frames)
            
        elif media_type == 'text':
            return []
            
        else:
            raise ValueError(f"Unsupported media_type: {media_type}")

    @staticmethod
    def load_image(image_path: str) -> List[Image.Image]:
        """加载单张图片"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # .convert('RGB') 是必须的，防止 PNG 的 Alpha 通道导致模型报错
            img = Image.open(image_path).convert('RGB')
            return [img]
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return []

    @staticmethod
    def load_multi_images(image_paths: List[str]) -> List[Image.Image]:
        """加载多张图片"""
        images = []
        for p in image_paths:
            images.extend(MediaLoader.load_image(p))
        return images

    @staticmethod
    def load_video(video_path: str, num_frames: int = 8) -> List[Image.Image]:
        """
        从视频中均匀抽取指定数量的帧。
        使用 OpenCV 实现，无需安装 decord，兼容性更好。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            print(f"Warning: Video {video_path} has 0 frames.")
            return []

        # 计算均匀采样的索引
        # 例如: total=100, num=8 -> [0, 14, 28, ..., 99]
        # 使用 linspace 保证采样均匀覆盖整个视频
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            # 高效 Seek: 直接跳转到指定帧，而不是逐帧读取
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # OpenCV 默认是 BGR 格式，需要转换为 RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame)
                frames.append(pil_img)
            else:
                print(f"Warning: Could not read frame at index {idx}")
                
        cap.release()
        
        # 兜底逻辑：如果读取失败导致帧数不足，复制最后一帧补齐
        # 某些模型对输入帧数有严格要求
        if len(frames) > 0 and len(frames) < num_frames:
            while len(frames) < num_frames:
                frames.append(frames[-1].copy())
                
        return frames