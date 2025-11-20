# src/utils/image_patch_mapper.py
import torch
import numpy as np
from PIL import Image, ImageDraw


class ImagePatchMapper:
    """
    负责将模型内部的图片 Patch 索引映射回原始图像的像素区域。
    主要用于可视化被删除的图片区域。
    """

    def __init__(self, image_size=448, patch_size=14):
        """
        Args:
            image_size (int): 模型输入的单个图片块 (e.g., 448x448)。
            patch_size (int): Vision Transformer 的 Patch 大小 (e.g., 14x14)。
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_per_side = image_size // patch_size  # 每边有多少个 patch
        self.total_patches_per_image = self.num_patches_per_side ** 2  # 一张图的总 patch 数

    def map_patch_to_pixel_coords(self, patch_index_in_single_image):
        """
        将在一个 448x448 图片中的 Patch 索引映射到其像素坐标 (左上角 x, y, 右下角 x, y)。

        Args:
            patch_index_in_single_image (int): 在一个 448x448 切片中的 Patch 索引。

        Returns:
            tuple: (x_min, y_min, x_max, y_max)
        """
        row = patch_index_in_single_image // self.num_patches_per_side
        col = patch_index_in_single_image % self.num_patches_per_side

        x_min = col * self.patch_size
        y_min = row * self.patch_size
        x_max = x_min + self.patch_size
        y_max = y_min + self.patch_size

        return (x_min, y_min, x_max, y_max)

    def visualize_removed_patches(self, original_image: Image.Image, removed_patch_indices: list,
                                  color=(255, 0, 0, 128),  # 半透明红色
                                  save_path="removed_patches_viz.png"):
        """
        在原始图片上标记被删除的 Patch 区域。

        Args:
            original_image (PIL.Image): 模型的原始输入图片 (已经过 _dynamic_preprocess 切片后的某个 448x448 块)。
            removed_patch_indices (list): 被删除的 Patch 索引列表 (相对于该 448x448 块)。
            color (tuple): 标记颜色 (RGBA)。
            save_path (str): 保存可视化结果的路径。
        """
        if not removed_patch_indices:
            print("No patches were removed. Skipping visualization.")
            return

        # 创建一个可绘制的副本
        img_viz = original_image.copy()
        draw = ImageDraw.Draw(img_viz, 'RGBA')

        for patch_idx in removed_patch_indices:
            if 0 <= patch_idx < self.total_patches_per_image:
                coords = self.map_patch_to_pixel_coords(patch_idx)
                draw.rectangle(coords, fill=color, outline=color)
            else:
                print(f"Warning: Invalid patch index {patch_idx} for visualization.")

        img_viz.save(save_path)
        print(f">>> [Viz] Removed patches visualization saved to {save_path}")

    def get_patch_range_for_image_slice(self, image_slice_idx, num_patches_list):
        """
        计算某个 image_slice (如 InternVL 动态切片后的 448x448 块) 在整个 pixel_values tensor 中的起始和结束 patch 索引。

        Args:
            image_slice_idx (int): 目标 image_slice 的索引 (例如，InternVL 切了 3 块，这是第 0, 1, 2 块)
            num_patches_list (list): InternVLPipeline._process_image_list 返回的 num_patches_list。
                                     形如 [N_patches_img1_slice1, N_patches_img1_slice2, ..., N_patches_imgK_sliceM]
        Returns:
            tuple: (start_global_patch_idx, end_global_patch_idx)
        """
        start_idx = sum(num_patches_list[:image_slice_idx])
        end_idx = start_idx + num_patches_list[image_slice_idx]
        return start_idx, end_idx