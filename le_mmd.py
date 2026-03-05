"""
LE-MMD模块 - 局部增强最大均值差异
"""

import torch
import torch.nn as nn
import numpy as np


class LEMD(nn.Module):
    """局部增强最大均值差异"""

    def __init__(self,
                 num_segments: int = 8,
                 alpha: float = 0.5,
                 sigma: float = 1.0):
        """
        初始化LE-MMD

        Args:
            num_segments: 分段数量
            alpha: 全局-局部平衡系数
            sigma: RBF核带宽参数
        """
        super().__init__()
        self.num_segments = num_segments
        self.alpha = alpha
        self.sigma = sigma

    def forward(self,
                X: torch.Tensor,
                Y: torch.Tensor,
                maximize: bool = False) -> torch.Tensor:
        """
        计算LE-MMD损失

        Args:
            X: 源域特征 [batch_size, dim]
            Y: 目标域特征 [batch_size, dim]
            maximize: 是否最大化差异（用于假新闻）

        Returns:
            LE-MMD损失
        """
        batch_size, dim