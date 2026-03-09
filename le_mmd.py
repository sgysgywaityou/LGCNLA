import torch
import torch.nn as nn


class LEMD(nn.Module):
    """局部增强最大均值差异"""

    def __init__(self,
                 num_segments: int = 8,
                 alpha: float = 0.5,
                 sigma: float = 1.0,
                 gamma: float = 0.5):
        super().__init__()
        self.num_segments = num_segments
        self.alpha = alpha
        self.sigma = sigma
        self.gamma = gamma
        self.segment_weights = None

    def compute_segment_weights(self, real_features, fake_features):
        """基于判别力计算分段权重"""
        device = real_features.device
        dim = real_features.shape[1]
        seg_dim = dim // self.num_segments

        weights = []
        for s in range(self.num_segments):
            start = s * seg_dim
            end = (s + 1) * seg_dim if s < self.num_segments - 1 else dim

            real_seg = real_features[:, start:end]
            fake_seg = fake_features[:, start:end]

            mu_real = real_seg.mean(dim=0).norm()
            mu_fake = fake_seg.mean(dim=0).norm()
            var_pooled = (real_seg.var(dim=0).mean() + fake_seg.var(dim=0).mean()) / 2

            score = torch.abs(mu_real - mu_fake) / (torch.sqrt(var_pooled) + 1e-8)
            weights.append(score)

        weights = torch.tensor(weights).to(device)
        self.segment_weights = weights / (weights.sum() + 1e-8)
        return self.segment_weights

    def rbf_kernel(self, x, y):
        """RBF核函数"""
        dist = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-dist / (2 * self.sigma ** 2))

    def local_kernel(self, x, y):
        """局部增强核函数"""
        if self.segment_weights is None:
            return self.rbf_kernel(x, y)

        batch_size = x.shape[0]
        dim = x.shape[1]
        seg_dim = dim // self.num_segments

        kernel_sum = 0
        for s in range(self.num_segments):
            start = s * seg_dim
            end = (s + 1) * seg_dim if s < self.num_segments - 1 else dim

            x_seg = x[:, start:end]
            y_seg = y[:, start:end]

            kernel_sum += self.segment_weights[s] * self.rbf_kernel(x_seg, y_seg)

        return kernel_sum

    def combined_kernel(self, x, y):
        """组合全局和局部核"""
        global_k = self.rbf_kernel(x, y)
        local_k = self.local_kernel(x, y)
        return self.alpha * global_k + (1 - self.alpha) * local_k

    def forward(self, x, y, maximize=False):
        """
        计算LE-MMD损失
        Args:
            x: 源域特征 [batch_size, dim]
            y: 目标域特征 [batch_size, dim]
            maximize: 是否最大化差异（用于假新闻）
        """
        m = x.shape[0]
        n = y.shape[0]

        # 计算核矩阵
        K_xx = self.combined_kernel(x, x)
        K_yy = self.combined_kernel(y, y)
        K_xy = self.combined_kernel(x, y)

        # 无偏MMD估计
        mmd2 = (K_xx.sum() - K_xx.diag().sum()) / (m * (m - 1)) + \
               (K_yy.sum() - K_yy.diag().sum()) / (n * (n - 1)) - \
               2 * K_xy.mean()

        if maximize:
            # 边界损失：只惩罚低于边界的样本
            return torch.clamp(self.gamma - mmd2, min=0)
        return mmd2