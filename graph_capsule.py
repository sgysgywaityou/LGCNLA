"""
图胶囊网络模块 - 实现动态路由机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphCapsuleLayer(nn.Module):
    """图胶囊层"""

    def __init__(self, in_dim: int, out_dim: int, num_iterations: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_iterations = num_iterations

        # 投影矩阵
        self.W = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.W)

        # 注意力参数
        self.attn_a = nn.Parameter(torch.Tensor(2 * out_dim, 1))
        nn.init.xavier_uniform_(self.attn_a)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            h: 节点特征 [num_nodes, in_dim]
            adj: 邻接矩阵 [num_nodes, num_nodes]

        Returns:
            输出胶囊 [num_nodes, out_dim]
        """
        num_nodes = h.shape[0]
        device = h.device

        # 归一化邻接矩阵
        d = adj.sum(dim=1) + 1e-8
        d_sqrt = torch.sqrt(d)
        d_inv_sqrt = 1.0 / d_sqrt
        adj_norm = d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)

        # 投影到胶囊空间
        u_hat = torch.matmul(h, self.W)  # [num_nodes, out_dim]

        # 初始化耦合系数
        b = adj_norm.clone()  # [num_nodes, num_nodes]

        # 注意力权重
        h_concat = torch.cat([h.unsqueeze(1).expand(-1, num_nodes, -1),
                              h.unsqueeze(0).expand(num_nodes, -1, -1)], dim=-1)
        attn_scores = torch.matmul(h_concat, self.attn_a).squeeze(-1)  # [num_nodes, num_nodes]
        attn_weights = F.leaky_relu(attn_scores, negative_slope=0.2)

        # 动态路由
        for _ in range(self.num_iterations):
            # 计算路由权重
            c = F.softmax(b, dim=1)  # [num_nodes, num_nodes]

            # 加权求和
            s = torch.matmul(c, u_hat)  # [num_nodes, out_dim]

            # Squash激活
            v = self._squash(s)

            # 更新耦合系数
            if _ < self.num_iterations - 1:
                agreement = torch.matmul(v, u_hat.T)  # [num_nodes, num_nodes]
                b = b + agreement

        # 融合注意力
        lambda_weight = 0.6  # 从配置读取
        combined_weights = lambda_weight * c + (1 - lambda_weight) * F.softmax(attn_weights, dim=1)
        s = torch.matmul(combined_weights, u_hat)
        v = self._squash(s)

        return v

    def _squash(self, s: torch.Tensor) -> torch.Tensor:
        """Squash激活函数"""
        s_norm = torch.norm(s, dim=1, keepdim=True)
        scale = s_norm / (1 + s_norm)
        return scale * s / (s_norm + 1e-8)


class GraphCapsuleNetwork(nn.Module):
    """图胶囊网络"""

    def __init__(self,
                 in_dim: int,
                 hidden_dims: List[int],
                 num_iterations: int = 3):
        super().__init__()

        self.layers = nn.ModuleList()
        prev_dim = in_dim

        for hidden_dim in hidden_dims:
            self.layers.append(
                GraphCapsuleLayer(prev_dim, hidden_dim, num_iterations)
            )
            prev_dim = hidden_dim

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, in_dim]
            adj: 邻接矩阵 [num_nodes, num_nodes]

        Returns:
            更新后的节点特征 [num_nodes, hidden_dims[-1]]
        """
        for layer in self.layers:
            x = layer(x, adj)
        return x