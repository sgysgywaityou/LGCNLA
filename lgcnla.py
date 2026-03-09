import torch
import torch.nn as nn
import torch.nn.functional as F
from .feature_extractor import TextFeatureExtractor, ImageFeatureExtractor
from .graph_builder import GraphBuilder
from .graph_capsule import GraphCapsuleNetwork
from .le_mmd import LEMD


class LGCNLA(nn.Module):
    """LGCNLA主模型"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 特征提取器
        self.text_extractor = TextFeatureExtractor(
            max_length=config.weibo_text_max_len
        )
        self.image_extractor = ImageFeatureExtractor(
            num_patches=config.image_patches
        )

        # 图构建器
        self.graph_builder = GraphBuilder(
            pmi_window=config.pmi_window_size,
            pmi_threshold=config.pmi_threshold,
            cos_threshold=config.cos_threshold
        )

        # 图胶囊网络
        self.capsule_net = GraphCapsuleNetwork(
            in_dim=config.text_dim,
            hidden_dims=[config.capsule_dim, config.capsule_dim],
            num_iterations=config.routing_iterations
        )

        # LE-MMD
        self.le_mmd = LEMD(
            num_segments=config.num_segments,
            alpha=config.mmd_alpha,
            sigma=config.mmd_sigma,
            gamma=config.gamma
        )

        # MLP层
        self.mlp1 = nn.Linear(config.text_dim, config.capsule_dim)
        self.mlp2 = nn.Linear(config.image_dim, config.capsule_dim)
        self.mlp3 = nn.Linear(config.capsule_dim, config.capsule_dim)
        self.mlp4 = nn.Linear(config.capsule_dim, config.capsule_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.capsule_dim * 2, config.capsule_dim),
            nn.ReLU(),
            nn.Linear(config.capsule_dim, 2)
        )

    def forward(self, batch, training=True):
        """前向传播"""
        texts = batch['text']
        images = batch['image'].to(self.config.device)
        labels = batch.get('label', None)

        # 1. 特征提取
        text_features = self.text_extractor.get_mean_vector(texts)  # [B, d]
        image_features = self.image_extractor(images)  # [B, p, d]
        image_features = image_features.mean(dim=1)  # [B, d]

        # 2. 图构建（简化版本，实际需要更复杂的处理）
        # 这里假设enhanced_docs和external_entities已经预先生成并传入
        enhanced_docs = batch.get('enhanced_docs', None)
        external_entities = batch.get('external_entities', None)

        if enhanced_docs is not None and external_entities is not None:
            # 构建图G1和G2
            adj1 = self._build_graph_G1(texts, enhanced_docs, text_features)
            adj2 = self._build_graph_G2(texts, external_entities, text_features)

            # 图胶囊网络处理
            g1_features = self.capsule_net(text_features, adj1)
            g2_features = self.capsule_net(text_features, adj2)
        else:
            # 推理阶段如果没有增强信息，使用原始特征
            g1_features = text_features
            g2_features = text_features

        # 3. 特征融合
        n = self.mlp1(text_features)
        n1 = self.mlp3(g1_features)
        n2 = self.mlp4(g2_features)
        p = self.mlp2(image_features)

        # 平均融合
        n_combined = (n + n1 + n2) / 3
        combined = torch.cat([n_combined, p], dim=1)

        # 4. 分类
        logits = self.classifier(combined)

        outputs = {'logits': logits}

        # 5. 训练时计算LE-MMD损失
        if training and labels is not None:
            real_mask = (labels == 0)
            fake_mask = (labels == 1)

            if real_mask.any():
                # 计算真实新闻的LE-MMD权重
                self.le_mmd.compute_segment_weights(
                    n_combined[real_mask],
                    n_combined[fake_mask] if fake_mask.any() else n_combined
                )

                # 真实新闻LE-MMD损失（最小化）
                loss_real = self.le_mmd(
                    n_combined[real_mask],
                    p[real_mask],
                    maximize=False
                )
            else:
                loss_real = torch.tensor(0.0).to(self.config.device)

            if fake_mask.any():
                # 虚假新闻LE-MMD损失（最大化，使用边界损失）
                loss_fake = self.le_mmd(
                    n_combined[fake_mask],
                    p[fake_mask],
                    maximize=True
                )
            else:
                loss_fake = torch.tensor(0.0).to(self.config.device)

            outputs['loss_real'] = loss_real
            outputs['loss_fake'] = loss_fake

        return outputs

    def _build_graph_G1(self, texts, enhanced_docs, text_features):
        """构建图G1的邻接矩阵"""
        # 简化实现：实际需要按照论文公式(11)构建
        batch_size = len(texts)
        adj = torch.eye(batch_size * 2).to(self.config.device)
        # 这里简化处理，实际需要构建包含word nodes的完整图
        return adj

    def _build_graph_G2(self, texts, external_entities, text_features):
        """构建图G2的邻接矩阵"""
        # 简化实现：实际需要按照论文公式(12)构建
        batch_size = len(texts)
        adj = torch.eye(batch_size * 2).to(self.config.device)
        return adj

    def compute_loss(self, outputs, labels):
        """计算总损失"""
        logits = outputs['logits']

        # 分类损失
        loss_cla = F.cross_entropy(logits, labels)

        # LE-MMD损失
        loss_real = outputs.get('loss_real', torch.tensor(0.0).to(self.config.device))
        loss_fake = outputs.get('loss_fake', torch.tensor(0.0).to(self.config.device))

        # 总损失
        total_loss = loss_cla + self.config.phi * loss_real + self.config.eta * loss_fake

        return total_loss, {
            'loss_cla': loss_cla.item(),
            'loss_real': loss_real.item(),
            'loss_fake': loss_fake.item()
        }