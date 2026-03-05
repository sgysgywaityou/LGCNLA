"""
特征提取模块 - 使用RoBERTa和MaxViT提取文本和图像特征
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
import timm


class TextFeatureExtractor(nn.Module):
    """文本特征提取器（RoBERTa）"""

    def __init__(self, model_name: str = 'roberta-base', max_length: int = 100):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.hidden_dim = self.roberta.config.hidden_size

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        提取文本特征

        Args:
            texts: 文本列表

        Returns:
            文本特征张量 [batch_size, max_length, hidden_dim]
        """
        # 分词
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 将输入移动到同一设备
        device = next(self.roberta.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # 前向传播
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # 返回所有token的隐藏状态
        return outputs.last_hidden_state

    def get_mean_vector(self, texts: List[str]) -> torch.Tensor:
        """获取文本的均值向量（用于质量控制）"""
        token_features = self.forward(texts)
        return token_features.mean(dim=1)


class ImageFeatureExtractor(nn.Module):
    """图像特征提取器（MaxViT）"""

    def __init__(self, model_name: str = 'maxvit_base_tf_512', num_patches: int = 200):
        super().__init__()
        self.num_patches = num_patches
        self.maxvit = timm.create_model(model_name, pretrained=True)

        # 移除分类头
        if hasattr(self.maxvit, 'head'):
            self.maxvit.head = nn.Identity()

        # 获取特征维度
        self.hidden_dim = self.maxvit.num_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        提取图像特征

        Args:
            images: 图像张量 [batch_size, channels, height, width]

        Returns:
            图像特征张量 [batch_size, num_patches, hidden_dim]
        """
        # MaxViT前向传播
        features = self.maxvit.forward_features(images)

        # 如果返回的是全局特征，需要调整为patch特征
        if len(features.shape) == 2:
            # 简单复制以模拟patch
            features = features.unsqueeze(1).expand(-1, self.num_patches, -1)

        return features