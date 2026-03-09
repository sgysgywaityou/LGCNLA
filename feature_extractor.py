import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
import timm


class TextFeatureExtractor(nn.Module):
    """文本特征提取器（RoBERTa）"""

    def __init__(self, model_name: str = 'roberta-base', max_length: int = 100, language='en'):
        super().__init__()
        self.max_length = max_length
        self.language = language

        # 语言特定的RoBERTa模型
        if language == 'zh':
            model_name = 'hfl/chinese-roberta-wwm-ext'
        else:
            model_name = 'roberta-base'

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.hidden_dim = self.roberta.config.hidden_size

    def forward(self, texts: List[str]) -> torch.Tensor:
        """提取文本特征"""
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        device = next(self.roberta.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        return outputs.last_hidden_state

    def get_mean_vector(self, texts: List[str]) -> torch.Tensor:
        """获取文本的均值向量"""
        token_features = self.forward(texts)
        return token_features.mean(dim=1)


class ImageFeatureExtractor(nn.Module):
    """图像特征提取器（MaxViT）"""

    def __init__(self, model_name: str = 'maxvit_base_tf_512', num_patches: int = 200):
        super().__init__()
        self.num_patches = num_patches
        self.maxvit = timm.create_model(model_name, pretrained=True)

        if hasattr(self.maxvit, 'head'):
            self.maxvit.head = nn.Identity()

        self.hidden_dim = self.maxvit.num_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """提取图像特征"""
        features = self.maxvit.forward_features(images)

        if len(features.shape) == 2:
            features = features.unsqueeze(1).expand(-1, self.num_patches, -1)

        return features