import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import RobertaTokenizer
from .preprocessing import preprocess_text


class FNDDataset(Dataset):
    """假新闻检测数据集基类"""

    def __init__(self, data_path, split='train', text_max_len=100, image_size=224):
        self.data_path = data_path
        self.split = split
        self.text_max_len = text_max_len
        self.image_size = image_size
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        # 加载数据
        self.data = self._load_data()

    def _load_data(self):
        """加载数据"""
        data_file = os.path.join(self.data_path, f'{self.split}.json')
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _load_image(self, image_path):
        """加载并预处理图像"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image) / 255.0
        image = torch.FloatTensor(image).permute(2, 0, 1)  # HWC -> CHW
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 文本
        text = preprocess_text(item['text'])
        text_encoding = self.tokenizer(
            text,
            max_length=self.text_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 图像
        image = self._load_image(item['image_path'])

        # 标签
        label = torch.tensor(item['label'], dtype=torch.long)

        return {
            'text': text,
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'image': image,
            'label': label,
            'news_id': item.get('news_id', str(idx))
        }


class WeiboDataset(FNDDataset):
    """微博数据集"""

    def __init__(self, data_path, split='train', **kwargs):
        super().__init__(data_path, split, text_max_len=100, **kwargs)


class PolitiFactDataset(FNDDataset):
    """PolitiFact数据集"""

    def __init__(self, data_path, split='train', **kwargs):
        super().__init__(data_path, split, text_max_len=300, **kwargs)


class GossipCopDataset(FNDDataset):
    """GossipCop数据集"""

    def __init__(self, data_path, split='train', **kwargs):
        super().__init__(data_path, split, text_max_len=200, **kwargs)