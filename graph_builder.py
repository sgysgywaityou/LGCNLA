"""
图构建模块 - 构建双通道异构图
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


class GraphBuilder:
    """图构建器类"""

    def __init__(self, pmi_threshold: float = 0.0, cos_threshold: float = 0.3):
        """
        初始化图构建器

        Args:
            pmi_threshold: PMI阈值
            cos_threshold: 余弦相似度阈值
        """
        self.pmi_threshold = pmi_threshold
        self.cos_threshold = cos_threshold
        self.tfidf_vectorizer = TfidfVectorizer()

    def build_G1(self,
                 news_docs: List[str],
                 enhanced_docs: List[List[str]],
                 word_vocab: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建图G1（新闻文档 + 增强文档）

        Args:
            news_docs: 新闻文档列表
            enhanced_docs: 每个新闻对应的增强文档列表
            word_vocab: 词表

        Returns:
            (邻接矩阵, 节点特征)
        """
        num_nodes = self._get_num_nodes_G1(news_docs, enhanced_docs, word_vocab)
        adj_matrix = torch.zeros((num_nodes, num_nodes))

        # 构建节点索引映射
        node2idx = self._build_node_index_G1(news_docs, enhanced_docs, word_vocab)

        # 1. 词-词边 (基于PMI)
        self._add_word_word_edges(adj_matrix, node2idx, word_vocab)

        # 2. 新闻文档-词边 (基于TF-IDF)
        self._add_doc_word_edges_G1(adj_matrix, node2idx, news_docs, word_vocab)

        # 3. 新闻文档-增强文档边 (基于点积)
        self._add_doc_enhanced_edges(adj_matrix, node2idx, news_docs, enhanced_docs)

        # 添加自环
        adj_matrix += torch.eye(num_nodes)

        return adj_matrix

    def build_G2(self,
                 news_docs: List[str],
                 external_entities: List[List[str]],
                 word_vocab: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建图G2（新闻文档 + 外部实体）

        Args:
            news_docs: 新闻文档列表
            external_entities: 每个新闻对应的外部实体列表
            word_vocab: 词表

        Returns:
            (邻接矩阵, 节点特征)
        """
        num_nodes = self._get_num_nodes_G2(news_docs, external_entities, word_vocab)
        adj_matrix = torch.zeros((num_nodes, num_nodes))

        # 构建节点索引映射
        node2idx = self._build_node_index_G2(news_docs, external_entities, word_vocab)

        # 1. 词-词边 (基于PMI)
        self._add_word_word_edges(adj_matrix, node2idx, word_vocab)

        # 2. 新闻文档-词边 (基于TF-IDF)
        self._add_doc_word_edges_G2(adj_matrix, node2idx, news_docs, word_vocab)

        # 3. 新闻文档-外部实体边 (基于余弦相似度)
        self._add_doc_entity_edges(adj_matrix, node2idx, news_docs, external_entities)

        # 添加自环
        adj_matrix += torch.eye(num_nodes)

        return adj_matrix

    def _compute_pmi(self, texts: List[str]) -> Dict[Tuple[str, str], float]:
        """计算点互信息"""
        from collections import Counter
        import math

        # 统计词频和共现
        word_counts = Counter()
        cooccur_counts = Counter()

        for text in texts:
            words = text.split()
            word_counts.update(words)

            # 滑动窗口统计共现
            window_size = 5
            for i, w1 in enumerate(words):
                for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                    if i != j:
                        w2 = words[j]
                        if w1 < w2:
                            cooccur_counts[(w1, w2)] += 1
                        else:
                            cooccur_counts[(w2, w1)] += 1

        total_words = sum(word_counts.values())
        total_pairs = sum(cooccur_counts.values())

        # 计算PMI
        pmi_dict = {}
        for (w1, w2), count in cooccur_counts.items():
            p_xy = count / total_pairs
            p_x = word_counts[w1] / total_words
            p_y = word_counts[w2] / total_words
            pmi = math.log2(p_xy / (p_x * p_y))
            pmi_dict[(w1, w2)] = pmi

        return pmi_dict

    def _add_word_word_edges(self, adj_matrix, node2idx, word_vocab):
        """添加词-词边（基于PMI）"""
        pmi_dict = self._compute_pmi(list(word_vocab.keys()))

        for (w1, w2), pmi in pmi_dict.items():
            if pmi > self.pmi_threshold and w1 in node2idx and w2 in node2idx:
                i = node2idx[w1]
                j = node2idx[w2]
                adj_matrix[i, j] = pmi
                adj_matrix[j, i] = pmi