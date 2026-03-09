import torch
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
import math


class GraphBuilder:
    """图构建器类"""

    def __init__(self, pmi_window: int = 5, pmi_threshold: float = 0.0, cos_threshold: float = 0.3):
        self.pmi_window = pmi_window
        self.pmi_threshold = pmi_threshold
        self.cos_threshold = cos_threshold
        self.vocab = None
        self.idf = None
        self.pmi_dict = None

    def build_vocabulary(self, documents: List[str]) -> Dict[str, int]:
        """构建词表"""
        word_counts = Counter()
        for doc in documents:
            words = doc.split()
            word_counts.update(words)

        # 过滤低频词
        self.vocab = {word: idx for idx, (word, count) in enumerate(word_counts.items()) if count >= 2}
        return self.vocab

    def compute_idf(self, documents: List[str]) -> Dict[str, float]:
        """计算IDF（基于全局语料）"""
        doc_count = len(documents)
        word_doc_freq = Counter()

        for doc in documents:
            words = set(doc.split())
            word_doc_freq.update(words)

        self.idf = {}
        for word, freq in word_doc_freq.items():
            self.idf[word] = math.log((doc_count + 1) / (freq + 1)) + 1

        return self.idf

    def compute_pmi(self, documents: List[str]) -> Dict[Tuple[str, str], float]:
        """计算PMI"""
        # 统计词频和共现
        word_counts = Counter()
        cooccur_counts = Counter()
        window_counts = 0

        for doc in documents:
            words = doc.split()
            for i, w1 in enumerate(words):
                word_counts[w1] += 1
                # 滑动窗口
                for j in range(max(0, i - self.pmi_window), min(len(words), i + self.pmi_window + 1)):
                    if i != j:
                        w2 = words[j]
                        if w1 < w2:
                            cooccur_counts[(w1, w2)] += 1
                        else:
                            cooccur_counts[(w2, w1)] += 1
                window_counts += 1

        total_words = sum(word_counts.values())

        # 计算PMI
        pmi_dict = {}
        for (w1, w2), count in cooccur_counts.items():
            p_xy = count / window_counts
            p_x = word_counts[w1] / total_words
            p_y = word_counts[w2] / total_words
            pmi = math.log2((p_xy + 1e-8) / (p_x * p_y + 1e-8))
            if pmi > self.pmi_threshold:
                pmi_dict[(w1, w2)] = pmi

        self.pmi_dict = pmi_dict
        return pmi_dict

    def build_G1(self,
                 news_docs: List[str],
                 enhanced_docs: List[List[str]],
                 doc_features: torch.Tensor,
                 enhanced_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建图G1（新闻文档 + 增强文档）"""
        # 构建词表
        all_docs = news_docs + [doc for sublist in enhanced_docs for doc in sublist]
        self.build_vocabulary(all_docs)
        self.compute_idf(news_docs)
        self.compute_pmi(news_docs)

        num_words = len(self.vocab)
        num_docs = len(news_docs)
        num_enhanced = sum(len(docs) for docs in enhanced_docs)
        total_nodes = num_words + num_docs + num_enhanced

        adj = torch.zeros((total_nodes, total_nodes))

        # 构建节点索引
        word_to_idx = {word: idx for word, idx in self.vocab.items()}
        doc_start = num_words
        enhanced_start = num_words + num_docs

        # 1. 词-词边 (PMI)
        for (w1, w2), pmi in self.pmi_dict.items():
            if w1 in word_to_idx and w2 in word_to_idx:
                i = word_to_idx[w1]
                j = word_to_idx[w2]
                adj[i, j] = pmi
                adj[j, i] = pmi

        # 2. 文档-词边 (TF-IDF)
        for i, doc in enumerate(news_docs):
            doc_idx = doc_start + i
            words = doc.split()
            word_counts = Counter(words)
            for word, count in word_counts.items():
                if word in word_to_idx:
                    word_idx = word_to_idx[word]
                    tf = count / len(words)
                    idf = self.idf.get(word, 1.0)
                    tfidf = tf * idf
                    adj[doc_idx, word_idx] = tfidf
                    adj[word_idx, doc_idx] = tfidf

        # 3. 文档-增强文档边 (点积)
        for i, doc_feat in enumerate(doc_features):
            doc_idx = doc_start + i
            start_idx = enhanced_start + sum(len(enhanced_docs[j]) for j in range(i))
            for j, enh_feat in enumerate(enhanced_features[i]):
                enh_idx = start_idx + j
                dot_product = torch.dot(doc_feat, enh_feat).item()
                adj[doc_idx, enh_idx] = dot_product
                adj[enh_idx, doc_idx] = dot_product

        # 添加自环
        adj += torch.eye(total_nodes)

        return adj, self._get_node_features(news_docs, enhanced_docs, doc_features, enhanced_features)

    def _get_node_features(self, news_docs, enhanced_docs, doc_features, enhanced_features):
        """获取节点特征"""
        # 简化实现：实际中需要根据节点类型构建特征矩阵
        return doc_features