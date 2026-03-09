import torch


class Config:
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集配置
    weibo_text_max_len = 100
    politifact_text_max_len = 300
    gossipcop_text_max_len = 200
    image_patches = 200

    # 特征维度
    text_dim = 768  # RoBERTa hidden dimension
    image_dim = 768  # MaxViT hidden dimension
    capsule_dim = 64
    num_capsules = 64

    # 图构建参数
    pmi_window_size = 5
    pmi_threshold = 0.0
    cos_threshold = 0.3

    # 图掩码比例
    word_mask_ratio = 0.2
    news_mask_ratio = 0.3
    enhanced_mask_ratio = 0.5
    entity_mask_ratio = 0.5

    # 路由参数
    routing_iterations = 3
    lambda_weight = 0.6  # 路由与注意力平衡系数

    # LE-MMD参数
    num_segments = 8
    mmd_alpha = 0.5  # 全局-局部平衡系数
    mmd_sigma = 1.0  # RBF核带宽
    gamma = 0.5  # 虚假新闻差异边界

    # 损失权重
    phi = 0.7  # 真实新闻LE-MMD权重
    eta = 0.5  # 虚假新闻LE-MMD权重
    tau = 0.6  # 质量控制阈值

    # 训练参数
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 50
    seed = 42

    # 路径配置
    weibo_path = './data/weibo/'
    politifact_path = './data/politifact/'
    gossipcop_path = './data/gossipcop/'
    cache_path = './cache/'
    model_save_path = './checkpoints/'