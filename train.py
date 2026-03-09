import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from config.config import Config
from data.dataset import WeiboDataset, PolitiFactDataset, GossipCopDataset
from models.lgcnla import LGCNLA
from utils.metrics import compute_metrics


def train_epoch(model, dataloader, optimizer, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        labels = batch['label'].to(config.device)

        # 前向传播
        outputs = model(batch, training=True)
        loss, loss_dict = model.compute_loss(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # 收集预测结果
        preds = outputs['logits'].argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cla': f'{loss_dict["loss_cla"]:.4f}'
        })

    # 计算指标
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


def validate(model, dataloader, config):
    """验证"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            labels = batch['label'].to(config.device)
            outputs = model(batch, training=False)
            preds = outputs['logits'].argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    return metrics


def main():
    config = Config()

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 加载数据集
    print("Loading datasets...")
    train_dataset = WeiboDataset(config.weibo_path, split='train')
    val_dataset = WeiboDataset(config.weibo_path, split='val')
    test_dataset = WeiboDataset(config.weibo_path, split='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )

    # 创建模型
    model = LGCNLA(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # 训练循环
    best_val_f1 = 0
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, config)
        print(
            f"Train - Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, Loss: {train_metrics['loss']:.4f}")

        # 验证
        val_metrics = validate(model, val_loader, config)
        print(f"Val - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

        # 学习率调整
        scheduler.step()

        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(config.model_save_path, 'best_model.pth'))
            print(f"Saved best model with F1: {best_val_f1:.4f}")

    # 测试最佳模型
    print("\nTesting best model...")
    model.load_state_dict(torch.load(os.path.join(config.model_save_path, 'best_model.pth')))
    test_metrics = validate(model, test_loader, config)
    print(f"Test - Acc: {test_metrics['accuracy']:.4f}, "
          f"Prec: {test_metrics['precision']:.4f}, "
          f"Rec: {test_metrics['recall']:.4f}, "
          f"F1: {test_metrics['f1']:.4f}")


if __name__ == '__main__':
    main()