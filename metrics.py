import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def compute_per_class_metrics(y_true, y_pred):
    """计算各类别的指标"""
    classes = np.unique(y_true)
    metrics = {}

    for cls in classes:
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        metrics[f'class_{cls}'] = {
            'precision': precision_score(y_true_binary, y_pred_binary),
            'recall': recall_score(y_true_binary, y_pred_binary),
            'f1': f1_score(y_true_binary, y_pred_binary)
        }

    return metrics


def compute_confidence_interval(metrics_list):
    """计算置信区间"""
    metrics_array = np.array(metrics_list)
    mean = np.mean(metrics_array, axis=0)
    std = np.std(metrics_array, axis=0)

    return mean, std