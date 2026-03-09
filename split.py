import json
import random
from collections import defaultdict


def split_weibo(data, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    """微博数据集随机分割（事件独立）"""
    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def split_politifact_gossipcop(data, event_key='event', seed=42):
    """PolitiFact和GossipCop的事件级分割"""
    random.seed(seed)

    # 按事件/实体分组
    event_to_items = defaultdict(list)
    for item in data:
        event = item.get(event_key, item.get('celebrity', 'unknown'))
        event_to_items[event].append(item)

    events = list(event_to_items.keys())
    random.shuffle(events)

    # 按事件数量分配
    n_events = len(events)
    train_end = int(n_events * 0.7)
    val_end = train_end + int(n_events * 0.1)

    train_events = events[:train_end]
    val_events = events[train_end:val_end]
    test_events = events[val_end:]

    train_data = []
    val_data = []
    test_data = []

    for event in train_events:
        train_data.extend(event_to_items[event])
    for event in val_events:
        val_data.extend(event_to_items[event])
    for event in test_events:
        test_data.extend(event_to_items[event])

    return train_data, val_data, test_data


def save_splits(data, train_data, val_data, test_data, output_dir):
    """保存分割后的数据"""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)