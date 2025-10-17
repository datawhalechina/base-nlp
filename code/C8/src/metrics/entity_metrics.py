
def _trans_entity2tuple(label_ids, id2tag):
    """
    将标签ID序列转换为实体元组列表。
    一个实体元组示例: ('PER', 0, 2) -> (实体类型, 起始位置, 结束位置)
    """
    entities = []
    current_entity = None

    for i, label_id in enumerate(label_ids):
        tag = id2tag.get(label_id.item(), 'O')

        if tag.startswith('B-'):
            # 遇到新的 B- 不自动闭合旧片段，直接开启新片段
            current_entity = (tag[2:], i, i + 1)
        elif tag.startswith('M-'):
            if current_entity and current_entity[0] == tag[2:]:
                current_entity = (current_entity[0], current_entity[1], i + 1)
            else:
                # 非法 M-，丢弃当前片段
                current_entity = None
        elif tag.startswith('E-'):
            if current_entity and current_entity[0] == tag[2:]:
                current_entity = (current_entity[0], current_entity[1], i + 1)
                entities.append(current_entity)
            # 无论是否匹配，E- 处都结束当前片段
            current_entity = None
        elif tag.startswith('S-'):
            # 单字实体直接落盘
            entities.append((tag[2:], i, i + 1))
            current_entity = None
        else:  # 'O'
            # O 不闭合未完成片段，直接丢弃
            current_entity = None
        
    return set(entities)

def calculate_entity_level_metrics(all_pred_ids, all_label_ids, all_masks, id2tag):
    """
    计算实体级别的精确率、召回率和 F1 分数（逐样本 + 严格解码）。
    """
    true_entities = set()
    pred_entities = set()

    sample_idx = 0
    # 逐 batch，逐样本地应用 mask 后再解码，避免跨样本串扰
    for preds_batch, labels_batch, masks_batch in zip(all_pred_ids, all_label_ids, all_masks):
        B = labels_batch.shape[0]
        for b in range(B):
            row_mask = masks_batch[b].bool()
            row_labels = labels_batch[b][row_mask]
            row_preds = preds_batch[b][row_mask]

            te = _trans_entity2tuple(row_labels, id2tag)
            pe = _trans_entity2tuple(row_preds, id2tag)

            true_entities.update({(sample_idx,) + e for e in te})
            pred_entities.update({(sample_idx,) + e for e in pe})
            sample_idx += 1

    num_correct = len(true_entities.intersection(pred_entities))
    num_true = len(true_entities)
    num_pred = len(pred_entities)

    precision = num_correct / num_pred if num_pred > 0 else 0.0
    recall = num_correct / num_true if num_true > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
