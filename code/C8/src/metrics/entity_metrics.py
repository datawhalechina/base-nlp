
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
            if current_entity:
                entities.append(current_entity)
            entity_type = tag[2:]
            current_entity = (entity_type, i, i + 1)
        elif tag.startswith('M-'):
            if current_entity and current_entity[0] == tag[2:]:
                current_entity = (current_entity[0], current_entity[1], i + 1)
            else:
                current_entity = None
        elif tag.startswith('E-'):
            if current_entity and current_entity[0] == tag[2:]:
                current_entity = (current_entity[0], current_entity[1], i + 1)
                entities.append(current_entity)
            current_entity = None
        elif tag.startswith('S-'):
            if current_entity:
                entities.append(current_entity)
            entity_type = tag[2:]
            entities.append((entity_type, i, i + 1))
            current_entity = None
        else: # 'O' tag
            if current_entity:
                entities.append(current_entity)
            current_entity = None
            
    if current_entity:
        entities.append(current_entity)
        
    return set(entities)

def calculate_entity_level_metrics(all_pred_ids, all_label_ids, all_masks, id2tag):
    """
    计算实体级别的精确率、召回率和 F1 分数。
    """
    # 过滤掉填充部分
    active_preds = [p[m] for p, m in zip(all_pred_ids, all_masks)]
    active_labels = [l[m] for l, m in zip(all_label_ids, all_masks)]

    true_entities = set()
    pred_entities = set()

    for i in range(len(active_labels)):
        # 为每个样本添加唯一标识符，以区分不同样本中的相同实体
        # (样本ID, 实体类型, 起始位置, 结束位置)
        sample_true_entities = {(i,) + entity for entity in _trans_entity2tuple(active_labels[i], id2tag)}
        sample_pred_entities = {(i,) + entity for entity in _trans_entity2tuple(active_preds[i], id2tag)}
        
        true_entities.update(sample_true_entities)
        pred_entities.update(sample_pred_entities)
        
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
