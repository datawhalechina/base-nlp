import json
import os
from collections import defaultdict


def save_json(data, file_path):
    """
    将 Python 对象以格式化的 JSON 形式保存到文件。
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_entity_types_from_file(file_path):
    """
    从单个数据文件中提取所有唯一的实体类型。
    """
    entity_types = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # 遍历实体列表，提取 'type' 字段
                for entity in data.get('entities', []):
                    entity_types.add(entity['type'])
            except json.JSONDecodeError:
                print(f"警告: 无法解析行: {line} in {file_path}")
    return entity_types


def build_label_map(data_files, output_file):
    """
    从多个数据文件构建并保存一个完整的标签到ID的映射表。
    """
    # 1. 从所有提供的数据文件中提取所有唯一的实体类型
    all_types = set()
    for file_path in data_files:
        all_types.update(get_entity_types_from_file(file_path))

    # 排序以确保每次生成的映射表顺序一致
    sorted_types = sorted(list(all_types))
    print(f"从数据中发现的实体类型: {sorted_types}")

    # 2. 基于BMES模式构建 label2id 映射字典
    label2id = {'O': 0}  # 'O' 代表非实体 (Outside)
    for entity_type in sorted_types:
        for prefix in ['B', 'M', 'E', 'S']:
            label_name = f"{prefix}-{entity_type}"
            label2id[label_name] = len(label2id)

    print(f"\n已生成 {len(label2id)} 个标签的映射关系。")

    # 3. 将映射表保存到指定的输出文件
    save_json(label2id, output_file)
    print(f"标签映射表已保存至: {output_file}")


if __name__ == '__main__':
    # 定义输入的数据文件和期望的输出路径
    train_file = './data/CMeEE-V2_train.json'
    dev_file = './data/CMeEE-V2_dev.json'
    output_path = './data/categories.json'

    build_label_map(data_files=[train_file, dev_file], output_file=output_path)
