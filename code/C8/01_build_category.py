import json
import os
from collections import Counter


def save_json_pretty(data, file_path):
    """
    将数据以格式化的 JSON 形式保存到文件。
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def collect_entity_types_from_file(file_path):
    """
    从单个文件中收集所有实体类型。
    """
    types = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                for entity in data.get('entities', []):
                    types.add(entity['type'])
            except json.JSONDecodeError:
                print(f"警告: 无法解析行: {line} in {file_path}")
    return types


def generate_tag_map(data_files, output_file):
    """
    从数据文件构建 BMES 标签映射并保存。
    """
    # 1. 从所有文件中收集实体类型
    all_entity_types = set()
    for file_path in data_files:
        all_entity_types.update(collect_entity_types_from_file(file_path))

    # 2. 排序以保证映射一致性
    sorted_types = sorted(list(all_entity_types))
    print(f"发现的实体类型: {sorted_types}")

    # 3. 构建 BMES 标签映射
    tag_to_id = {'O': 0}  # 'O' 代表非实体
    for entity_type in sorted_types:
        for prefix in ['B', 'M', 'E', 'S']:
            tag_name = f"{prefix}-{entity_type}"
            tag_to_id[tag_name] = len(tag_to_id)

    print(f"\n已生成 {len(tag_to_id)} 个标签映射。")

    # 4. 保存映射文件
    save_json_pretty(tag_to_id, output_file)
    print(f"标签映射已保存至: {output_file}")


if __name__ == '__main__':
    # 定义输入的数据文件和期望的输出路径
    train_file = './data/CMeEE-V2_train.json'
    dev_file = './data/CMeEE-V2_dev.json'
    output_path = './data/categories.json'

    generate_tag_map(data_files=[train_file, dev_file], output_file=output_path)
