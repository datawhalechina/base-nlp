# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/8/10 16:16
Create User : 19410
Desc : 将类别名称中从原始数据中提取出来，构建成一个字典mapping，并保存到磁盘
"""

import json
import os
from typing import List


def save_json(file, obj):
    """
    将obj对象以json格式的形式保存到对应的磁盘路径
    :param file: 对应文件保存路径
    :param obj: 对应待保存的对象
    :return:
    """
    # 创建输出文件夹
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "w", encoding="utf-8") as writer:
        json.dump(obj, writer, indent=2, ensure_ascii=False)


def extract_labels_per_file(file):
    labels = set()
    with open(file, "r", encoding="utf-8") as reader:
        for line in reader:  # 遍历文件中的每一行数据
            line = line.strip()  # 前后空格及不可见字符去除
            obj = json.loads(line)  # json字符串转换为obj对象(字典)
            for entity in obj['entities']:
                label_type = entity['label_type']
                labels.add(label_type)
    return labels


def extract_labels(in_files, out_file):
    """
    构建类别标签的映射mapping
    :param in_files: list[str] 原始数据所在的磁盘路径所组成的list集合
    :param out_file: str 最终的映射mapping希望输出的文件夹路径
    :return:
    """
    # 1. 将类别名称中从原始数据中提取出来
    labels = set()
    for in_file in in_files:
        # 将in_file对应的标签全部合并到一起
        labels = labels.union(extract_labels_per_file(in_file))
    labels = sorted(list(labels))
    print(f"所有的标签列表为:{labels}")

    # 2. 构建成一个字典mapping
    categories = {
        'Other': 0,  # 不属于实体
    }
    for label in labels:
        for prefix in ['B', 'M', 'E', 'S']:
            categories[f'{prefix}-{label}'] = len(categories)
    print(f"映射mapping信息:{categories}")

    # 3. 并保存到磁盘
    save_json(file=out_file, obj=categories)


if __name__ == '__main__':
    extract_labels(
        [
            "./datas/medical/training.txt",
            "./datas/medical/test.json"
        ],
        "./datas/medical/categories.json"
    )
