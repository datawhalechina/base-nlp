import json
import os

def load_json(file_path):
    """
    从文件加载 JSON 数据。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path, indent=4):
    """
    将数据以格式化的 JSON 形式保存到文件。
    """
    # 确保目录存在
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
