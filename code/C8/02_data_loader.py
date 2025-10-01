import json


def process_ner_data(file_path):
    """
    加载NER数据，处理第一行以生成BMES标签，
    并打印结果用于验证。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # 作为演示，我们只处理文件的第一行
        first_line = f.readline()
        if not first_line:
            print("文件为空。")
            return

        data = json.loads(first_line)

        # 1. 按字切分文本 (Tokenization)
        text = data['text']
        tokens = list(text)

        # 2. 初始化标签列表，全部标记为 "O" (Outside)
        labels = ['O'] * len(tokens)

        # 3. 根据实体标注信息，应用 BMES 标签
        for entity in data.get('entities', []):
            entity_type = entity['type']
            start_idx = entity['start_idx']
            # 原始end_idx不包含，我们将其-1转换为包含模式，便于处理
            end_idx = entity['end_idx'] - 1
            entity_len = end_idx - start_idx + 1

            if entity_len == 1:
                # 单字实体
                labels[start_idx] = f'S-{entity_type}'
            else:
                # 多字实体
                labels[start_idx] = f'B-{entity_type}'
                labels[end_idx] = f'E-{entity_type}'
                for i in range(start_idx + 1, end_idx):
                    labels[i] = f'M-{entity_type}'

        # 4. 打印Token和对应的标签，用于检查逻辑正确性
        print("文本Tokens及其生成的BMES标签:")
        for token, label in zip(tokens, labels):
            print(f"{token}\t{label}")


if __name__ == '__main__':
    train_file = './data/CMeEE-V2_train.json'
    print(f"--- 正在处理文件的第一行: {train_file} ---")
    process_ner_data(train_file)
