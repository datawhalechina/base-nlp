# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/8/10 16:38
Create User : 19410
Desc :
加载原始文件数据，并将数据拆分为x和y
x就是原始文本对应的token id --> 文本的分词实际上就是采用字(也就是每个字就是一个token)
y就是每个token对应的类别id
    PS:
        x: [bs,t] bs个样本，每个样本对应的token id 内部数据的取值：[0,vocab_size)
        y: [bs,t] bs个样本，每个样本中的每个token对应的类别id 内部数据的取值: [0, 1+4*实体数目)
"""
import json


def t0(file):
    with open(file, "r", encoding="utf-8") as reader:
        for line in reader:  # 遍历文件中的每一行数据
            line = line.strip()  # 前后空格及不可见字符去除
            obj = json.loads(line)  # json字符串转换为obj对象(字典)

            # 获取得到当前数据的原始文本
            text = obj['originalText']
            tokens = list(text)  # 分词

            # 对应的token 类别id
            token_label_names = ['Other'] * len(tokens)
            for entity in obj['entities']:
                label_type = entity['label_type']  # 实体类别名称
                start_pos = entity['start_pos']  # 实体在token中起始位置，包含
                end_pos = entity['end_pos']  # 实体在token中结束位置，不包含

                entity_len = end_pos - start_pos
                if entity_len == 1:
                    token_label_names[start_pos] = f'S-{label_type}'
                elif entity_len > 1:
                    token_label_names[start_pos] = f'B-{label_type}'
                    token_label_names[end_pos - 1] = f'E-{label_type}'
                    for i in range(start_pos + 1, end_pos - 1):
                        token_label_names[i] = f'M-{label_type}'
                else:
                    raise ValueError(f"数据标准异常:{line}")
            print(list(zip(tokens, token_label_names)))
            break


if __name__ == '__main__':
    t0(
        "./datas/medical/training.txt"
    )
