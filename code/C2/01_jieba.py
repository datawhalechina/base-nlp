import jieba

text1 = "我在梦里收到清华大学录取通知书"
seg_list = jieba.lcut(text1, cut_all=False) # cut_all=False 表示精确模式
print(seg_list)

# 未加载词典前的错误分词
text2 = "九头虫让奔波儿灞把唐僧师徒除掉"
print(f"精准模式: {jieba.lcut(text2, cut_all=False)}")

# 加载自定义词典
jieba.load_userdict("./user_dict.txt") 
print(f"加载词典后: {jieba.lcut(text2, cut_all=False)}")

text3 = "我在Boss直聘找工作"

# 开启HMM（默认）
seg_list_hmm = jieba.lcut(text3, HMM=True)
print(f"HMM开启: {seg_list_hmm}")

# 关闭HMM
seg_list_no_hmm = jieba.lcut(text3, HMM=False)
print(f"HMM关闭: {seg_list_no_hmm}")

import jieba.posseg as pseg

words = pseg.lcut(text2, HMM=False)
print(f"默认词性输出: {words}")

jieba.load_userdict("./user_pos_dict.txt")

dic_words = pseg.lcut(text2, HMM=False)
print(f"加载词性词典后: {dic_words}")