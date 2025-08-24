# 第四节 基于Gensim的词向量实战

前面已经学习了多种词向量表示，接下来尝试将这些理论转化为可运行的代码。本节将使用**Gensim**进行实践，通过简洁的代码示例来实现前几章的核心算法，旨在加深对模型工作原理的理解，并掌握其基本使用方法。

## 一、Gensim 简介

**Gensim (Generate Similar)** 是一个功能强大且高效的Python库，专门用于处理原始的、非结构化的纯文本文档。它内置了多种主流的词向量和主题模型算法，如Word2Vec、TF-IDF、LSA、LDA等。

### 1.1 关键要素

在使用Gensim时，会遇到几个关键要素：

1.  **语料库**：这是Gensim处理的主要对象，可以简单理解为**训练数据集**。分词后的文档通常表示为 `list[list[str]]`；用于 TF-IDF、LDA 等模型的标准 BoW 语料库是包含稀疏向量的可迭代对象，每篇文档表示为 `[(token_id, frequency), ...]`。例如 `[["我", "爱", "吃", "海参"], ["国足", "惨败", "泰国"]]` 中每个子列表代表一篇独立的文档。

2.  **词典**：这是一个将词语（token）映射到唯一整数ID的**词汇表**。在使用词袋模型之前，必须先根据整个语料库构建一个词典。

3.  **向量**：在Gensim中，一篇文档最终会被转换成一个数学向量。例如，使用词袋模型时，一篇文档 `["我", "爱", "我"]` 可能会被表示为 `[(0, 2), (1, 1)]`。

4.  **稀疏向量**：这是Gensim为了节省内存而采用的一种高效表示法。对于像One-Hot或词袋模型这样维度巨大且绝大多数值为0的向量，Gensim不会存储所有0。例如，一个词袋向量 `[2, 1, 0, 0, ... , 0]` 会被表示成 `[(0, 2), (1, 1)]`，仅记录**非零项的索引和值**，极大地减少了存储开销。

5.  **模型**：在Gensim中，模型是一个用于实现**向量转换**的算法。例如，`TfidfModel` 可以将一个由词频构成的词袋向量，转换为一个由TF-IDF权重构成的向量。

### 1.2 内置模型

Gensim几乎涵盖了前面章节中讨论过的所有经典算法：
-   **TF-IDF**: `models.TfidfModel`
-   **主题模型 / 矩阵分解**:
    -   LSA (Latent Semantic Analysis): `models.LsiModel`
    -   LDA (Latent Dirichlet Allocation): `models.LdaModel`
    -   NMF (Non-negative Matrix Factorization): `models.Nmf`
-   **神经网络词向量**:
    -   Word2Vec: `models.Word2Vec`
    -   FastText: `models.FastText`
    -   Doc2Vec: `models.Doc2Vec`

### 1.3 安装gensim

直接使用pip即可：

```bash
pip install gensim
```

## 二、Gensim工作流

在Gensim中，将原始文本转换为TF-IDF或主题模型向量，通常遵循一个标准的三步流程。这个流程是后续应用的基础。

1.  **准备语料 (Tokenization)**：将原始的文本文档进行分词，并整理成Gensim要求的格式——一个由列表构成的列表 `list[list[str]]`，其中每个子列表代表一篇独立的文档。

2.  **创建词典 (Dictionary)**：遍历所有分词后的文档，创建一个词典，将每个唯一的词元（Token）映射到一个从0开始的整数ID。

3.  **词袋化 (Bag-of-Words)**：使用创建好的词典，将每一篇文档转换为其稀疏的词袋（BoW）向量。这个向量只记录文档中出现的词的ID及其频次，格式为 `[(token_id, frequency), ...]`。

这个最终生成的 **BoW语料库**，就是训练TF-IDF、LDA等模型的标准输入。

```python
import jieba
from gensim import corpora

# Step 1: 准备分词后的语料 (新闻标题)
raw_headlines = [
    "央行降息，刺激股市反弹",
    "球队赢得总决赛冠军，球员表现出色"
]
tokenized_headlines = [list(jieba.cut(doc)) for doc in raw_headlines]
print(f"分词后语料: {tokenized_headlines}")

# Step 2: 创建词典
dictionary = corpora.Dictionary(tokenized_headlines)
print(f"词典: {dictionary.token2id}")

# Step 3: 转换为BoW向量语料库
corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized_headlines]
print(f"BoW语料库: {corpus_bow}")
```

示例输出：

```text
分词后语料: [['央行', '降息', '，', '刺激', '股市', '反弹'], ['球队', '赢得', '总决赛', '冠军', '，', '球员', '表现出色']]
词典: {'刺激': 0, '反弹': 1, '央行': 2, '股市': 3, '降息': 4, '，': 5, '冠军': 6, '总决赛': 7, '球员': 8, '球队': 9, '表现出色': 10, '赢得': 11}
BoW语料库: [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)], [(5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1)]]
```

## 三、TF-IDF与关键词权重

TF-IDF是衡量一个词在文档中重要性的经典加权方法。下面将继续使用新闻标题的例子，演示如何计算其TF-IDF向量。

```python
import jieba
from gensim import corpora, models

# 1. 准备语料 (新闻标题，包含财经和体育两个明显主题)
headlines = [
    "央行降息，刺激股市反弹",
    "球队赢得总决赛冠军，球员表现出色",
    "国家队公布最新一期足球集训名单",
    "A股市场持续震荡，投资者需谨慎",
    "篮球巨星刷新历史得分记录",
    "理财产品收益率创下新高"
]
tokenized_headlines = [list(jieba.cut(title)) for title in headlines]

# 2. 创建词典和BoW语料库
dictionary = corpora.Dictionary(tokenized_headlines)
corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized_headlines]

# 3. 训练TF-IDF模型
tfidf_model = models.TfidfModel(corpus_bow)

# 4. 将BoW语料库转换为TF-IDF向量表示
corpus_tfidf = tfidf_model[corpus_bow]

# 打印第一篇标题的TF-IDF向量
print("第一篇标题的TF-IDF向量:")
print(list(corpus_tfidf)[0])

# 5. 对新标题应用模型
new_headline = "股市大涨，牛市来了"
new_headline_bow = dictionary.doc2bow(list(jieba.cut(new_headline)))
new_headline_tfidf = tfidf_model[new_headline_bow]
print("\n新标题的TF-IDF向量:")
print(new_headline_tfidf)
```

输出：

```text
第一篇标题的TF-IDF向量:
[(0, 0.44066740566370055), (1, 0.44066740566370055), (2, 0.44066740566370055), (3, 0.44066740566370055), (4, 0.44066740566370055), (5, 0.1704734229377651)]

新标题的TF-IDF向量:
[(3, 0.9326446771245245), (5, 0.360796211497975)]
```

### 结果分析
- 原始的BoW向量只包含词频（整数），而TF-IDF向量则包含浮点数权重。
- 像“，”这样在多篇文档中都出现的词，其TF-IDF权重会较低；而在特定财经新闻中出现的“股市”、“降息”等词，权重会相对较高。
- 这个TF-IDF向量后续可用于计算文档相似度或作为机器学习模型的输入特征。
 - 词典外的新词（OOV）会被忽略，新文本的向量仅由词典中已有词构成。
 - 本示例中标点“，”进入了词典并具有非零权重；如不希望其影响权重或相似度，建议在构建词典前移除标点/停用词。
 - 你会看到新标题的TF-IDF仅包含“股市”和“，”两项，这是因为其他词为OOV被忽略。

## 四、LDA与文档主题挖掘

主题模型（如LDA）能从大量文档中自动发现隐藏的、无监督的主题。它的输入同样是词典和BoW语料库。

```python
from gensim import corpora, models

# 1. 准备语料
headlines = [
    "央行降息，刺激股市反弹",
    "球队赢得总决赛冠军，球员表现出色",
    "国家队公布最新一期足球集训名单",
    "A股市场持续震荡，投资者需谨慎",
    "篮球巨星刷新历史得分记录",
    "理财产品收益率创下新高"
]
tokenized_headlines = [list(jieba.cut(title)) for title in headlines]

# 2. 创建词典和BoW语料库
dictionary = corpora.Dictionary(tokenized_headlines)
corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized_headlines]

# 3. 训练LDA模型 (假设需要发现2个主题)
lda_model = models.LdaModel(corpus=corpus_bow, id2word=dictionary, num_topics=2, random_state=100)

# 4. 查看模型发现的主题
print("模型发现的2个主题及其关键词:")
for topic in lda_model.print_topics():
    print(topic)

# 5. 推断新文档的主题分布
new_headline = "詹姆斯获得常规赛MVP"
new_headline_bow = dictionary.doc2bow(list(jieba.cut(new_headline)))
topic_distribution = lda_model[new_headline_bow]
print(f"\n新标题 '{new_headline}' 的主题分布:")
print(topic_distribution)
```

输出：

```text
模型发现的2个主题及其关键词:
(0, '0.045*"，" + 0.040*"公布" + 0.039*"一期" + 0.039*"名单" + 0.039*"足球" + 0.039*"最新" + 0.038*"集训" + 0.038*"国家队" + 0.037*"A股" + 0.037*"市场"')
(1, '0.066*"，" + 0.039*"篮球" + 0.039*"刷新" + 0.039*"历史" + 0.039*"记录" + 0.038*"得分" + 0.038*"巨星" + 0.037*"刺激" + 0.036*"降息" + 0.036*"反弹"')

新标题 '詹姆斯获得常规赛MVP' 的主题分布:
[(0, 0.5), (1, 0.5)]
```
通过 LDA，不仅可以将一篇文档表示为一个主题概率分布（Gensim 默认以稀疏列表返回；例如 90% 体育、10% 财经），还能清晰地看到每个主题由哪些核心词构成。注意：若新文本在词典中几乎无重叠词（`doc2bow` 为空），推断出的主题分布可能接近均匀（例如 2 个主题时约为 0.5/0.5）。

## 五、Word2Vec模型实战

与前两者不同，Word2Vec的输入直接是**分词后的句子列表**。它的目标不是加权或寻找主题，而是根据上下文（“分布式假设”）学习每个词语本身内在的、稠密的语义向量。

> 目标与手段
> 需要强调的核心观点是：Word2Vec训练结束后，神经网络本身通常会被丢弃。其**最终目标**是获得那个高质量的**词向量查询表 (Embedding Matrix)**，它存储在 `model.wv` 属性中。后续所有的应用，都是围绕这个查询表展开的。

### 5.1 模型训练与核心参数

训练Word2Vec模型非常直接，关键在于理解其核心参数的设置。继续使用新闻标题的例子。

```python
from gensim.models import Word2Vec

# 1. 准备语料
headlines = [
    "央行降息，刺激股市反弹",
    "球队赢得总决赛冠军，球员表现出色",
    "国家队公布最新一期足球集训名单",
    "A股市场持续震荡，投资者需谨慎",
    "篮球巨星刷新历史得分记录",
    "理财产品收益率创下新高"
]
tokenized_headlines = [list(jieba.cut(title)) for title in headlines]


# 2. 训练Word2Vec模型 (核心参数的解释见下方)
model = Word2Vec(tokenized_headlines, vector_size=50, window=3, min_count=1, sg=1)

# 训练完成后，所有词向量都存储在 model.wv 对象中
# model.wv 是一个 KeyedVectors 实例
```

#### 核心参数解析
理解以下几个核心参数至关重要：
-   `sentences`: 输入的语料库，必须是 `list[list[str]]` 格式。
-   `vector_size`: 词向量的维度。维度越高，能表达的语义信息越丰富，但计算量也越大。通常在50-300之间选择。
-   `window`: 上下文窗口大小。表示在预测一个词时，需要考虑其前后多少个词。
-   `min_count`: **最小词频过滤**。任何在整个语料库中出现次数低于此值的词将被直接忽略。这是非常关键的一步，可以过滤掉大量噪音（如错别字、罕见词），并显著减小模型体积。
-   `sg`: 选择训练算法。`0` 表示 **CBOW** (根据上下文预测中心词)；`1` 表示 **Skip-gram** (根据中心词预测上下文)。
-   `hs`: 选择优化策略。`0` 表示使用 **Negative Sampling** (负采样)；`1` 表示使用 **Hierarchical Softmax**。当 `hs=0` 时，下面的 `negative` 参数才会生效。
-   `negative`: 当使用负采样时，为每个正样本随机选择多少个负样本。通常在5-20之间。
-   `sample`: **高频词二次重采样阈值**。这是一个控制高频词（如“的”、“是”）被随机跳过的机制，目的是减少它们对模型训练的过多影响，并加快训练速度。值越小，高频词被跳过的概率越大。

### 5.2 使用词向量

模型训练完成后，所有的操作都围绕 `model.wv` 展开，用于探索词语间的语义关系。小语料示例下，相似度数值通常较低且不稳定，仅作演示参考。

```python
# 1. 寻找最相似的词
# 在小语料上，结果可能不完美，但能体现出模型学习到了主题内的关联
similar_to_market = model.wv.most_similar('股市')
print(f"与 '股市' 最相似的词: {similar_to_market}")

# 2. 计算两个词的余弦相似度
similarity = model.wv.similarity('球队', '球员')
print(f"\n'球队' 和 '球员' 的相似度: {similarity:.4f}")

# 3. 获取一个词的向量
market_vector = model.wv['市场']
print(f"\n'市场' 的向量维度: {market_vector.shape}")
```

输出：

```text
'球队' 和 '球员' 的相似度: 0.0959
'市场' 的向量维度: (50,)
```

### 5.3 模型的持久化

在实际项目中，通常会把训练好的词向量保存下来，避免重复训练。推荐只保存 `KeyedVectors` 对象，它更轻量、高效。

```python
from gensim.models import KeyedVectors

# 保存词向量到文件
model.wv.save("news_vectors.kv")

# 从文件加载词向量
loaded_wv = KeyedVectors.load("news_vectors.kv")

# 加载后可以执行同样的操作
print(f"\n加载后，'球队' 和 '球员' 的相似度: {loaded_wv.similarity('球队', '球员'):.4f}")
```

示例输出：

```text
加载后，'球队' 和 '球员' 的相似度: 0.0959
```

通过Gensim，可以非常方便地训练自己的Word2Vec模型，并利用其强大的语义捕捉能力进行相似度计算、语义类比等高级NLP任务。
