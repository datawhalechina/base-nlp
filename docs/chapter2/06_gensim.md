# 第四节 基于 Gensim 的词向量实战

前面已经学习了多种词向量表示，接下来我们尝试将这些理论转化为可运行的代码。本节将使用 **Gensim** 进行实践，通过简洁的代码示例应用前几节介绍的算法，加深对模型工作原理的理解，并掌握其基本使用方法。

## 一、Gensim 简介

**Gensim (Generate Similar)** 是一个功能强大且高效的Python库，专门用于处理原始的、非结构化的纯文本文档。它内置了多种主流的词向量和主题模型算法，如 Word2Vec、TF-IDF、LSA、LDA 等。

### 1.1 核心概念

使用 Gensim 时，会遇到几个概念：

- **语料库**：这是 Gensim 处理的主要对象，可以简单理解为**训练数据集**。分词后的文档通常表示为 `list[list[str]]`；用于 TF-IDF、LDA 等模型的标准 BoW 语料库是包含稀疏向量的可迭代对象，每篇文档表示为 `[(token_id, frequency), ...]`。例如 `[["我", "爱", "吃", "海参"], ["国足", "惨败", "泰国"]]` 中每个子列表代表一篇独立的文档。
- **词典**：这是一个将词语（token）映射到唯一整数ID的**词汇表**。在使用词袋模型之前，必须先根据整个语料库构建一个词典。
- **向量**：在 Gensim 中，一篇文档最终会被转换成一个数学向量。例如，使用词袋模型时，一篇文档 `["我", "爱", "我"]` 可能会被表示为 `[(0, 2), (1, 1)]`。
- **稀疏向量**：这是 Gensim 为了节省内存而采用的一种高效表示法。对于像 One-Hot 或词袋模型这样维度巨大且绝大多数值为0的向量，Gensim 不会存储所有0。例如，一个词袋向量 `[2, 1, 0, 0, ... , 0]` 会被表示成 `[(0, 2), (1, 1)]`，仅记录**非零项的索引和值**，极大地减少了存储开销。
- **模型**：在 Gensim 中，模型是一个用于实现**向量转换**的算法。例如，`TfidfModel` 可以将一个由词频构成的词袋向量，转换为一个由TF-IDF权重构成的向量。

### 1.2 内置算法与安装

Gensim 几乎涵盖了前面章节中讨论过的所有经典算法。对于基础的权重计算，它提供了 **TF-IDF** (`models.TfidfModel`)；在主题模型与矩阵分解方面，支持 **LSA** (`models.LsiModel`)、**LDA** (`models.LdaModel`) 以及 **NMF** (`models.Nmf`)；而在神经网络词向量领域，它实现了经典的 **Word2Vec** (`models.Word2Vec`)、**FastText** (`models.FastText`) 以及用于段落向量化的 **Doc2Vec** (`models.Doc2Vec`)。

如果还没安装 `gensim`，直接使用如下 `pip` 命令即可安装：

```bash
pip install gensim
```

## 二、Gensim 工作流

> [本节完整代码](https://github.com/datawhalechina/base-llm/blob/main/code/C2/04_gensim.ipynb)

在 Gensim 中，将原始文本转换为TF-IDF或主题模型向量，通常遵循一个标准的三步流程。这个流程是后续应用的基础。

（1）**准备语料**：将原始的文本文档进行分词，并整理成 Gensim 要求的由列表构成的列表 `list[list[str]]` 格式，其中每个子列表代表一篇独立的文档。

（2）**创建词典**：遍历所有分词后的文档，创建一个词典，将每个唯一的词元映射到一个从 0 开始的整数 ID。

（3）**词袋化**：使用创建好的词典，将每一篇文档转换为其稀疏的词袋（BoW）向量。这个向量只记录文档中出现的词的 ID 及其频次，格式为 `[(token_id, frequency), ...]`。

这个最终生成的 **BoW语料库**，就是训练 TF-IDF、LDA 等模型的标准输入。

> 以上三步适用于 TF-IDF、LSA、LDA、NMF 等基于 BoW 的模型；不适用于 Word2Vec/FastText/Doc2Vec 等神经网络词向量模型。后者直接以分词后的句子序列（`list[list[str]]`）为输入，无需词袋化。

```python
import jieba
from gensim import corpora

# Step 1: 准备分词后的语料 (新闻标题)
raw_headlines = [
    "央行降息，刺激股市反弹",
    "球队赢得总决赛冠军，球员表现出色"
]
tokenized_headlines = [jieba.lcut(doc) for doc in raw_headlines]
print(f"分词后语料: {tokenized_headlines}")

# Step 2: 创建词典
dictionary = corpora.Dictionary(tokenized_headlines)
print(f"词典: {dictionary.token2id}")

# Step 3: 转换为 BoW 向量语料库
corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized_headlines]
print(f"BoW语料库: {corpus_bow}")
```

输出如下：

```text
分词后语料: [['央行', '降息', '，', '刺激', '股市', '反弹'], ['球队', '赢得', '总决赛', '冠军', '，', '球员', '表现出色']]
词典: {'刺激': 0, '反弹': 1, '央行': 2, '股市': 3, '降息': 4, '，': 5, '冠军': 6, '总决赛': 7, '球员': 8, '球队': 9, '表现出色': 10, '赢得': 11}
BoW语料库: [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)], [(5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1)]]
```

## 三、TF-IDF与关键词权重

TF-IDF 是衡量一个词在文档中重要性的经典加权方法。下面将继续使用新闻标题的例子，演示如何计算其 TF-IDF 向量。

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
tokenized_headlines = [jieba.lcut(title) for title in headlines]

# 2. 创建词典和 BoW 语料库
dictionary = corpora.Dictionary(tokenized_headlines)
corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized_headlines]

# 3. 训练 TF-IDF 模型
tfidf_model = models.TfidfModel(corpus_bow)

# 4. 将BoW语料库转换为 TF-IDF 向量表示
corpus_tfidf = tfidf_model[corpus_bow]

# 辅助函数：把 (token_id, weight) 转成 (token, weight)，并按权重降序展示
def tfidf_with_words(tfidf_vec, id2word):
    pairs = [(id2word[token_id], weight) for token_id, weight in tfidf_vec]
    return sorted(pairs, key=lambda x: x[1], reverse=True)

# 打印第一篇标题的 TF-IDF 向量
first_tfidf = list(corpus_tfidf)[0]
print("第一篇标题的 TF-IDF 向量:")
print(first_tfidf)
print("第一篇标题的 TF-IDF 向量(带词语):")
print(tfidf_with_words(first_tfidf, dictionary))

# 5. 对新标题应用模型
new_headline = "股市大涨，牛市来了"
new_headline_bow = dictionary.doc2bow(jieba.lcut(new_headline))
new_headline_tfidf = tfidf_model[new_headline_bow]
print("\n新标题的 TF-IDF 向量:")
print(new_headline_tfidf)
```

输出如下：

```bash
第一篇标题的 TF-IDF 向量:
[(0, 0.44066740566370055), (1, 0.44066740566370055), (2, 0.44066740566370055), (3, 0.44066740566370055), (4, 0.44066740566370055), (5, 0.1704734229377651)]
第一篇标题的 TF-IDF 向量(带词语):
[("刺激", 0.44066740566370055), ("反弹", 0.44066740566370055), ("央行", 0.44066740566370055), ("股市", 0.44066740566370055), ("降息", 0.44066740566370055), ("，", 0.1704734229377651)]

新标题的 TF-IDF 向量:
[(3, 0.9326446771245245), (5, 0.360796211497975)]
```

通过输出可以看出，原始的 BoW 向量只包含词频（整数），而 TF-IDF 向量则包含浮点数权重。像“，”这样在多篇文档中都出现的词，其 TF-IDF 权重会较低；而在特定财经新闻中出现的“股市”、“降息”等词，权重会相对较高。这个 TF-IDF 向量后续可用于计算文档相似度或作为机器学习模型的输入特征。需注意，词典外的新词会被忽略，新文本的向量仅由词典中已有词构成。本示例中标点“，”进入了词典并具有非零权重；如不希望其影响权重或相似度，建议在构建词典前移除标点/停用词。此外，新标题的 TF-IDF 仅包含“股市”和“，”两项，这是因为其他词为 OOV 被忽略。

## 四、LDA 与文档主题挖掘

主题模型（如 LDA）能从大量文档中自动发现隐藏的、无监督的主题。它的输入同样是词典和 BoW 语料库。

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
tokenized_headlines = [jieba.lcut(title) for title in headlines]

# 2. 创建词典和 BoW 语料库
dictionary = corpora.Dictionary(tokenized_headlines)
corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized_headlines]

# 3. 训练 LDA 模型 (假设需要发现 2 个主题)
lda_model = models.LdaModel(corpus=corpus_bow, id2word=dictionary, num_topics=2, random_state=100)

# 4. 查看模型发现的主题
print("模型发现的2个主题及其关键词:")
for topic in lda_model.print_topics():
    print(topic)

# 5. 推断新文档的主题分布
new_headline = "巨星詹姆斯获得常规赛MVP"
new_headline_bow = dictionary.doc2bow(jieba.lcut(new_headline))
topic_distribution = lda_model[new_headline_bow]
print(f"\n新标题 '{new_headline}' 的主题分布:")
print(topic_distribution)
```

输出如下：

```text
模型发现的2个主题及其关键词:
(0, '0.045*"，" + 0.040*"公布" + 0.039*"一期" + 0.039*"名单" + 0.039*"足球" + 0.039*"最新" + 0.038*"集训" + 0.038*"国家队" + 0.037*"A股" + 0.037*"市场"')
(1, '0.066*"，" + 0.039*"篮球" + 0.039*"刷新" + 0.039*"历史" + 0.039*"记录" + 0.038*"得分" + 0.038*"巨星" + 0.037*"刺激" + 0.036*"降息" + 0.036*"反弹"')

新标题 '巨星詹姆斯获得常规赛MVP' 的主题分布:
[(0, np.float32(0.27243596)), (1, np.float32(0.72756404))]
```
通过 LDA，不仅可以将一篇文档表示为一个主题概率分布（Gensim 默认以稀疏列表返回；例如 90% 体育、10% 财经），还能清晰地看到每个主题由哪些核心词构成。若新文本在词典中几乎无重叠词（`doc2bow` 为空），推断出的主题分布可能接近均匀（例如 2 个主题时约为 0.5/0.5）。

## 五、Word2Vec 模型实战

与前两者不同，Word2Vec 的输入直接是**分词后的句子列表**。它的目标不是加权或寻找主题，而是根据上下文学习每个词语本身内在的、稠密的语义向量。不过，Word2Vec 训练结束后，神经网络本身通常会被丢弃。因为它的**最终目标**是获得那个高质量的**词向量查询表**，存储在 `model.wv` 属性中。后续所有的应用，都是围绕这个查询表展开的。

### 5.1 模型训练与核心参数

得益于 Gensim 的封装，Word2Vec 的训练代码极其简洁。主要的难点是如何根据语料特点调整超参数（如窗口大小、最小词频等），这些设置直接决定了词向量的质量，下面我们复用前文的新闻标题语料来进行演示。

```python
import jieba
from gensim.models import Word2Vec

# 1. 准备语料
headlines = [
    # 财经
    "央行降息，刺激股市反弹",
    "A股市场持续震荡，投资者需谨慎",
    "理财产品收益率创下新高",
    "证监会发布新规，规范市场交易",
    "创业板指数上涨，科技股领涨大盘",
    "房价调控政策出台，房地产市场降温",
    "全球股市动荡，影响资本市场信心",
    "分析师认为，当前股市风险与机遇并存，市场情绪复杂",

    # 体育
    "球队赢得总决赛冠军，球员表现出色",
    "国家队公布最新一期足球集训名单",
    "篮球巨星刷新历史得分记录",
    "奥运会开幕，中国代表团旗手确定",
    "马拉松比赛圆满结束，选手创造佳绩",
    "电子竞技联赛吸引大量年轻观众",
    "这支球队的每位球员都表现出色",
    "球员转会市场活跃，多支球队积极引援"
]
tokenized_headlines = [jieba.lcut(title) for title in headlines]


# 2. 训练Word2Vec模型
model = Word2Vec(tokenized_headlines, vector_size=50, window=3, min_count=1, sg=1)

```

训练完成后，所有词向量都存储在 `model.wv` 对象中，这是一个 `KeyedVectors` 实例。代码中我们主要用到了 `sentences`、`vector_size`、`window`、`min_count` 和 `sg` 这几个参数，其中：

-   `sentences`: 输入的语料库（对应代码中的 `tokenized_headlines`），必须是 `list[list[str]]` 格式。
-   `vector_size`: 词向量的维度。维度越高，能表达的语义信息越丰富，但计算量也越大。通常在 50-300 之间选择。
-   `window`: 上下文窗口大小。表示在预测一个词时，需要考虑其前后多少个词。
-   `min_count`: **最小词频过滤**。任何在整个语料库中出现次数低于此值的词将被直接忽略。这是非常关键的一步，可以过滤掉大量噪音（如错别字、罕见词），并显著减小模型体积。
-   `sg`: 选择训练算法。`0` 表示 **CBOW** (根据上下文预测中心词)；`1` 表示 **Skip-gram** (根据中心词预测上下文)。

还有几个比较重要但是当前未使用的参数，它们负责模型的优化与采样：

-   `hs`: 选择优化策略。`0` 表示使用 **Negative Sampling** (负采样)；`1` 表示使用 **Hierarchical Softmax**。
-   `negative`: 当 `hs=0` 使用负采样时，为每个正样本随机选择多少个负样本。通常在 5-20 之间。
-   `sample`: **高频词二次重采样阈值**。这是一个控制高频词（如“的”、“是”）被随机跳过的机制，目的是减少它们对模型训练的过多影响，并加快训练速度。

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

输出如下：

```text
与 '股市' 最相似的词: [('名单', 0.3699544370174408), ('央行', 0.3554984927177429), ('年轻', 0.24192844331264496), ('联赛', 0.23673078417778015), ('马拉松', 0.20006775856018066), ('证监会', 0.19390541315078735), ('每位', 0.17580750584602356), ('积极', 0.16841839253902435), ('球队', 0.16585905849933624), ('理财产品', 0.16510605812072754)]

'球队' 和 '球员' 的相似度: -0.1552

'市场' 的向量维度: (50,)
```

### 5.3 模型的持久化

在实际项目中，通常会把训练好的词向量保存下来。如果后续不再进行增量训练，推荐只保存 `KeyedVectors` 对象 (`model.wv`)。因为完整的模型对象包含了哈夫曼树、梯度累积量等仅在训练阶段需要的中间状态，去除这些冗余信息不仅能节省存储空间，更能降低内存占用并提升加载速度，保障线上服务的稳定性。

```python
from gensim.models import KeyedVectors

# 保存词向量到文件
model.wv.save("news_vectors.kv")

# 从文件加载词向量
loaded_wv = KeyedVectors.load("news_vectors.kv")

# 加载后可以执行同样的操作
print(f"\n加载后，'球队' 和 '球员' 的相似度: {loaded_wv.similarity('球队', '球员'):.4f}")
```

输出如下：

```text
加载后，'球队' 和 '球员' 的相似度: -0.1552
```

通过 Gensim，我们就可以非常方便地训练自己的 Word2Vec 模型，并利用它的语义捕捉能力进行相似度计算、语义类比等 NLP 任务。
