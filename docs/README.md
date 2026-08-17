<div align="center">
  <h1>Base LLM | 从 NLP 到 LLM 的算法全栈教程</h1>
</div>

<div align='center'>
  <img src="./logo.svg" alt="base-llm Logo" width="70%">
</div>

<div align="center">
  <h2>大模型前世今生</h2>
  <p><em>Base LLM is all you need</em></p>
</div>

<div align="center">
  <img src="https://img.shields.io/github/stars/datawhalechina/base-llm?style=for-the-badge&logo=github&color=ff6b6b" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/forks/datawhalechina/base-llm?style=for-the-badge&logo=github&color=4ecdc4" alt="GitHub forks"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
</div>

<div align="center">
  <a href="https://github.com/datawhalechina/base-llm/releases">
    <img src="https://img.shields.io/github/v/release/datawhalechina/base-llm?style=for-the-badge&logo=github&color=brightgreen" alt="GitHub release"/>
  <a href="https://datawhalechina.github.io/base-llm/">
    <img src="https://img.shields.io/badge/📖_在线阅读-立即开始-success?style=for-the-badge&logoColor=white" alt="在线阅读"/>
  <a href="https://github.com/datawhalechina">
    <img src="https://img.shields.io/badge/💬_讨论交流-加入我们-purple?style=for-the-badge&logoColor=white" alt="讨论交流"/>
</div>

> **注意：本项目可能进行大幅调整当前暂不接受 Pull Request。**
> 如果您有任何建议或发现任何问题，欢迎通过 [Issue](https://github.com/datawhalechina/base-nlp/issues) 进行反馈。

## 📖 项目简介

本项目是一个**从传统自然语言处理（NLP）到大语言模型（LLM）的全栈式学习教程**，旨在为开发者提供一条从理论入门到工程实战的清晰路径。

在 LLM 爆发的今天，许多开发者直接上手调用 API 或微调大模型，却往往忽视了底层的自然语言处理（NLP）基础。本项目主张 **"Base LLM is all you need"**，通过系统性地回顾 NLP 发展历程中的核心技术——从早期的词向量、循环神经网络（RNN），到变革性的 Transformer 架构，再到如今的 BERT、GPT 及 Llama 系列大模型——帮助读者构建坚实的技术护城河。

**核心内容覆盖：**

1.  **NLP 理论基石**：深入浅出地讲解分词、词向量（Word2Vec）、RNN/LSTM 等经典算法。
2.  **Transformer 架构**：剖析 Attention 机制，详解 Encoder-Decoder 架构，奠定大模型认知基础。
3.  **预训练语言模型**：全面覆盖 BERT、GPT、T5 等里程碑式模型的设计与应用。
4.  **大模型进阶实战**：从零手搓 Llama2 模型，掌握参数高效微调（PEFT/LoRA）、RLHF 等前沿技术。
5.  **工程化落地**：涵盖模型量化、推理加速、Docker 容器化及服务部署的全流程实战。
6.  **大模型安全与多模态**：探索模型安全挑战、伦理问题，以及图文多模态模型的前沿技术。

## 🌟 项目意义

随着人工智能技术的飞速发展，掌握大语言模型已成为 AI 工程师的必备技能。然而，市面上的教程往往存在断层：要么过于偏重学术理论，晦涩难懂；要么仅停留在 API 调用的应用层，缺乏底层原理的支撑。

本项目致力于填补这一空白，通过**理论与代码并重**的方式，帮助开发者：

*   **打通知识脉络**：理解技术演进的内在逻辑（如：为什么从 RNN 发展到 Transformer？）。
*   **掌握核心原理**：不仅会用，更懂其"所以然"，具备排查复杂问题和优化模型结构的能力。
*   **学习代码演进**：采用“提出问题-迭代重构”的教学模式，展示从简易脚本到工业级框架的演变过程，培养真正的工程化思维。
*   **提升工程能力**：通过 NER、文本分类及 LLM 微调部署等实战项目，积累生产环境下的开发经验。
*   **拓宽技术视野**：探索大模型安全、多模态等前沿领域，紧跟 AI 技术发展的最新趋势。

## 🎯 项目受众

**本项目适合以下人群学习：**

*   🎓 **在校学生**：希望系统学习 NLP 知识，为科研或求职打下基础。
*   💻 **AI 算法工程师**：需要从传统机器学习/深度学习转型到大模型领域的开发者。
*   🤔 **LLM 爱好者**：对大模型底层原理感兴趣，希望通过手写代码深入理解大模型架构运行机制的开发者。
*   🔬 **研究人员**：需要快速回顾 NLP 经典算法或寻找基线代码实现的学者。

**前置要求：**

*   🟢 **Python 基础**：熟练掌握 Python 语法及常用数据结构。
*   🔵 **PyTorch 框架**：具备基本的 PyTorch 深度学习框架使用经验。
*   🟠 **深度学习基础**：理解神经网络基本原理、反向传播机制及模型训练的基础流程。
*   🟡 **数学基础**：了解基本的线性代数、概率论及梯度下降等深度学习概念。

## ✨ 项目亮点

1.  **体系化进阶路径**：从基础的文本表示到 RLHF 和量化技术，内容层层递进。
2.  **手写核心代码**：拒绝"调包侠"，带领读者手写 Llama2、Transformer 等核心架构代码。
3.  **实战导向**：包含文本分类、命名实体识别（NER）、私有数据微调 Qwen2.5 等多个完整项目。
4.  **全流程覆盖**：不仅教你怎么训练，还教你怎么用 Docker 和 FastAPI 将模型部署成服务。
5.  **图文并茂**：配合大量图解，将抽象的算法原理可视化，降低学习门槛。
6.  **直观易懂**：尽量弱化复杂的数学公式推导，让数学基础薄弱的读者也能轻松掌握算法原理。

## 内容大纲

### 第一部分：理论篇
- **第1章：NLP简介**
    - [x] [NLP 概述](./chapter1/01_nlp_intro.md)
    - [x] [环境准备](./chapter1/02_preparation.md)
- **第2章：文本表示与词向量**
    - [x] [初级分词技术](./chapter2/03_tokenization.md)
    - [x] [词向量表示](./chapter2/04_word_vector.md)
    - [x] [从主题模型到 Word2Vec](./chapter2/05_Word2Vec.md)
    - [x] [基于 Gensim 的词向量实战](./chapter2/06_gensim.md)
- **第3章：循环神经网络**
    - [x] [循环神经网络](./chapter3/08_RNN.md)
    - [x] [LSTM 与 GRU](./chapter3/09_LSTM&GRU.md)
- **第4章：注意力机制与Transformer**
    - [x] [Seq2Seq 架构](./chapter4/10_seq2seq.md)
    - [x] [注意力机制](./chapter4/11_attention.md)
    - [x] [深入解析 Transformer](./chapter4/12_transformer.md)
- **第5章：预训练模型**
    - [x] [BERT 结构及应用](./chapter5/13_Bert.md)
    - [x] [GPT 结构及应用](./chapter5/14_GPT.md)
    - [x] [T5 结构及应用](./chapter5/15_T5.md)
    - [x] [Hugging Face 生态与核心库](./chapter5/16_HuggingFace.md)
- **第6章：深入大模型架构**
    - [x] [手搓一个大模型](./chapter6/17_handcraft_llama2.md)
    - [x] [MOE 架构解析](./chapter6/18_MoE.md)
    - [x] [手撕大模型生成策略](./chapter6/19_text_generation.md)
    - [x] [上下文学习与提示词技术](./chapter6/20_in_context_learning.md)

### 第二部分：实战篇
- **第1章：文本分类**
    - [x] [文本分类简单实现](./chapter7/01_text_classification.md)
    - [x] [基于 LSTM 的文本分类](./chapter7/02_lstm_text_classification.md)
    - [x] [微调 BERT 模型进行文本分类](./chapter7/03_bert_text_classification.md)
- **第2章：命名实体识别**
    - [x] [命名实体识别概要](./chapter8/01_named_entity_recognition.md)
    - [x] [NER 项目的数据处理](./chapter8/02_data_processing.md)
    - [x] [模型构建、训练与推理](./chapter8/03_model_building_and_training.md)
    - [x] [模型的推理与优化](./chapter8/04_evaluation_and_prediction.md)

### 第三部分：微调量化篇
- **第1章：参数高效微调**
    - [x] [PEFT 技术综述](./chapter11/01_PEFT.md)
    - [x] [LoRA 方法详解](./chapter11/02_lora.md)
    - [x] [基于 peft 库的 LoRA 实战](./chapter11/03_peft_lora.md)
    - [x] [Qwen2.5 微调私有数据](./chapter11/04_qwen2.5_qlora.md)
- **第2章：高级微调技术**
    - [x] [RLHF 技术详解](./chapter12/01_rlhf.md)
    - [x] [LLaMA-Factory RLHF（DPO）实战](./chapter12/02_llama_factory.md)
- **第3章：大模型训练与量化**
    - [x] [模型量化实战](./chapter13/01_quantization.md)
    - [x] [Deepspeed 框架介绍](./chapter13/02_deepspeed.md)

### 第四部分：应用部署篇
- **第1章：模型服务部署**
    - [x] [FastAPI 模型部署实战](./chapter14/01_fastapi.md)
    - [x] [云服务器模型部署实战](./chapter14/02_uv_linux.md)
    - [x] [使用 Docker Compose 部署模型服务](./chapter14/03_docker_deploy.md)
- **第2章：自动化与性能优化**
    - [x] [搭建 Jenkins CI/CD 自动化部署流水线](./chapter15/02_Jenkins.md)

### 第五部分：大模型安全
- **第 1 章：安全全景与威胁建模**
    - [x] [大模型安全总览](./docs/chapter16/01_LLM_safety_overview.md)
    - [x] [威胁建模及风险分析](./docs/chapter16/02_threat_modeling_analysis.md)
- **第 2 章：安全工程：对齐与架构设计**
    - [ ] [行为对齐工程]()
    - [ ] [安全架构设计]()

### 第六部分：多模态前沿
- **第 1 章：认识多模态边界**
    - [x] [多模态概述](./chapter19/01_multimodal_definition.md)
    - [x] [图文多模态](./chapter19/02_ViT_CLIP.md)
- **第 2 章：视觉问答**
    - [x] [BLIP-2 与 LLaVA](./chapter20/01_blip2_llava.md)
    - [x] [原生统一架构](./chapter20/02_unified_architecture.md)
    - [x] [从 0 训练简化版 Omni 模型](./chapter20/03_simplified_omni.md)

### Extra-chapter
    - []()
    

## 致谢

**核心贡献者**

- [dalvqw-项目负责人](https://github.com/FutureUnreal)（项目发起人与主要贡献者）

### 特别感谢
- 感谢 [@Sm1les](https://github.com/Sm1les) 对本项目的帮助与支持
- 感谢所有为本项目做出贡献的开发者们
- 感谢开源社区提供的优秀工具和框架支持
- 特别感谢以下为教程做出贡献的开发者！

[![Contributors](https://contrib.rocks/image?repo=datawhalechina/base-nlp)](https://github.com/datawhalechina/base-nlp/graphs/contributors)

*Made with [contrib.rocks](https://contrib.rocks).*

## 参与贡献

- 发现问题请提交 Issue。

## Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=datawhalechina/base-nlp&type=Date)](https://star-history.dera.page/#datawhalechina/base-nlp&Date)

<div align="center">
  <p>如果这个项目对你有帮助，请给我们一个 ⭐️</p>
  <p>让更多人发现这个项目（护食？发来！）</p>
</div>

## 关注我们

<div align=center>
  <p>扫描下方二维码关注公众号：Datawhale</p>
    <img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## 许可证

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
