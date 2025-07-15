
all-MiniLM-L6-v2项目旨在使用**自监督**对**比学习目标**在非常大的句子级数据集上训练句子嵌入模型。我们使用了预训练的 nreimers/MiniLM-L6-H384-ucased 模型，并在1B句子对数据集上进行了微调。我们使用对比学习目标：<mark style="background: #FFB86CA6;">给定一对句子中的一个句子，模型应该预测在一组随机抽样的其他句子中，哪一个在我们的数据集中实际与它配对</mark>。

这个项目是  [Train the Best Sentence Embedding Model Ever with 1B Training Pairs](https://discuss.huggingface.co/t/train-the-best-sentence-embedding-model-ever-with-1b-training-pairs/7354)（用1B训练对训练有史以来最好的句子嵌入模型） 项目的一部分。受益于高效的硬件基础设施来运行该项目：TPU-v3-8（128GB的显存），以及谷歌Flax、JAX和云团队成员对高效深度学习框架的干预。

## 1 模型来源
all-MiniLM-L6-v2由Sentence Transformers蒸馏训练，适用来自微软的MiniLM方法。Sentence Transformers (aka [SBERT](https://www.sbert.net) )  最初由 UKP实验室（Ubiquitous Knowledge Processing Lab）开发‌。该实验室隶属于德国达姆施塔特工业大学（Technische Universität Darmstadt）。该库致力于获取、使用、训练先进的embedding和reranker模型。

| embedding维度 | 默认窗口           | usage          |
| ----------- | -------------- | -------------- |
| 384         | 256 word piece | 检索、聚类或句子相似性任务。 |
- all-MiniLM-L6-v2：在nreimers/MiniLM-L6-H384-ucased 基础上用1B句子对数据集进行微调。
- nreimers/MiniLM-L6-H384-ucased：microsoft/MiniLM-L12-H384-uncased的6层版本（仅保留偶数层），仅保留偶数层。[nreimers](https://huggingface.co/nreimers)自己搞的,  "We distill the teacher model into 12-layer and 6-layer models with 384 hidden size using the same corpora. The 12x384 model is used as the teacher assistant to train the 6x384 model."
- microsoft/MiniLM-L12-H384-uncased：distilled model from the paper "[MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)".MiniLM 是微软亚洲研究院（Microsoft Research Asia）开发的一种轻量级的语言模型，旨在以较小的参数量和计算成本实现与大型语言模型（如 BERT）相当的性能。它是基于 Transformer 架构的预训练模型，通过深度自注意力蒸馏（Deep Self-Attention Distillation）等技术进行压缩和优化，使其能够在资源受限的环境下高效运行。
- UNILM：microsoft/MiniLM-L12-H384-uncased蒸馏自该模型，与$BERT_{base}$ 尺寸一致。

## 2 截断逻辑
transformers库中tokenization类的truncation参数
1. **`max_length: 256`**
    - 设定分词后的最大长度（包括特殊标记如 `[CLS]`、`[SEP]`）。
    - 如果输入超过该长度，会被截断。
2. **`stride: 0`**
    - 截断时的重叠步长（用于滑动窗口场景，如问答任务）。
    - `0` 表示无重叠（默认）。例如，若设为 `10`，则截断时会保留前一个窗口的末尾 10 个 token 作为下一个窗口的开头。
3. **`strategy: 'longest_first'`**
    - 截断策略：
        - `'longest_first'`：优先截断较长的部分（例如，对于文本对 `[文本A, 文本B]`，会先截断两者中更长的部分）。
        - `'only_first'`/`'only_second'`：仅截断第一个或第二个部分（适用于问答或句子对任务）。
4. **`direction: 'right'`**
    - 截断方向：
        - `'right'`：从右侧（末尾）截断（默认）。
        - `'left'`：从左侧（开头）截断（例如保留文本末尾更重要的信息）。

## 3 论文要点
论文：[MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)
代码：https://github.com/microsoft/unilm/tree/master/minilm
核心：深度注意力蒸馏，用于对预训练Transformer模型进行任务无关压缩。2020年2月，Microsoft Research。

损失函数：QK注意力矩阵和V注意力矩阵两个KL散度之和。不用MSE。

预训练模型（如2018年的BERT）虽然在许多NLP任务中取得成功，但数亿参数使得其很难用于微调和在线服务（现实应用中对延迟和显存容量要求很高）。因此本文提出一种压缩Transformer预训练模型的方法，deep self-attention distillation（深度自注意力蒸馏），<mark style="background: #FFB86CA6;">创新点在三个方面</mark>：
- 小模型（学生，层数和Embedding维度都小）通过深度模仿大模型（教师）的最后一层的 self-attention 模块进行训练。
- 除了self-attention 模块中原本的 query 和 key 缩放点积外，引入新的 value 之间的缩放点积。
- 此外，引入教师助理（teacher assistant，层数和教师一致，embedding维度和学生一致）也有助于蒸馏。


## 4 训练数据情况

### 4.1 预训练
MiniLM仅在下面两份预训练数据语料上使用蒸馏
- Wikipedia：version enwiki-20181101
- BookCorpus：来自论文Aligning books and movies: Towards story-like visual explanations by watching movies and reading books.


### 4.2 微调
数据清单见：https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
我们使用对比目标对模型进行微调。形式上，我们对批处理中的每个可能的句子对计算余弦相似度。然后，我们通过与真对进行比较来应用交叉熵损失。

我们使用多个数据集的连接来微调我们的模型。句子对的总数超过10亿个句子。我们在给定加权概率的情况下对每个数据集进行采样，该配置在data_config.json文件中有详细说明。