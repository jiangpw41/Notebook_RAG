题目：Pre-training of Deep Bidirectional Encoder Representations from Transformers（预训练深度双向编码表示）
论文：https://arxiv.org/pdf/1810.04805

BERT，2018年，由Google提出。2017年Google提出Transformer原型（Encoder-Decoder架构）后，OpenAI和Google分别于2018年提出了Decoder-only架构的GPT和Encoder-only的BERT。前者成为了LLMs的代表，后者成为了embedding模型的代码。
## 1 概述

与之前 word embedding + 双向RNN 以及最近 Decoder-only两类模型（ELMo, Peters等人，2018a；GPT, Radford等人，2018）不同，BERT旨在<mark style="background: #FFB86CA6;">通过在所有层中对前后文进行联合 conditioning ，从未标记的文本中预训练深度双向表示</mark>。正因此，预训练的BERT模型可以<mark style="background: #ADCCFFA6;">通过一个额外的输出层进行微调</mark>，为各种任务（如问答和语言推理）创建最先进的模型，而无需对特定任务的架构进行大量修改。

BERT在概念上简单，在实证上强大。它在11个自然语言处理任务上获得了最新的最先进的结果，包括将GLUE评分提高到80.5%（绝对提高7.7%），将MultiNLI准确率提高到86.7%（绝对提高4.6%），将SQuAD v1.1问答测试F1提高到93.2（绝对提高1.5分），将SQuAD v2.0测试F1提高至83.1（绝对提高5.1分）。

## 2 创新点

在BERT以前，NLP任务按阶段可以分为两类：
- Feature-based：基于特征的。2017年Transformer出现之前，NLP的主要范式为静态word embedding + 双向RNN，例如早期的Word2Vec和后期的ELMo。静态词嵌入仅仅提供对词的特征表示，具体的应用依托特定于任务的RNN。
- Fine-tuning微调：如GPT，在Feature-based的基础上前进一步，用微调方式解决了传统方法适配成本高的问题。核心思想是最小限度使用特定于任务的参数，通过简单微调就可以应用于下游任务。

但目前的 Fine-tuning 方法还不够，虽然通过微调减少了适配成本，但其放弃了传统 Feature-based 方法中双向 RNN 所带来的双向注意力。这种单向注意力限制有对两类任务非常有害：
- Sentence-level句子级任务：如语言推理、释义，特点是通过整体分析句子来预测句子之间的关系。
- Token-level词元级：问答、命名实体识别NER，特点需要输出细粒度token的任务。

因此BERT在保留预训练的基础上，仅保留Transformers的Encoder部分，采用双向注意力。

## 架构

<p align="middle">
  <img src="images/BERT.png" width=60%><br>
</p>
Bert 模型结构采用了Transformer 的编码器（L是layer数，H是hidden states的维数，A是多头自注意力的Head数）：
- BERT_BASE (L=12, H=768, A=12, Total Parameters=110M)
- BERT_LARGE (L=24, H=1024, A=16, Total Parameters=340M)
- 输入/输出表示：
  - 使用WordPiece Embedding（2016年提出）和3w个token的词表
  - 能无歧义地表示单个句子sentence和句子对：
    - sentence：以[CLS]作为第一个token（CLS是classification的简称），该token的最后一个隐状态被用于分类任务中作为整个句子的表示(The final hidden state corresponding to this token)。
    - sentence pair：两种方法区分AB
      - 使用[SEP]作为句子间的分隔符，比如用于分割问题和答案两个sentence
      - 向每个token都add一个习得的embedding，用以指示该token是属于句子A还是句子B。
需要对输入进行三个层面的embedding相加。
<p align="left">
  <img src="images/BERT输入表示.png" width=100%><br>
  图2:BERT的输入embedding由三个embedding相加而成：token本身的embedding、A句还是B句的Embedding、位置Embedding。
</p>
## 训练目标
<p align="left">
  <img src="images/BERT架构.png" width=100%><br>
  图1:BERT的总体预训练和微调过程。除了<strong>输出层</strong>，预训练和微调都使用了相同的架构。相同的预训练模型参数用于初始化不同下游任务的模型。在微调过程中，所有参数都会被微调。[CLS]是添加在每个输入示例前面的特殊符号，[SEP]是一个特殊的分隔符标记（例如分隔问题/答案）。
</p>
Bert 的预训练任务包括两个需要双向推理的任务：
- MLM (Masked Language Modeling) ：**掩码语言建模**，受到完形填空任务（Cloze task, ["Cloze Procedure": A New Tool For Measuring Readability](https://gwern.net/doc/psychology/writing/1953-taylor.pdf)）的启发，从输入中随机屏蔽Mask一些token，并基于上下文预来预测屏蔽词。
- NSP (Next Sentence Prediction)：**下个句子预测**，二分类任务，输入句子对，输出二分类标签。最初目的是弥补MLM仅关注token级预测的不足，使模型学习句子间的连贯性。随着 NLP 模型的发展, 一些研究发现去除 NSP 对某些模型的性能影响不大, 例如: Roberta, Xlnet, 和 Deberta 等。因为这些模型的底层双向结构已经足够强大, 能在没有 NSP 的情况下理解句子间的复杂关系.
