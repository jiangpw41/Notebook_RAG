核心：深度注意力蒸馏，用于对预训练Transformer模型进行任务无关压缩。2020年2月，Microsoft Research。

论文：[MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)
代码：https://github.com/microsoft/unilm/tree/master/minilm

损失函数：QK和V的注意力矩阵的KL散度之和。不用MSE。

# Abstract
预训练模型（如2018年的BERT）虽然在许多NLP任务中取得成功，但数亿参数使得其很难用于微调和在线服务（现实应用中对延迟和显存容量要求很高）。

因此本文提出一种压缩Transformer预训练模型的方法，deep self-attention distillation（深度自注意力蒸馏），<mark style="background: #FFB86CA6;">创新点在三个方面</mark>：
- 小模型（学生，层数和Embedding维度都小）通过深度模仿大模型（教师）的最后一层的 self-attention 模块进行训练。
- 除了self-attention 模块中原本的 query 和 key 缩放点积外，引入新的 value 之间的缩放点积。
- 此外，我们还表明，引入教师助理（teacher assistant，层数和教师一致，embedding维度和学生一致）也有助于蒸馏。

实验结果表明，不同参数大小的学生模型（单语）的表现优于最先进的基线。特别是，它使用50%的Transformer参数和教师模型的计算，在SQuAD 2.0和几个GLUE基准任务上保持了99%以上的准确性。

# Introduction
预训练LM效果很好，但参数量太大，延迟、显存、微调成本过高。

知识蒸馏Knowledge distillation已经被证明有效，用于将大参数教师模型压缩为小参数的学生模型，缩减尺寸和计算量但保留相似的性能。当时已有很多特定于任务 task-specifical 的蒸馏（现在特定任务上微调再蒸馏），但微调本身很昂贵；而任务无关 task-agnostic 的蒸馏直接模拟预训练行为，其学生模型可以在下游任务上直接微调使用。区别：预训练模型适配下游任务时，先微调再蒸馏还是先蒸馏再微调（后者开销更低）。

先前的工作使用软目标概率 soft target probabilities 进行**掩码语言建模**预测 masked language modeling predictions 或**教师LM的中间表示**，以指导任务无关学生的训练。DistilBERT（Sanh等人，2019）采用软标签蒸馏损失和余弦嵌入损失，并通过**从每两层中选取一层来初始化学生**。但是，学生的每个Transformer层都需要与教师具有相同的架构。TinyBERT (Jiao et al., 2019) 和 MOBILEBERT使用更细粒度的知识，包括Transformer网络的隐藏状态和自我注意力分布，并将这些知识逐层传递给学生模型。为了进行逐层蒸馏，TinyBERT采用统一函数来确定教师和学生层之间的映射，并使用参数矩阵对学生隐藏状态进行线性变换。MOBILEBERT假设教师和学生的层数相同，并引入瓶颈模块以保持其隐藏大小相同。

在这项工作中，我们提出了基于任务无关 Transformer 的LM蒸馏的深度自我注意力蒸馏框架。关键思想是**去深入mimic模仿 self-attention**，这是基于Transformer的教师和学生模型中至关重要的组成部分。具体来说:
- 建议提取教师模型最后一个Transformer层的自我注意力模块。与之前的方法相比，使用最后一个Transformer层的知识而不是执行层到层的知识蒸馏**减轻了教师和学生模型之间的层映射困难**，并且我们的学生模型的**层数可以更灵活**。
- 此外，除了现有作品中使用的注意力分布（即查询和键的缩放点积）之外，我们还引入了自关注模块中**值之间的缩放点乘积**作为新的深度自关注知识。在自我关注值之间使用缩放的点积也可以**将不同维度的表示转换为具有相同维度的关系矩阵**，而无需引入额外的参数来转换学生表示，从而允许学生模型的**任意隐藏维度**。
- 最后，我们表明，引入教师助理（Mirzadeh等人，2019）有助于提炼大型预训练的基于Transformer的模型，而提出的深度自我注意力提炼可以进一步提高性能。
我们对下游NLP任务进行了广泛的实验。实验结果表明，在不同参数大小的学生模型中，我们的单语模型优于最先进的基线。具体来说，从BERTBASE中提取的768个隐藏维度的6层模型速度提高了2.0倍，同时在SQuAD 2.0和几个GLUE基准任务上保持了99%以上的准确性。此外，我们从XLM RBase中提取的多语言模型也以更少的Transformer参数实现了具有竞争力的性能


# 2 Preliminary
介绍Transformer架构尤其是self-attention机制，以及现有的Transformer网络知识蒸馏 knowledge distillation 方法，特别是在将基于大型Transformer的预训练模型蒸馏为小型Transformer模型的情况下。

## 2.1 输入表示
在BERT（Devlin等人，2018）中，WordPiece（Wu等人，2016）将文本标记 tokenize 为子词subword单元。例如，单词“predicted”被拆分为“predictor”和“##ed”，其中“##”表示这些片段属于一个单词。如果输入文本包含多个段，则使用特殊的边界标记\[SEP]来分隔段。在序列的开头，添加一个特殊的标记\[CLS]来获得整个输入的表示。通过对对应的tokens 进行嵌入、绝对位置嵌入和segment嵌入的求和（sum）来计算输入令牌的二维的向量表示 $\{X_{i}\}$，其中每个 $x_i$ 都是一个embedding向量。

## 2.2 Backbone网络: Transformer
Transformer 用于对输入 tokens 的上下文信息进行编码。所有 tokens 的向量表示都被打包为初始状态向量 $H^0 = [x_1,...,x_{|x|}]$，然后堆叠的 Transformer Block就会迭代计算编码向量： $H^l = \text{Transformer}_l(H^{l-1}), l \in [1, L]$，其中l是层数。最终输出是 $H^L = [h_1^L, ..., h_{|x|^L}]$，隐状态向量 $h_i^L$ 被用作 $x_i$ 的上下文表示。每一个Transformer Block都有一个attention子模块和一个全连接前馈子模块（fully connected feed-forward network， FFN），两者用残差，并进行归一化。


## 2.3 Transformer蒸馏
知识蒸馏（Hinton等人，2015；Romero等人，2015）是在具有软标签（soft labels）和大型教师模型T提供的中间表示（intermediate representations）的**转移特征集**（transfer feature set）上训练小型学生模型S。知识提炼被建模为最小化教师特征和学生特征之间的差异
$$Loss_{KD} = \sum_{e \in D}L(f^S(e), f^T(e))$$
其中D是数据集，$f^S(·)$ 表示学生模型的特征，MSE和KL散度通常作为L损失函数。

基于Transformer的LM蒸馏使用下列要素作为特征features，以帮助学生的训练：
- 掩码语言建模预测的软目标概率：例如教师模型对`[MASK]`的预测概率分布为 `{"apple":0.7, "fruit":0.2, "pear":0.1}`，学生模型需拟合这一分布。
- 嵌入层输出
- 自我注意力分布
- 教师模型的每个Transformer层的输出（隐藏状态）

例如，DistillBERT中使用软标签和嵌入层输出。TinyBERT和MOBILEBERT进一步利用了每个Transformer层的自我注意力分布和输出。对于MOBILEBERT，学生需要与老师有相同的层数才能进行逐层蒸馏。此外，还引入了瓶颈和反向瓶颈模块，以保持教师和学生的隐藏大小相同。为了将知识层转移到层，TinyBERT采用了一个统一的函数来映射教师和学生层。由于学生的隐藏大小可能小于其老师，因此参数矩阵用于转换为学生特征。


# 3 Deep Self-Attention Distillation
![[MiniLM_overview.png]]
图1给出了深度自我关注蒸馏的概述。关键思想有三个方面。首先，我们建议通过深度模仿**教师最后一层的自我注意模块**来训练学生，这是Transformer中的重要组成部分。其次，除了在自我注意力模块中执行注意力分布（即查询和键的缩放点积）迁移外，我们还引入了迁移值之间的关系（即值之间的缩放点乘积）来实现更深层次的模仿。此外，我们发现，当教师模型和学生模型之间的尺寸差距较大时，引入教师助理（Mirzadeh等人，2019）也有助于提炼大型预训练的Transformer模型

## 3.1 Self-Attention Distribution Transfer
一些研究表明，预训练的LM的**自我注意力分布捕捉到了丰富的语言信息层次**（Jawahar等人，2019；Clark等人，2019）。迁移（Transfer）自我注意力分布已被用于transformer蒸馏的先前工作中（Jiao等人，2019；Sun等人，2019b；Aguilar等人，2019）。具体来说，我们最小化了教师和学生自注意力分布之间的KL散度：
$$Loss_{AT} = \frac{1}{A_h|x|}\sum_{a=1}^{A_h}\sum_{t=1}^{|x|}D_{KL}(A^T_{L,a,t}||A_{M,a,t}^{S})$$
其中，$|x|$ 和 $A_h$ 分别表示输入序列的长度和注意力头数。L和M分别表示教师和学生的层数，$A_L^T$ 表示教师模型L层block的最后一层的注意力分布（注意力矩阵）。与以往将教师的知识层转移到层的工作不同，我们只使用了教师最后一个Transformer层的注意力map。提取最后一个Transformer层的注意力知识，可以为学生模型的层数提供更大的灵活性，避免了寻找最佳层映射的麻烦。

## 3.2 Self-Attention Value-Relation Transfer
除了注意力分布，我们还建议使用自我注意力模块中的值之间的关系来指导学生的训练。值关系是通过值之间的多头缩放点积计算的。以师生价值关系的KL散度为训练目标。值关系VR计算方式如下$$VR^T_{l,a}=\text{softmax}(\frac{V^T_{l,a} V^{T\text{T}}_{l,a}}{\sqrt{d_k}})$$那么，值关系之间的KL散度计算如下
$$Loss_{VR} = \frac{1}{A_h|x|}\sum_{a=1}^{A_h}\sum_{t=1}^{|x|}D_{KL}(VR^T_{L,a,t}||VR_{M,a,t}^{S})$$

因此最终的损失函数为$$Loss=Loss_{AT}+Loss_{VR}$$
引入价值之间的关系，使学生能够**更深刻地模仿教师的自注意力行为**。此外，使用缩放的点积将不同隐藏维度的向量转换为**相同大小的关系矩阵**，这使我们的学生能够使用更灵活的隐藏维度，并避免引入额外的参数来转换学生的表示

## 3.3 Teacher Assistant
根据Mirzadeh等人（2019）的研究，我们引入了一种教师助理（即中等规模的学生模型），以进一步提高较小学生的模型表现。

假设教师模型的隐状态维度为 $d_h$ ，有L层，学生模型隐状态维度为 $d_h'$ ，有M层。对于较小的学生（$M≤0.5L，d′_h≤0.5d_h$），我们首先使用L层Transformer和 $d′_h$ 隐藏大小将教师提取为教师助理。然后，辅助模型被用作教师来指导最终学生的培训。教师助理的引入弥合了教师和较小学生模型之间的规模差距，有助于提炼基于Transformer的预训练学习模型。此外，将深度自我注意力蒸馏与教师助理相结合，可以为较小的学生模型带来进一步的改进。

教师助理： 本质是教师和学生的过渡，在**层数上和教师保持一致，在隐向量维度上与学生一致**。

## 3.4 与之前工作的比较
![[MiniLM-compare.png]]
表2。比较不同方法（统一从 $BERT_{BASE}$ 蒸馏），隐藏维度768，层数为6。我们比较了任务无关蒸馏模型，不进行任务特定蒸馏和数据增强。我们报告了SQuAD 2.0的F1，以及其他数据集的准确性。DistillBERT的GLUE结果来自Sanh等人（2019）。我们通过微调他们发布的模型3来报告SQuAD 2.0的结果。对于TinyBERT，我们微调了其公共模型的最新版本4，以便进行公平的比较。我们微调实验的结果是每个任务平均运行4次。

MOBILEBERT建议使用一个专门设计的倒置瓶颈模型作为教师，该模型与BERTLARGE的模型大小相同。其他方法利用BERTBASE进行实验。对于用于提炼的知识，我们的方法在自我关注模块中引入了值之间的缩放点积作为新知识，以深度模仿教师的自我关注行为。TinyBERT和MOBILEBERT将教师的知识层层传递给学生。MOBILEBERT假设学生的层数与老师相同。TinyBERT采用统一的策略来确定其图层映射。DistillBERT使用教师的参数初始化学生，因此仍然需要选择教师模型的层。MINILM提取了教师最后一个Transformer层的自我关注知识，这为学生提供了灵活的层数，并减轻了寻找最佳层映射的工作量。DistillBERT和MOBILEBERT的学生隐藏尺寸必须与其老师相同。TinyBERT使用参数矩阵来转换学生隐藏状态。使用值关系允许我们的学生在不引入额外参数的情况下使用任意隐藏大小。

# 4 Experiments
我们在不同参数大小的学生模型中进行了蒸馏实验，并在下游任务上评估了蒸馏模型，包括提取式问答和GLUE基准测试。

## 4.1 Distillation Setup
我们使用 uncased （不区分大小写）版本的 $BERT_{BASE}$ 作为我们的老师。 $BERT_{BASE}$ （Devlin等人，2018）是一款12层Transformer，具有768个隐藏大小和12个注意力头，包含约109M个参数。对于学生模型，注意力分布和价值关系的头数设置为12。我们使用英语维基百科2和BookCorpus（Zhu等人，2015）的文档作为预训练数据，词汇量为30522。最大序列长度为512。我们使用Adam（Kingma&Ba，2015），β1=0.9，β2=0.999。我们使用1024作为批量大小，5e-4作为40万步的峰值学习率，训练了768个隐藏大小的6层学生模型。对于其他架构的学生模型，批处理大小和峰值学习率分别设置为256和3e-4。我们在前4000步中使用线性预热和线性衰减。dropout rate为0.1。重量衰减为0.01。

我们还使用 $BERT_{BASE}$ 大小的内部预训练Transformer模型作为教师模型，并将其提取为具有384个隐藏大小的12层和6层学生模型。对于12层模型，我们使用Adam（Kingma&Ba，2015），β1=0.9，β2=0.98。该模型使用2048作为批量大小，6e-4作为400000步的峰值学习率进行训练。对于6层模型，批大小和峰值学习率分别设置为512和4e-4。其余超参数与上述基于BERT的蒸馏模型相同。

对于多语言MINILM模型的训练，我们使用Adam（Kingma&Ba，2015），β1=0.9，β2=0.999。我们使用256作为批量大小，3e-4作为100万步的峰值学习率来训练12层学生模型。使用512作为批量大小，6e-4作为400000步的峰值学习率来训练6层学生模型。我们使用8个V100 GPU进行混合精度训练，提取学生模型。根据Sun等人（2019a）和Jiao等人（2019）的研究，在具有相同超参数的QNLI训练集上评估推理时间。我们报告了单个P100 GPU上100个批次的平均运行时间。

## 4.2 Downstream Tasks
![[MiniLM_layers.png]]
在之前的语言模型预训练（Devlin等人，2018；Liu等人，2019）和任务无关的预训练语言模型提取（Sanh等人，2019；Jiao等人，2019，Sun等人，2019b）之后，我们在提取式问答和GLUE基准上评估了我们提取的模型。从BERTBASE中提炼出的不同架构的学生模型之间的比较。M和d′h表示学生模型的层数和隐藏维度。TA表示助教。微调结果在4次运行中取平均值。

### Extractive Question Answering
提取式文档：给定一篇文章P，任务是通过预测文章的开始和结束位置来选择文章中连续的文本跨度，以回答问题Q。我们在SQuAD 2.0（Rajpurkar等人，2018）上进行了评估，该标准已成为主要的问答基准。

### GLUE
GLUE通用语言理解评估（GLUE）基准（Wang等人，2019）由九个句子级分类任务组成，包括
- Corpus of Linguistic Acceptability：语言可接受性语料库（CoLA）（Warstadt等人，2018）
- Stanford Sentiment Treebank：斯坦福情感树库（SST）（Socher等人，2013）
- Microsoft Research Paraphrase Corpus：微软研究院短语语料库（MRPC）（Dolan&Brockett，2005）
- Semantic Textual Similarity Benchmark：语义文本相似性基准（STS）（Cer等人，2017）
- Quora Question Pairs：Quora问题对（QQP）（Chen等人，2018）
- Multi-Genre Natural Language Inference：多体裁自然语言推理（MNLI）（Williams等人，2018）
- Question Natural Language Inference问题自然语言推理（QNLI）（Rajpurkar等人，2016）
- Recognizing Textual Entailment识别文本蕴涵（RTE）（Dagan等人，2006；Bar-Haim等人，2006）、Giampiccolo等人，2007；Bentivogli等人，2009）
- Winograd Natural Language Inference：Winograd自然语言推断（WNLI）（Levesque等人，2012）。我们在[CLS]标记之上添加一个线性分类器来预测标签概率。
