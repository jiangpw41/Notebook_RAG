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
