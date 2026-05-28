# Q1(1): Data Selection Methods for Instruction-Tuning in 数据科学领域

**参考论文：** *A Survey on Data Selection for Language Models* (Albalak et al., 2024, TMLR)
**章节：** Data Selection for Instruction-Tuning and Multitask Training

---

## 概述

该 Survey 将 instruction-tuning 的数据选择方法分为三大类：**基于质量**、**基于多样性** 和 **基于重要性**。在数据科学与数学建模领域，agent 轨迹涉及多轮推理、代码执行和错误处理，以下三种方法被认为最为有效。

---

## 方法 1：IFD — Instruction Following Difficulty（指令跟随难度）

**论文：** *From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning* (arXiv:2308.12032)

### 核心思想
IFD 通过比较两个概率来量化一条指令对模型来说有多"难"跟从：
- **有条件回答分数 s_θ(A|Q)：** 给定指令时模型生成答案的对数概率
- **无条件回答分数 s_θ(A)：** 无指令时模型生成答案的对数概率（衡量答案文本本身的流畅性）

**IFD 公式：**
```
IFD(Q, A) = s_θ(A|Q) / s_θ(A)
```

### 为什么适合数据科学和数学领域

数据分析 agent 轨迹包含复杂的推理链：
1. 理解用户的数据分析问题
2. 编写 Python/pandas 代码
3. 执行代码并解读输出
4. 处理错误并修正方案

**高 IFD 分数** 意味着指令真正引导模型产生了针对性的回答——这在数学推理和数据分析中至关重要，因为答案高度依赖问题上下文。反之，如果无论指令如何，答案都很通用（低 IFD），则该轨迹训练价值不大。

### 在轨迹数据上的实现
- **阶段 1 — 热身：** 在小规模多样化子集上训练（对指令 embedding 做 K-Means → 100 个聚类 × 每个 10 个样本 = 1,000 条轨迹，1 个 epoch）。
- **阶段 2 — 评分：** 用热身后的模型对所有 ~12k 轨迹计算 IFD 分数。
- **阶段 3 — 筛选：** 保留 **中高 IFD 分数** 的轨迹，过滤掉过低（太简单）和过高（可能有噪声）的分数。

**关键发现：** 中等 IFD 分数的样本下游表现最好。

---

## 方法 2：InstructMining — 多指标质量评估

**论文：** *Instruction Mining: When Data Mining Meets Large Language Model Finetuning*
(Cao et al., 2023; arXiv:2307.06290)

### 核心思想
InstructMining 使用经典的**数据挖掘技术**，通过 9 个人工设计+模型计算的指标来预测模型在某个数据集上的微调 loss，而无需实际微调模型。使用多元线性回归模型预测期望推理 loss：

```
log L(M_ft, D_eval) ≈ β₀ + Σ βᵢ·Iᵢ(D)
```

### 9 个指标

| 指标 | 描述 | 对数据科学的意义 |
|------|------|-----------------|
| 输入长度 (Len_in) | 指令的 token 数 | 更长的指令 = 更复杂的数据任务 |
| 输出长度 (Len_out) | 回答的 token 数 | 更长的推理 = 更丰富的轨迹 |
| 奖励分数 (Rew) | 来自奖励模型 | 与人类偏好相关 |
| 困惑度 (PPL) | 指数化平均负对数似然 | 低 PPL = 自然流畅的轨迹 |
| MTLD | 词汇多样性度量 | 多样的词汇 = 更丰富的解释 |
| KNN-i | embedding 空间中到第 i 近邻的距离 | 相对于其他轨迹的新颖性 |
| UniEval-自然度 | 回答听起来是否自然 | 代码+自然语言的混合质量 |
| UniEval-连贯性 | 回答是否是有效的延续 | 逻辑推理链的质量 |
| UniEval-可理解性 | 回答是否易于理解 | 分析解释的清晰度 |

### 为什么适合数据科学和数学领域

数据分析 agent 轨迹有多个质量维度：
- **代码质量**（Python/pandas 代码是否正确、可运行？）
- **推理质量**（数学推导是否合理？）
- **输出质量**（最终答案是否准确回答了问题？）
- **可视化质量**（图表生成是否恰当？）

InstructMining 的多指标方法比任何单一指标都能更好地捕捉这些不同方面。**奖励分数** 和 **UniEval 分数** 被证明是最显著的预测因子——它们与数据分析任务中的轨迹质量直接相关。

### 实现步骤
1. 使用轻量模型为每条 ~12k 轨迹**计算指标**
2. 在小规模标注子集上**训练回归模型**（原文用了 ~129 个子集）
3. **预测 loss** —— loss 越低 = 质量越高
4. **选择 top-K 轨迹**（原文只用 2.5% 的数据就达到最优，我们目标 ~16%）

---

## 方法 3：K-Center Greedy — 多样性驱动核心集选择

**论文：** 多种核心集选择方法（Sener & Savarese, 2018; Survey 中引用）

### 核心思想
K-Center Greedy 通过解决 **k-中心问题** 来选择多样化子集：给定 embedding 空间中的一组点，选择 k 个中心使得**任意点到其最近中心的最大距离最小化**。简单说，就是迭代地选**离所有已选样本最远**的那个点。

**算法：**
1. 使用轻量编码器为所有指令轨迹计算 embedding
2. 随机初始化一个样本
3. 每一步选**与当前已选集的最小距离最大**的样本
4. 重复直到达到目标子集大小（训练 2000，验证 500）

### 为什么适合数据科学和数学领域

DataMind-12K 数据集涵盖多样的领域和任务类型：

| 领域类别 | 示例 |
|---------|------|
| 数据清洗 | 缺失值处理、异常检测 |
| 统计分析 | 假设检验、描述性统计 |
| 数据可视化 | Matplotlib/seaborn 绘图、图表选择 |
| 机器学习 | 模型训练、评估、特征工程 |
| 数学建模 | 方程拟合、优化、仿真 |
| 数据转换 | 聚合、合并、重塑、ETL |
| 报告生成 | 洞察提取、自然语言总结 |

随机采样容易过度代表常见任务类型而低估稀有但重要的类型。K-Center Greedy 确保**各类数据科学子领域的均衡覆盖**，使训练集包含完整的数据分析挑战谱系。

### 与质量方法集成
K-Center Greedy 最适合作为质量筛选后的**第二道过滤器**：
1. 先用 IFD 或 InstructMining 过滤低质量轨迹
2. 然后在剩余池上用 K-Center Greedy 确保多样性
3. 这种 **质量 → 多样性** 管线产生紧凑而全面的训练集

---

## 总结

上述三种方法覆盖了 Survey 论文识别的两个关键维度：**质量** 和 **多样性**。

- **IFD** 通过衡量指令与回答之间的依赖强度来评估单条轨迹的训练价值
- **InstructMining** 结合多个 NLP 指标来预测微调收益，提供更全面的质量评估
- **K-Center Greedy** 确保所选子集覆盖数据科学任务的完整谱系，避免任务类型偏差

在实际的 DataMind-12K 处理中，我们组合使用这些方法：先用 IFD 代理评分保留高质量样本，再用 K-Center Greedy 做多样性筛选，最终得到 2000 训练 + 500 验证的结构化数据集。

---

## 参考文献

1. Albalak, A., Elazar, Y., Xie, S. M., Longpre, S., Lambert, N., Wang, X., ... &
   Yang, W. W. (2024). A Survey on Data Selection for Language Models. *TMLR*.
   arXiv:2402.16827
2. Li, M., et al. (2023). From Quantity to Quality: Boosting LLM Performance with
   Self-Guided Data Selection for Instruction Tuning. arXiv:2308.12032
3. Cao, Y., et al. (2023). Instruction Mining: When Data Mining Meets Large Language
   Model Finetuning. arXiv:2307.06290
4. Sener, O., & Savarese, S. (2018). Active Learning for Convolutional Neural
   Networks: A Core-Set Approach. *ICLR*.
5. Qiao, S., et al. (2025). Scaling Generalist Data-Analytic Agents. arXiv:2509.25084
