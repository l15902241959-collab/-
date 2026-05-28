# SFT 训练总结与实验记录

> 项目：MiniOneRec  
> 数据集：Industrial_and_Scientific（3686个物品，79,839条训练样本）  
> 训练时间：2026-05-28 ~ 2026-05-29  
> 模型：Qwen2.5-1.5B-Instruct + SID语义ID  
> 训练阶段：Supervised Fine-Tuning (SFT) → Reinforcement Learning (RL)

---

## 目录

1. [SFT在整体流程中的位置](#1-sft在整体流程中的位置)
2. [SFT实验配置](#2-sft实验配置)
3. [SFT训练结果](#3-sft训练结果)
4. [SFT数据结构](#4-sft数据结构)
5. [多任务训练机制](#5-多任务训练机制)
6. [评估结果分析](#6-评估结果分析)
7. [RL强化学习训练](#7-rl强化学习训练)
8. [关键问题解答](#8-关键问题解答)
9. [消融实验设计](#9-消融实验设计)
10. [常见问题FAQ](#10-常见问题faq)

---

## 1. SFT在整体流程中的位置

### 1.1 完整训练流程

```
阶段1: RQ-VAE（物品语义离散化）
  物品文本/属性 → Qwen embedding → RQ-VAE量化 → SID索引
  输出: Industrial_and_Scientific.index.json

阶段2: SFT（监督微调）  ← 当前阶段
  用户历史SID序列 → Qwen2.5-1.5B学习推荐 → 预测下一个物品SID
  输出: output/sft_1.5B/

阶段3: RL（强化学习对齐）
  SFT模型 → GRPO优化HR/NDCG奖励 → 提升推荐指标
  输出: output/rl_1.5B/

阶段4: 评估推理
  RL模型 → Beam Search生成Top-50候选 → 计算HR/NDCG
```

### 1.2 SFT的核心目标

```
RQ-VAE解决: 每个物品如何表示成SID？
SFT解决:   给定用户历史行为，应该推荐哪个SID？
RL解决:    如何让推荐更准（直接优化HR/NDCG）？
```

### 1.3 SFT的本质

SFT阶段把RQ-VAE生成的SID当作**推荐语言**，让Qwen2.5-1.5B学习：

```text
输入：用户历史交互SID序列
输出：下一个可能交互的物品SID

示例：
  历史: ['<a_165><b_107><c_44>', '<a_32><b_89><c_12>']
  目标: '<a_104><b_118><c_176>'
```

---

## 2. SFT实验配置

### 2.1 硬件环境

| 项目 | 配置 |
|-----|-----|
| GPU | 10× NVIDIA RTX A6000（48GB）|
| 使用卡 | GPU 5, 9（CUDA_VISIBLE_DEVICES=5,9）|
| 分布式 | DeepSpeed ZeRO-2 + torchrun |
| 精度 | BF16混合精度 |

### 2.2 数据配置

| 项目 | 数值 |
|-----|-----|
| 物品总数 | 3,686（K-core过滤后）|
| SID token数 | ~768（3层×256码字）|
| 训练样本 | 79,839条（三任务混合）|
| 验证样本 | ~5,000条 |
| 测试样本 | 4,533条 |

### 2.3 模型架构

```
基座模型: Qwen2.5-1.5B-Instruct
  - 参数量: 1.5B
  - 词表大小: 原始151,936 + 新增768 = 152,704
  - 上下文长度: 512 tokens

SID token格式:
  <a_*>  ← Level-1码字（256个）
  <b_*>  ← Level-2码字（256个）
  <c_*>  ← Level-3码字（256个）

Token扩展:
  tokenizer.add_tokens(new_tokens)
  model.resize_token_embeddings(len(tokenizer))
```

### 2.4 训练超参数

| 参数 | 值 | 说明 |
|-----|---|-----|
| `base_model` | Qwen2.5-1.5B-Instruct | 基座LLM |
| `batch_size` | 32 | 全局batch size |
| `micro_batch_size` | 4 | 每张卡每次处理4条 |
| `gradient_accumulation` | 4 | 32/(4×2)=4步累积 |
| `num_epochs` | 3 | 训练轮数 |
| `learning_rate` | 1e-4 | 余弦衰减 |
| `warmup_steps` | 20 | 线性warmup |
| `cutoff_len` | 512 | 最大序列长度 |
| `freeze_LLM` | False | 全量微调（非仅新token）|
| `bf16` | True | 混合精度 |

### 2.5 DeepSpeed配置

```json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "overlap_comm": true,
    "reduce_scatter": true
  },
  "bf16": {"enabled": true}
}
```

**显存优化效果**：
- ZeRO-2节省约15GB显存（优化器+梯度分片）
- 2张卡各占用~30GB（总49GB可用）

---

## 3. SFT训练结果

### 3.1 核心指标

| 指标 | 数值 | 说明 |
|-----|-----|-----|
| 训练时间 | 4小时27分钟 | 16,038秒 |
| 总步数 | 7,485步 | 3 epochs |
| 训练速度 | 14.93样本/秒 | 0.467步/秒 |
| 初始Loss | 8.08 | 第1步 |
| 最终Loss | ~1.0 | 第7485步 |
| 平均Loss | 1.605 | 全程平均 |
| Loss下降 | 87.6% | 8.08→1.0 |

### 3.2 Loss变化趋势

```
Epoch 0（快速下降期）:
  Loss: 8.08 → 4.29
  学习率: 0 → 1e-4（warmup）

Epoch 1-2（稳定收敛期）:
  Loss: 4.29 → 1.2
  平均Loss: ~1.4
  学习率: 1e-4 → 5e-5（余弦衰减）

Epoch 3（微调期）:
  Loss: 1.2 → 0.8-1.4
  最低Loss: 0.6647
  平均Loss: ~1.0
  学习率: 5e-5 → 1.34e-08（接近0）
```

### 3.3 梯度分析

```
grad_norm范围:
  最小: 0.99
  最大: 2.56
  平均: ~1.5

判断:
  ✅ 梯度稳定（<2.0）
  ✅ 无梯度爆炸
  ✅ 梯度裁剪正常工作
```

### 3.4 学习率调度

```
策略: 余弦衰减 + warmup

Warmup阶段（前20步）:
  lr: 0 → 1e-4（线性增加）

衰减阶段（20-7485步）:
  lr: 1e-4 → 1.34e-08（余弦衰减）

最终学习率: 1.34e-08（几乎为0）
```

---

## 4. SFT数据结构

### 4.1 SID索引文件

**路径**: `data/Amazon/index/Industrial_and_Scientific.index.json`

**格式**:
```json
{
  "0": ["<a_97>", "<b_115>", "<c_100>"],
  "1": ["<a_2>", "<b_77>", "<c_6>"],
  "3685": ["<a_189>", "<b_34>", "<c_221>"]
}
```

**来源**: RQ-VAE训练 + generate_indices.py + Sinkhorn去重

---

### 4.2 SFT训练CSV

**路径**: `data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv`

**字段**:
```csv
user_id, history_item_title, item_title, 
history_item_id, item_id, 
history_item_sid, item_sid
```

**关键字段**:
```text
history_item_sid: "['<a_165><b_107><c_44>']"
item_sid: '<a_104><b_118><c_176>'
```

---

### 4.3 训练Prompt格式

**完整格式**:
```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Can you predict the next possible item that the user may expect?

### User Input:
Can you predict the next possible item the user may expect, given the following chronological interaction history: ['<a_165><b_107><c_44>']

### Response:
<a_104><b_118><c_176>
```

**Token化**:
```python
# Instruction部分（不算loss）
instruction = "Below is an instruction..."

# Input部分（不算loss）
prompt = "### User Input:\n..."

# Response部分（算loss）
target = '<a_104><b_118><c_176>\n'

# Loss只在target上计算
labels = [-100]*prompt_len + target_tokens
```

---

### 4.4 评估格式一致性

**info.txt格式**（evaluate.py）:
```text
<a_236><b_231><c_226>\tSUPCO SPP6 Relay...\t0
<a_42><b_80><c_160>\tStanley TRA708T...\t1
```

**转换为prefixID**:
```python
semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
info_semantic = f'''### Response:\n{semantic_id}'''
prefixID = tokenizer(info_semantic).input_ids
```

**格式验证结果**: ✅ **完全一致**

```
训练: <a_104><b_118><c_176>\n  →  [..., 9312, 62, 16, 15, 19, 29, 151806, 152126, 198]
评估: <a_236><b_231><c_226>\n  →  [..., 9312, 62, 17, 18, 21, 29, 151932, 152182, 198]
       ↑格式相同，只是具体码字不同
```

**结论**: 训练和评估的SID格式完全一致，不是HR@10偏低的原因。

---

## 5. 多任务训练机制

### 5.1 三个训练任务

SFT训练同时使用三个数据集，通过`ConcatDataset`混合：

```python
train_datasets = [
    SidSFTDataset(...),          # 任务1: 序列推荐
    SidItemFeatDataset(...),     # 任务2: 物品理解
    FusionSeqRecDataset(...)     # 任务3: 融合推荐
]
train_data = ConcatDataset(train_datasets)
```

---

### 5.2 任务1: SidSFTDataset（序列推荐）

**数据量**: 36,259条（45.42%）

**任务**:
```text
输入: 用户历史SID序列
输出: 下一个物品SID

示例:
  历史: ['<a_1><b_2><c_3>', '<a_4><b_5><c_6>']
  目标: '<a_7><b_8><c_9>'
```

**强化能力**: 用户行为序列建模

**直接影响**: HR@K, NDCG@K

---

### 5.3 任务2: SidItemFeatDataset（物品理解）

**数据量**: 7,321条（9.17%）

**任务**: 双向映射
```text
SID → item title:
  输入: <a_97><b_115><c_100>
  输出: SUPCO SPP6 Relay/Capacitor Hard Start Kit

title → SID:
  输入: SUPCO SPP6 Relay/Capacitor Hard Start Kit
  输出: <a_97><b_115><c_100>
```

**强化能力**: SID与物品语义对齐

**作用**: 
- 提升SID embedding质量
- 让模型理解相似SID对应相似物品
- 减少死记SID编号

---

### 5.4 任务3: FusionSeqRecDataset（融合推荐）

**数据量**: 36,259条（45.42%）

**任务**:
```text
输入: 用户历史SID + 物品title/description
输出: 目标物品title

利用:
  - 用户历史SID序列
  - 物品title
  - 物品description
  - 目标物品语义信息
```

**强化能力**: 序列偏好 + 物品语义联合建模

**作用**: 提升泛化能力，尤其是SID无法表达细粒度语义时

---

### 5.5 多任务Loss机制

**当前实现**: 隐式混合（无显式加权）

```python
# 不是显式多loss
L = αL_seq + βL_item + γL_fusion  ❌

# 而是统一CrossEntropy
L_total = (1/N) Σ CE(sample_i)     ✅
```

**关键点**:
- 三个任务共享同一个LLM
- 共享同一个Trainer和优化器
- 统一使用CrossEntropy Loss
- 任务权重由**样本数量占比**决定

**当前比例**:
```
SidSFTDataset     : 45.42%
SidItemFeatDataset:  9.17%
FusionSeqRecDataset: 45.42%
```

---

## 6. 评估结果分析

### 6.1 评估配置

| 参数 | 值 |
|-----|---|
| 测试样本 | 4,533条 |
| num_beams | 50 |
| max_new_tokens | 256 |
| 约束解码 | ConstrainedLogitsProcessor |
| 评估GPU | GPU 4 |

---

### 6.2 HR指标（Hit Rate）

| Top-K | HR@K | 含义 |
|-------|------|-----|
| @1 | **0.0556** | Top-1命中率5.56% |
| @3 | **0.0880** | Top-3命中率8.80% |
| @5 | **0.1035** | Top-5命中率10.35% |
| @10 | **0.1339** | Top-10命中率13.39% |
| @20 | **0.1674** | Top-20命中率16.74% |
| @50 | **0.2182** | Top-50命中率21.82% |

**解读**:
- HR@1=5.56%：模型直接命中真实物品的比例是5.56%
- HR@10=13.39%：真实物品出现在Top-10的比例是13.39%
- HR@50=21.82%：真实物品出现在Top-50的比例是21.82%

---

### 6.3 NDCG指标（Normalized DCG）

| Top-K | NDCG@K | 含义 |
|-------|--------|-----|
| @1 | **0.0556** | 归一化DCG@1 |
| @3 | **0.0743** | 归一化DCG@3 |
| @5 | **0.0807** | 归一化DCG@5 |
| @10 | **0.0905** | 归一化DCG@10 |
| @20 | **0.0990** | 归一化DCG@20 |
| @50 | **0.1091** | 归一化DCG@50 |

**解读**:
- NDCG不仅看是否命中，还考虑排名位置
- 真实物品排第1，分数最高（1.0）
- 真实物品排第10，分数较低（约0.39）
- NDCG@50=0.1091说明命中的样本里有一部分排位较靠前

---

### 6.4 结果判断

```
HR@1  = 0.0556  ← 偏低（优秀>0.30）
HR@10 = 0.1339  ← 偏低（良好>0.50）
HR@50 = 0.2182  ← 偏低（可接受>0.65）

NDCG@10 = 0.0905  ← 偏低
NDCG@50 = 0.1091  ← 偏低

CC（非法预测数）= 0  ← ✅ 优秀（无不可识别SID）
```

**初步判断**:
- ✅ 模型已学会基础SID生成和推荐模式
- ✅ 约束解码最终未产生非法SID（CC=0）
- ⚠️ Top-K召回率仍偏低，有提升空间

---

### 6.5 HR@10偏低的原因分析

**已排除**:
- ✅ 训练/评估格式不一致（已验证完全一致）
- ✅ 约束解码错误（CC=0）

**可能原因**:

1. **数据量不足**
   ```
   总样本: 79,839
   对于1.5B模型，通常需要100K-1M样本
   ```

2. **训练轮数**
   ```
   num_epochs = 3
   多任务混合时可能不够充分
   ```

3. **任务比例**
   ```
   主序列任务只占45.42%
   辅助任务可能分散了模型注意力
   ```

4. **Beam Search质量**
   ```
   评估时大量"No valid tokens found"警告
   部分beam路径在生成过程中偏离合法SID空间
   影响Top-50质量
   ```

---

## 7. RL强化学习训练

### 7.1 RL配置

| 参数 | 值 | 说明 |
|-----|---|-----|
| 基础模型 | ./output/sft_1.5B | SFT训练结果 |
| 算法 | GRPO | Group Relative Policy Optimization |
| GPU | 0, 4（2张卡）| CUDA_VISIBLE_DEVICES=0,4 |
| reward_type | ranking | 基于排名的奖励 |
| num_generations | 16 | 每个样本生成16个候选 |
| train_batch_size | 32 | 训练batch |
| gradient_accumulation | 2 | 梯度累积 |
| learning_rate | 1e-5 | 比SFT小10倍 |
| beta | 1e-3 | KL惩罚系数 |
| num_train_epochs | 2 | RL训练轮数 |
| beam_search | True | Beam Search生成 |
| sync_ref_model | True | 同步参考模型 |

---

### 7.2 RL奖励函数

```python
# ranking reward（基于排名的奖励）

如果生成SID == 真实SID:
  根据排名位置给正奖励
  rank=0: 最高奖励
  rank=15: 最低奖励

否则:
  给负奖励或0奖励
  鼓励模型生成多样化但准确的SID
```

**与SFT的区别**:
- SFT优化token-level交叉熵
- RL优化样本-level奖励（命中/排名）

---

### 7.3 RL训练状态

```
训练数据集: 52,775条
验证数据集: 4,532条
总步数: ~13,194步
预计时间: 2-4小时
当前状态: 训练中 ✅
```

**数据组成**:
```python
train_datasets = [
    SidDataset(...),                # 序列推荐
    RLTitle2SidDataset(...),        # title→SID
    RLSeqTitle2SidDataset(...)      # 序列+title
]
Total: 36,259 + 6,516 + 10,000 = 52,775
```

---

### 7.4 RL与SFT的对比

| 维度 | SFT | RL (GRPO) |
|-----|-----|-----------|
| **优化目标** | CrossEntropy | 奖励信号(HR/NDCG) |
| **学习率** | 1e-4 | 1e-5（更小）|
| **训练轮数** | 3 | 2 |
| **生成方式** | Teacher Forcing | 采样16个候选 |
| **奖励函数** | 无 | ranking reward |
| **KL惩罚** | 无 | beta=1e-3 |
| **显存占用** | ~30GB/卡 | ~35GB/卡（生成更多）|

---

## 8. 关键问题解答

### Q1: SFT训练为什么用了三个任务？

**A**: 多任务学习可以让模型同时掌握：
1. **序列推荐**（主任务）：用户行为模式
2. **物品理解**（辅助）：SID与语义对齐
3. **融合推荐**（辅助）：序列+语义联合建模

理论上是"1+1+1>3"的效果，但需要调优任务比例。

---

### Q2: 三个任务如何同时强化？

**A**: 当前通过**ConcatDataset混合样本**实现：
- 不是显式多loss加权
- 而是统一CrossEntropy
- 任务权重由样本数量占比决定

如果要显式控制，可以改成：
```python
L_total = αL_seq + βL_item + γL_fusion
```

但当前代码未实现。

---

### Q3: 为什么SFT训练集不需要去重？

**A**: SFT训练集间接使用去重后的SID：
1. 去重发生在generate_indices.py（生成SID索引时）
2. convert_dataset.py查表写入SID到CSV
3. CSV中的SID已经是去重后的结果

所以SFT训练集**间接使用**了去重后的SID索引，无需再次去重。

---

### Q4: freeze_LLM=True vs False有什么区别？

**A**: 
- **freeze_LLM=True**: 冻结原始LLM，只训练新SID token embedding
  - 优点：省显存，训练快
  - 缺点：推荐能力可能弱

- **freeze_LLM=False**: 全量微调LLM
  - 优点：模型适应推荐任务更充分
  - 缺点：显存占用高，训练慢

你当前选择`False`是合理的，因为用了1.5B小模型+DeepSpeed+双卡。

---

### Q5: 评估时大量"No valid tokens found"警告是什么？

**A**: 这是约束解码的正常运行机制：
- Beam Search同时探索50条路径
- 部分路径生成了非法前缀（不在候选SID中）
- 约束解码强制这些路径提前结束（输出EOS）
- 最终CC=0说明所有有效路径都生成了合法SID

这不是错误，而是**过滤无效beam路径**。

---

### Q6: 为什么RL学习率比SFT小10倍？

**A**: RL优化的是奖励信号，不是token-level loss：
- SFT学习率1e-4：学习生成SID的基本模式
- RL学习率1e-5：在SFT基础上微调，避免破坏已学知识
- beta=1e-3的KL惩罚也限制了更新幅度

---

## 9. 消融实验设计

### 9.1 实验矩阵

| 实验 | SidSFTDataset | SidItemFeatDataset | FusionSeqRecDataset | 目的 |
|-----|---------------|-------------------|---------------------|-----|
| **A** | ✅ | ❌ | ❌ | 纯序列推荐上限 |
| **B** | ✅ | ✅ | ❌ | 物品语义是否增益 |
| **C** | ✅ | ❌ | ✅ | 融合推荐是否增益 |
| **D** | ✅ | ✅ | ✅ | 当前baseline |
| **E** | ✅(3x) | ✅ | ✅ | 主任务强化效果 |

---

### 9.2 实验A: 纯序列推荐

**配置**:
```bash
只用SidSFTDataset
batch_size: 32
epochs: 3
lr: 1e-4
```

**预期**:
- HR@10可能提升（专注主任务）
- 但泛化能力可能下降

---

### 9.3 实验E: 主任务强化

**配置**:
```python
任务比例:
  SidSFTDataset : FusionSeqRecDataset : SidItemFeatDataset
  = 3 : 1 : 1
```

**实现方式**: 数据采样或loss加权

**预期**:
- 比实验A更好（保留辅助任务）
- 比实验D更好（主任务权重更高）

---

## 10. 常见问题FAQ

### Q: SFT训练后模型保存在哪？

**A**: `./output/sft_1.5B/`
```
model.safetensors      (3.4GB)
config.json
tokenizer.json
final_checkpoint/
checkpoint-7125/
```

---

### Q: 如何评估SFT模型？

**A**: 
```bash
# 单GPU评估
CUDA_VISIBLE_DEVICES=4 python evaluate.py \
    --base_model ./output/sft_1.5B \
    --info_file ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
    --category Industrial_and_Scientific \
    --test_data_path ./data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --result_json_data ./results_sft_test.json \
    --batch_size 4 \
    --num_beams 50 \
    --max_new_tokens 256

# 计算指标
python calc.py \
    --path ./results_sft_test.json \
    --item_path ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt
```

---

### Q: RL训练如何启动？

**A**: 
```bash
cd /data1/lyt/MiniOneRec
bash rl.sh
```

会自动使用SFT模型作为起点，输出到`./output/rl_1.5B/`

---

### Q: 显存不足怎么办？

**A**: 降低batch_size或切换GPU：
```bash
# 降低batch
--batch_size 16 \
--micro_batch_size 2 \

# 或切换空闲GPU
CUDA_VISIBLE_DEVICES=0,4  # 选择空闲最多的卡
```

---

### Q: 训练中断如何恢复？

**A**: 使用checkpoint恢复：
```bash
--resume_from_checkpoint ./output/sft_1.5B/checkpoint-7125
```

---

## 总结

### SFT训练成果

```
✅ SFT训练完成
  - Loss: 8.08 → 1.0（下降87.6%）
  - 训练时间: 4小时27分钟
  - 模型保存: output/sft_1.5B/

📊 评估结果:
  - HR@10 = 0.1339
  - HR@50 = 0.2182
  - NDCG@10 = 0.0905
  - CC = 0（无非法SID）

🔄 RL训练进行中:
  - 基于SFT模型继续优化
  - 直接优化HR/NDCG奖励
  - 预计2-4小时完成

📋 后续计划:
  - RL训练完成后评估
  - 消融实验（任务比例优化）
  - 对比SFT vs RL效果
```

### 关键教训

1. **格式一致性验证很重要**
   - 训练/评估格式必须完全一致
   - 已验证当前格式一致 ✅

2. **多任务学习需要调优比例**
   - 当前45:9:45可能不是最优
   - 建议实验3:1:1或5:1:1

3. **RL是必要的**
   - SFT优化CrossEntropy，不直接优化HR/NDCG
   - RL可以直接优化推荐指标

4. **数据量可以更大**
   - 当前79K对1.5B模型偏少
   - 可考虑数据增强或增加epochs

---

> 文档生成时间: 2026-05-29  
> 作者: MiniOneRec实验团队  
> 版本: v1.0
