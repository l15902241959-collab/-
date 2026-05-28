# RQ-VAE 训练总结与实验记录

> 项目：MiniOneRec  
> 数据集：Industrial_and_Scientific（3686个物品）  
> 训练时间：2026-05-28  
> 模型：RQ-VAE（残差量化变分自编码器）

---

## 目录

1. [实验配置](#1-实验配置)
2. [训练结果](#2-训练结果)
3. [核心问题解答](#3-核心问题解答)
4. [消融实验设计](#4-消融实验设计)
5. [关键细节总结](#5-关键细节总结)
6. [常见问题FAQ](#6-常见问题faq)

---

## 1. 实验配置

### 1.1 硬件环境

| 项目 | 配置 |
|-----|-----|
| GPU | 10× NVIDIA RTX A6000（48GB）|
| 使用卡 | GPU 4（CUDA_VISIBLE_DEVICES=4）|
| 空闲显存 | ~31GB |

### 1.2 数据配置

| 项目 | 数值 |
|-----|-----|
| 物品数量 | 3,686 |
| 嵌入维度 | 2560（Qwen文本编码器提取）|
| 数据来源 | `data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy` |
| 嵌入范围 | [-18.5, 17.7]，均值0.009 |

### 1.3 网络架构

```
Encoder（压缩）:
  2560 → Linear+ReLU → 2048
  2048 → Linear+ReLU → 1024
  1024 → Linear+ReLU → 512
  512  → Linear+ReLU → 256
  256  → Linear+ReLU → 128
  128  → Linear       → 64   ← 最后层无激活
  64   → Linear       → 32   ← 量化潜空间

ResidualVectorQuantizer（残差量化）:
  Level-1: codebook 256×32  （主方向）
  Level-2: codebook 256×32  （Level-1误差）
  Level-3: codebook 256×32  （Level-2误差）

Decoder（重建）:
  32   → Linear+ReLU → 64
  64   → Linear+ReLU → 128
  128  → Linear+ReLU → 256
  256  → Linear+ReLU → 512
  512  → Linear+ReLU → 1024
  1024 → Linear+ReLU → 2048
  2048 → Linear       → 2560
```

### 1.4 超参数

| 参数 | 值 | 说明 |
|-----|---|-----|
| `num_emb_list` | [256, 256, 256] | 3层量化，总词表256³=1677万 |
| `e_dim` | 32 | 量化空间维度，区分度√(2/32)=0.25 |
| `layers` | [2048,1024,512,256,128,64] | Encoder隐藏层，渐进压缩 |
| `lr` | 1e-3 | AdamW学习率 |
| `batch_size` | 20480 | 全量batch（3686<20480）|
| `epochs` | 10000 | 训练轮数 |
| `beta` | 0.25 | commitment loss权重（VQ-VAE推荐值）|
| `kmeans_init` | True | K-means++初始化codebook |
| `sk_epsilons` | [0.0, 0.0, 0.0] | 训练时不启用Sinkhorn |
| `quant_loss_weight` | 1.0 | RVQ损失权重 |

---

## 2. 训练结果

### 2.1 核心指标

| 指标 | 训练时 | 生成SID后 | 说明 |
|-----|-------|---------|-----|
| 最佳碰撞率 | 9.52% (epoch 9049) | - | 训练过程最低 |
| 最终碰撞率 | 11.67% (epoch 9999) | **0.30%** | 去重后大幅降低 |
| 唯一SID数 | 3300 | **3675** | 只有11个物品仍相同 |
| 最佳Loss | 0.212 | - | 历史最低 |
| 重建Loss | - | 0.07 | 重建质量好 |
| 量化Loss | - | 0.14 | 可接受 |

### 2.2 训练过程

#### 碰撞率变化

```
Epoch 49:   99.78%  → 刚开始，几乎全碰撞
Epoch 499:  85.70%  → 10%进度，学会基础区分
Epoch 999:  34.81%  → 快速下降期
Epoch 2999: ~20%    → 大部分已区分
Epoch 9049: 9.52%   → 最佳状态 ✅
Epoch 9999: 11.67%  → 最终状态（略有波动）
```

#### Loss变化

```
Epoch 0:    Total=1.0, Recon=0.90  → 网络初始化，什么都不会
Epoch 100:  Total=0.5, Recon=0.25  → 快速学习重建能力
Epoch 1000: Total=0.3, Recon=0.12  → 重建稳定，学习量化
Epoch 5000: Total=0.25,Recon=0.09  → 两者都接近收敛
Epoch 9999: Total=0.21,Recon=0.07  → 训练完成
```

#### 训练耗时

- 每epoch：约0.28秒
- 总耗时：10000 × 0.28s ≈ 47分钟
- 原因：数据量小（3686），单batch可容纳全部

### 2.3 SID生成结果

```
文件：Industrial_and_Scientific.index.json
总物品：3686
唯一SID：3675
碰撞率：0.30%

示例SID：
  物品0: ['<a_97>', '<b_115>', '<c_100>']
  物品1: ['<a_2>', '<b_77>', '<c_6>']
  物品2: ['<a_132>', '<b_76>', '<c_85>']
```

**去重机制效果：**
- 训练时碰撞率：9.52%
- 第一轮生成（argmin）：9.52%
- 第二轮生成（Sinkhorn）：0.74%
- 第三轮生成（Sinkhorn）：0.30%

---

## 3. 核心问题解答

### 3.1 为什么要降维再量化？

**维度诅咒（Curse of Dimensionality）：**

```
设两个D维标准正态向量x, y，距离平方分布：
  E[||x-y||²] = 2D
  Std[||x-y||²] = 2√(2D)
  
区分度指标 = Std/E = √(2/D) → 0 (当D→∞)

实测：
  D=32:   区分度=0.250 ✅ 可有效区分
  D=512:  区分度=0.063 ⚠️ 区分度弱
  D=2560: 区分度=0.028 ❌ 近/远距离几乎相同
```

**结论：** 2560维空间中，256个码字与任意z_e的距离几乎相等，argmin退化为随机选择，必须先降维到32维。

### 3.2 重建Loss vs 量化Loss

#### 重建Loss（Reconstruction Loss）

```python
recon_loss = MSE(x_recon, x)
           = 平均((x_recon - x)²)
```

**衡量什么：** Decoder能不能从SID还原出原始物品？  
**当前值：** 0.07 → 很好，重建误差很小  
**含义：** 相对于[-18.5, 17.7]的范围，平均偏差约√0.07≈0.26，误差很小

#### 量化Loss（Quantization Loss）

```python
codebook_loss = MSE(z_q, z_e.detach())      # 更新码书
commitment_loss = MSE(z_q.detach(), z_e)    # 更新encoder
quant_loss = codebook_loss + β × commitment_loss
```

**衡量什么：** 连续的32维向量能不能用离散码字表示？  
**当前值：** 0.14 → 可接受，离散化损失不大  
**含义：** 256个码字基本覆盖32维空间，但不够完美

#### 两者关系

```
total_loss = recon_loss + quant_loss_weight × quant_loss
           = 0.07 + 1.0 × 0.14
           = 0.21

量化Loss大 → z_q和z_e差很多 → Decoder拿到"失真"向量 → recon_loss也会大
量化Loss小 → z_q≈z_e → Decoder拿到准确向量 → recon_loss也会小
```

### 3.3 为什么VQ Loss分两项？

**数学上相同，但梯度流向不同：**

```
codebook_loss:
  梯度流向 → codebook（更新码字）
  作用：把码字e_k拉向encoder输出z_e

commitment_loss:
  梯度流向 → encoder（更新encoder）
  作用：把encoder输出z_e拉向码字e_k

.detach()阻断梯度：
  codebook_loss = MSE(z_q, z_e.detach())  ← 阻断流向encoder
  commitment_loss = MSE(z_q.detach(), z_e) ← 阻断流向codebook
```

**β=0.25的物理意义：**
- β小（0.1）：codebook主动适应encoder，encoder自由度高
- β=0.25：平衡两者（VQ-VAE论文推荐）
- β大（1.0）：encoder被迫靠近码字，量化误差小但表达能力受限

### 3.4 K-means初始化的作用

**随机初始化的问题（Codebook Collapse）：**

```
随机初始化：码字分布在[-0.004, 0.004]
z_e分布：[-2.5, 3.1]
→ 严重不匹配，229/256个码字永远不被使用

实测对比：
  随机初始化：码字利用率10.5%，MSE=0.6022
  K-means初始化：码字利用率100%，MSE=0.0421（↓93%）
```

**K-means++算法过程：**

```
Phase A：分散策略
  1. 随机选第1个中心c₁
  2. 以概率∝D(x)²选下一个中心（距离越远概率越大）
  3. 重复直到选出K个中心

Phase B：Lloyd迭代
  E步（分配）：label[i] = argmin_k ||xᵢ - cₖ||²
  M步（更新）：cₖ = mean(xᵢ where label[i]=k)
```

### 3.5 未见物品为什么也能生成SID？

**核心原因：Encoder学到的是通用压缩规则**

```
训练时：
  3686个物品嵌入 → Encoder → 32维连续空间 → 量化 → SID

推理时（新物品）：
  新物品嵌入 → Encoder → 32维连续空间 → 量化 → SID
  
Encoder学到的不是"记住3686个物品"，而是：
  "如何把2560维向量压缩到32维"
  "压缩后的向量能用256×256×256个码字表示"
```

**泛化能力验证：**

```
训练时物品：重建误差 0.07
"未见"物品：重建误差 0.39（约5倍差距）

说明：
  ✅ 能生成SID（任何新物品都有SID）
  ⚠️ 质量有差异（取决于是否落在已学习的32维空间区域）
```

**实际应用建议：**
- 定期重新训练RQ-VAE（包含新产品）
- 或增量学习（用新数据微调）

### 3.6 信息损失为什么还能用SID表示物品？

**关键理解：SID不是"复制"物品，而是"表示"物品**

```
类比1：身份证号
  你本人：身高、体重、指纹...（无限信息）
  身份证号：18个数字（丢失大部分信息）
  → 但在"识别身份"任务上，够了！

类比2：商品条形码
  商品：包装、颜色、重量...
  条形码：13个数字
  → 超市能用条形码管理商品
```

**信息论分析：**

```
区分3686个物品需要的信息量：
  log₂(3686) ≈ 12 bits

SID提供的信息量：
  256³种组合 = log₂(16,777,216) ≈ 24 bits

24 bits > 12 bits → SID有足够能力区分所有物品
```

**SID能做什么：**
- ✅ 区分不同物品（碰撞率0.3%）
- ✅ 供LLM理解和推理
- ✅ 用于推荐排序

**SID不能做什么：**
- ❌ 完全还原物品的所有细节
- ❌ 替代原始嵌入做精细计算

### 3.7 量化损失说明什么？

```
quant_loss = 0.14

说明：
  1. 信息丢失：z_e中的一些细微信息在量化时丢失
  2. 码书覆盖不完整：256个码字无法完美覆盖32维空间
  3. 离散化固有缺陷：连续→离散必然有量化误差

评价：
  ✅ 可接受（0.1~0.3范围）
  ⚠️ 但不是最优（可通过增加层数或码字改善）
```

---

## 4. 消融实验设计

### 4.1 Sinkhorn均衡分配

#### 实验配置

```bash
# 实验1：训练时启用Sinkhorn
python rqvae.py \
    --sk_epsilons 0.003 0.003 0.003 \
    --sk_iters 50

# 对比：当前配置（训练时不启用）
--sk_epsilons 0.0 0.0 0.0
```

#### 预期效果

| 指标 | 不用Sinkhorn | 用Sinkhorn |
|-----|-------------|-----------|
| 训练时碰撞率 | 9.52% | 预计5-7% |
| 码字利用率 | 可能不均衡 | 强制均衡 |
| 训练速度 | 快 | 稍慢（每步多50次迭代）|
| 适用场景 | 数据量大 | 数据量小、防collapse |

#### 原理

```
argmin（标准最近邻）：
  纯贪心，每个样本选最近的码字
  → 热门码字被过度使用，冷门码字浪费

Sinkhorn（均衡分配）：
  软分配 + 均衡约束
  → 每个码字使用次数差不多
  
算法：
  Q = exp(-distances / epsilon)
  交替按行/列归一化，强制均衡
```

### 4.2 不同量化层数

```bash
# 实验2：4层量化
--num_emb_list 256 256 256 256  # 总词表256⁴=42.9亿

预期：
  碰撞率：↓ 降至2-3%
  SID长度：↑ 增加1个token
  训练时间：↑ 增加约30%
```

### 4.3 不同量化维度

```bash
# 实验3：e_dim=16（更高区分度）
--e_dim 16  # 区分度√(2/16)=0.35

预期：
  量化误差：↓ 可能降低
  信息容量：↓ 可能丢失更多信息
  重建误差：↑ 可能变大

# 实验4：e_dim=64（更低区分度）
--e_dim 64  # 区分度√(2/64)=0.18

预期：
  量化误差：↑ 可能变大（区分度弱）
  信息容量：↑ 能保留更多信息
  重建误差：？ 不确定
```

### 4.4 不同码字数量

```bash
# 实验5：512码字/层
--num_emb_list 512 512 512  # 总词表512³=1.34亿

预期：
  碰撞率：↓ 降至1-2%
  量化误差：↓ 降低
  LLM词表：↑ 需要学习更多token
```

### 4.5 不同beta值

```bash
# 实验6：beta=0.1（encoder更自由）
--beta 0.1

预期：
  codebook主动适应数据
  量化误差可能增大

# 实验7：beta=0.5（encoder更保守）
--beta 0.5

预期：
  encoder被迫靠近码字
  量化误差减小，但表达能力受限
```

### 4.6 实验优先级

```
高优先级（推荐先做）：
  1. Sinkhorn均衡分配 → 最直接改善碰撞率
  2. 4层量化 → 增加表达力

中优先级：
  3. beta值调优 → 影响量化/重建权衡
  4. 码字数量（512）→ 改善覆盖率

低优先级：
  5. e_dim=16 → 风险较大
  6. e_dim=64 → 区分度下降
```

---

## 5. 关键细节总结

### 5.1 网络设计细节

#### 为什么Encoder最后层不加激活？

```
原因：z_e需要覆盖(-∞, +∞)的完整空间
如果加ReLU：
  z_e只能取正值[0, +∞)
  → 一半的codebook浪费（负值部分永远不用）
  
实测对比：
  不加ReLU：命中88个码字，MSE=0.7899
  加ReLU：命中56个码字，MSE=0.3839
  → 加ReLU后MSE更小但码字浪费更多（反直觉）
```

#### Xavier初始化的作用

```python
xavier_normal_(module.weight.data)
# σ = √(2/(fan_in+fan_out))

作用：保持各层激活方差恒定
  - 防止梯度消失（权重太小）
  - 防止梯度爆炸（权重太大）
  
对比：
  随机初始化：训练可能不稳定
  Xavier初始化：训练稳定，收敛快
```

#### 为什么不用Batch Normalization？

```
当前配置：bn=False

原因：
  1. 量化损失已起强正则化作用
  2. 数据集小（3686），BN统计量可能不准
  3. 增加计算开销
  
何时用BN：
  - 大数据集（>10万）
  - 训练不稳定
  - 需要加速收敛
```

### 5.2 训练策略细节

#### 为什么batch_size=20480？

```
数据量：3686
batch_size：20480

原因：
  - 3686 < 20480，一次全量更新
  - K-means初始化更准确（看到全部数据）
  - 梯度稳定，不受batch采样影响
  
大数据集时：
  - 10万物品：batch_size=2048或4096
  - 100万物品：batch_size=1024
```

#### 为什么用AdamW而不是Adam？

```
AdamW = Adam + 正确的权重衰减

区别：
  Adam：权重衰减作用于梯度
  AdamW：权重衰减直接作用于参数
  
效果：
  - AdamW泛化更好
  - 防止过拟合
  - 深度学习默认选择
```

#### 学习率调度为什么用Constant？

```python
lr_scheduler_type = "constant"
warmup_epochs = 50

策略：
  前50 epochs：lr从0线性增加到1e-3
  之后：保持1e-3不变

为什么不用Cosine或Linear衰减？
  - 数据量小，不需要精细调lr
  - Constant + Warmup已足够
  - 简化调参
```

### 5.3 评估细节

#### 碰撞率计算

```python
# trainer.py L128-152
indices_set = set()
num_sample = 0
for data in valid_data:
    indices = model.get_indices(data)  # 只用Encoder+RVQ
    for index in indices:
        code = "-".join([str(int(_)) for _ in index])
        indices_set.add(code)

collision_rate = (num_sample - len(indices_set)) / num_sample
```

**含义：** 有多少物品得到了相同的SID

#### 为什么评估用get_indices而不是forward？

```python
@torch.no_grad()
def get_indices(self, xs, use_sk=False):
    z_e = self.encoder(xs)
    _, _, indices = self.rq(z_e, use_sk=use_sk)
    return indices

原因：
  - 推理阶段不需要Decoder
  - 只需SID，不需要重建
  - 节省计算
```

### 5.4 去重机制细节

#### 生成脚本的去重流程

```python
# generate_indices.py L106-133

# 第1步：标准生成（argmin）
all_indices = model.get_indices(d, use_sk=False)

# 第2步：启用Sinkhorn处理碰撞物品
model.rq.vq_layers[-1].sk_epsilon = 0.003
while True:
    collision_item_groups = get_collision_item(all_indices_str)
    if len(collision_item_groups) == 0:
        break
    
    for collision_items in collision_item_groups:
        indices = model.get_indices(d, use_sk=True)
        # 更新碰撞物品的SID
```

**效果：**
- 训练时碰撞率：9.52%
- 去重后碰撞率：0.30%
- 改善：31倍

#### 为什么只在最后一层用Sinkhorn？

```python
for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon = 0.0  # 前几层不用
model.rq.vq_layers[-1].sk_epsilon = 0.003  # 最后一层用

原因：
  - 前几层负责主方向，argmin足够
  - 最后一层负责精细区分，需要均衡
  - 计算效率考虑
```

---

## 6. 常见问题FAQ

### Q1: 训练时Loss震荡正常吗？

```
现象：
  epoch 60: loss=1.17
  epoch 70: loss=11.7  ← 突然升高
  epoch 72: loss=16.8  ← 继续升高

原因：
  - K-means初始化后网络在适应新的量化空间
  - codebook和encoder在互相适应
  - 前100 epochs的震荡是正常的

解决：
  - 继续训练，会自动收敛
  - 如果一直震荡，降低lr到1e-4
```

### Q2: 碰撞率为什么不是0？

```
原因：
  1. 物品本身相似（同系列不同型号）
  2. 量化误差（32维连续→离散有信息损失）
  3. Encoder压缩（2560→32维可能丢失细微差异）

改善方法：
  - 增加层级：[256,256,256,256]
  - 增加e_dim：32→64（但区分度下降）
  - 用Sinkhorn训练
```

### Q3: 为什么重建误差0.07还说"好"？

```
对比基准：
  - x的范围：[-18.5, 17.7]（跨度36.2）
  - 误差0.07意味着平均偏差√0.07≈0.26
  - 0.26/36.2 = 0.7%（相对误差不到1%）

结论：
  相对于原始范围，重建误差很小
```

### Q4: 训练多久合适？

```
判断标准：
  - 碰撞率不再下降
  - Loss稳定
  - 重建质量可接受

当前实验：
  - 10000 epochs，47分钟
  - 碰撞率9.52% → 已接近收敛
  
建议：
  - 小数据集（<1万）：5000-10000 epochs
  - 中等数据集（1-10万）：3000-5000 epochs
  - 大数据集（>10万）：1000-3000 epochs
```

### Q5: 如何选择超参数？

```
快速决策树：

1. 碰撞率>20%？
   → 是：增加num_emb_list或层数
   → 否：继续

2. 重建误差>0.2？
   → 是：增加e_dim或layers深度
   → 否：继续

3. 训练不稳定？
   → 是：降低lr或增加warmup
   → 否：继续

4. 码字利用率<50%？
   → 是：开启K-means初始化
   → 否：参数合理
```

### Q6: 能否在CPU上训练？

```
可以，但很慢。

当前配置（GPU）：
  - 每epoch 0.28秒
  - 总耗时47分钟

CPU预估：
  - 每epoch 2-5秒
  - 总耗时5-14小时

建议：
  - 小数据集可以CPU调试
  - 正式训练用GPU
```

### Q7: 多数据集如何训练？

```bash
# 为每个数据集独立训练
for dataset in "Industrial_and_Scientific" "Office_Products"; do
    python rqvae.py \
        --data_path ../data/Amazon/index/${dataset}.emb-qwen-td.npy \
        --ckpt_dir ./output/${dataset} \
        ...
done

注意：
  - 不同数据集的SID词表独立
  - SFT阶段需要合并词表
```

---

## 7. 下一步计划

### 7.1 消融实验（待做）

- [ ] Sinkhorn均衡分配训练
- [ ] 4层量化实验
- [ ] beta值调优（0.1, 0.5）
- [ ] 码字数量对比（256 vs 512）

### 7.2 SFT训练

```bash
cd /data1/lyt/MiniOneRec
bash sft.sh
```

配置：
- 模型：Qwen2.5-1.5B-Instruct
- 数据：Industrial_and_Scientific训练集
- 卡数：4张（GPU 4/5/7/8/9中的空卡）

### 7.3 RL训练

```bash
cd /data1/lyt/MiniOneRec
bash rl.sh
```

配置：
- 模型：SFT后的checkpoint
- 策略：GRPO强化学习
- 奖励：Ranking-based

---

## 8. 文件清单

| 文件 | 路径 | 说明 |
|-----|-----|-----|
| 训练脚本 | `rq/rqvae.py` | RQ-VAE主训练脚本 |
| 训练日志 | `rq/rqvae_train.log` | 完整训练日志 |
| 最佳模型 | `rq/output/.../best_collision_model.pth` | 碰撞率最低的检查点 |
| SID索引 | `data/Amazon/index/Industrial_and_Scientific.index.json` | 生成的SID文件 |
| 物品元数据 | `data/Amazon/index/Industrial_and_Scientific.item.json` | 物品信息 |
| SFT脚本 | `sft.sh` | 监督微调启动脚本 |
| RL脚本 | `rl.sh` | 强化学习启动脚本 |

---

## 9. 关键代码片段

### 9.1 RQ-VAE前向传播

```python
# rq/models/rqvae.py L61-66
def forward(self, x, use_sk=True):
    z_e = self.encoder(x)                        # 压缩
    z_q, rq_loss, indices = self.rq(z_e, ...)   # 残差量化
    out = self.decoder(z_q)                      # 重建
    return out, rq_loss, indices
```

### 9.2 残差量化核心

```python
# rq/models/rq.py L39-56
def forward(self, x, use_sk=True):
    x_q = 0
    residual = x
    
    for quantizer in self.vq_layers:
        x_res, loss, indices = quantizer(residual, use_sk=use_sk)
        residual = residual - x_res    # 更新残差
        x_q = x_q + x_res              # 累积量化向量
        all_losses.append(loss)
        all_indices.append(indices)
    
    return x_q, mean_losses, all_indices
```

### 9.3 VQ Loss计算

```python
# rq/models/vq.py L90-95
commitment_loss = F.mse_loss(x_q.detach(), x)  # 更新encoder
codebook_loss = F.mse_loss(x_q, x.detach())    # 更新codebook
loss = codebook_loss + self.beta * commitment_loss

# Straight-Through Estimator
x_q = x + (x_q - x).detach()
```

### 9.4 K-means初始化触发

```python
# rq/models/vq.py L67-68
def forward(self, x, use_sk=True):
    if not self.initted and self.training:
        self.init_emb(latent)  # 第一次forward时触发
```

---

## 10. 完整数据流程（新增）

### 10.1 原始数据集

```
来源: Amazon Reviews 2018
官网: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

原始文件格式:
  Industrial_and_Scientific.json (评论数据)
  {
    "reviewerID": "A2X4VPK8ZT5XYZ",
    "asin": "B000067ZQZ",
    "overall": 5.0,
    "reviewText": "This product is great...",
    "unixReviewTime": 1540000000
  }
  
  meta_Industrial_and_Scientific.json (物品元数据)
  {
    "asin": "B000067ZQZ",
    "title": "SUPCO SPP6 Relay/Capacitor Hard Start Kit",
    "description": "This is a hard start kit...",
    "brand": "SUPCO",
    "categories": [["Industrial & Scientific", "Power Transmission"]]
  }
```

### 10.2 数据预处理流程

```
步骤1: K-core过滤 (amazon18_data_process.py)
  输入: 原始.json文件
  过滤规则:
    - user_k=5: 只保留交互≥5个物品的用户
    - item_k=5: 只保留被≥5个用户交互的物品
    - 时间范围: 2016-10 ~ 2018-11
    - 标题过滤: 删除无标题、HTML标签、过长标题
  
  输出:
    Industrial_and_Scientific.item.json    ← 物品元数据
    Industrial_and_Scientific.train.inter  ← 训练集交互
    Industrial_and_Scientific.valid.inter  ← 验证集交互
    Industrial_and_Scientific.test.inter   ← 测试集交互
    Industrial_and_Scientific.user2id      ← 用户ID映射
    Industrial_and_Scientific.item2id      ← 物品ID映射

.inter文件格式 (TSV):
  user_id:token    item_id_list:token_seq    item_id:token
  22    117 52 89 130    118
  869   130              130
  1959  1154 1153 1155 1156    1157

.item.json格式:
  {
    "0": {
      "title": "SUPCO SPP6 Relay/Capacitor Hard Start Kit...",
      "description": "This is a hard start kit...",
      "brand": "SUPCO",
      "categories": "Industrial & Scientific,Power Transmission"
    },
    "1": {...},
    ...
    "3685": {...}
  }
```

### 10.3 物品嵌入生成

```bash
# 脚本: rq/text2emb/amazon_text2emb.py
bash rq/text2emb/amazon_text2emb.sh \
     --dataset Industrial_and_Scientific \
     --root ./data/Amazon18/Industrial_and_Scientific \
     --plm_name qwen \
     --plm_checkpoint /path/to/Qwen-model
```

**转换过程:**

```python
步骤1: 加载物品元数据
  Industrial_and_Scientific.item.json
  ↓

步骤2: 提取文本
  item_text = title + " " + description
  示例: "SUPCO SPP6 Relay... This is a hard start kit..."
  ↓

步骤3: Qwen模型提取嵌入
  encoded = tokenizer(item_text)
  outputs = model(input_ids, attention_mask)
  embedding = mean_pooling(outputs.last_hidden_state)  # [2560维]
  ↓

步骤4: 保存
  Industrial_and_Scientific.emb-qwen-td.npy
  shape: (3686, 2560)
  3686个物品 × 2560维
```

**关键点:**
- 数据来源: `.item.json` (K-core过滤后的物品元数据)
- 文本拼接: `title + description`
- 提取方法: Mean Pooling (所有token平均，不是[CLS])
- 文件命名: `.emb-qwen-td` (td=title+description)

### 10.4 RQ-VAE训练与SID生成

```
步骤3: 训练RQ-VAE (rqvae.py)
  输入: Industrial_and_Scientific.emb-qwen-td.npy
  处理: 训练Encoder-Decoder+残差量化
  输出: best_collision_model.pth
  
  过滤规则: ❌ 无过滤！
  - 包含所有3686个物品
  - 每个物品只有1条记录（它的2560维嵌入）
  - 原因: RQ-VAE需要为"所有可能推荐的物品"生成SID
  ↓

步骤4: 生成SID索引 (generate_indices.py)
  输入: 3686个物品的嵌入 + 训练好的模型
  处理:
    - 第1阶段: argmin生成所有物品的SID
    - 第2阶段: Sinkhorn处理碰撞物品
  输出: Industrial_and_Scientific.index.json
  
  碰撞率: 9.52% → 0.30% (去重后)
```

### 10.5 SFT训练集构建

```bash
# 脚本: convert_dataset.py
bash convert_dataset.sh
```

**转换过程:**

```python
步骤1: 加载.inter文件
  Industrial_and_Scientific.train.inter
  user_id \t history_items \t target_item
  22    117 52 89 130    118
  ↓

步骤2: 查SID索引
  index.json["117"] = ['<a_165>', '<b_107>', '<c_44>']
  index.json["118"] = ['<a_104>', '<b_118>', '<c_176>']
  ↓

步骤3: 写入CSV
  Industrial_and_Scientific_5_2016-10-2018-11.csv
  user_id, history_item_id, item_id, history_item_sid, item_sid
  A22, [117,52,89,130], 118, ['<a_165>...','...'], '<a_104>...'
```

**关键点:**
- SFT训练集的SID是"预处理"好的，不是训练时动态查找
- CSV文件已经包含SID字符串，sft.py直接读取
- 如果改了SID索引，需要重新运行convert_dataset.py

### 10.6 数据量对比

| 数据集 | 数据量 | 过滤规则 | 内容 |
|-------|-------|---------|-----|
| **原始Amazon** | 500万交互 | 无 | 用户评论+物品元数据 |
| **K-core过滤后** | 30万交互 | user_k=5, item_k=5 | 高质量交互 |
| **RQ-VAE训练集** | 3686条 | ❌ 无过滤 | 物品嵌入(2560维) |
| **.inter文件** | 30万条 | K-core+时间+标题 | 用户交互序列 |
| **SFT训练集** | 25万条 | 同.inter | 含SID的训练样本 |

**为什么RQ-VAE只有3686条，而SFT有25万条？**

```
RQ-VAE训练集:
  - 对象: 物品
  - 数量: 3686个唯一物品
  - 每条: 1个物品的2560维嵌入
  - 去重: ✅ 每个物品1条

SFT训练集:
  - 对象: 用户-物品交互
  - 数量: 30万条交互 → 25万条训练样本
  - 每条: 用户历史SID序列 → 目标SID
  - 去重: ❌ 同一物品可被多用户交互

放大效应:
  用户A历史: [1, 2, 3, 4, 5]
  生成样本:
    [1] → 2
    [1, 2] → 3
    [1, 2, 3] → 4
    [1, 2, 3, 4] → 5
  → 5个物品生成4个训练样本！
```

### 10.7 RQ-VAE模型的生命周期

```
阶段1: RQ-VAE训练（一次性）
  输入: 3686个物品的嵌入
  训练: 47分钟
  输出: best_collision_model.pth
  ↓

阶段2: SID生成（一次性）
  输入: 3686个物品的嵌入
  模型: best_collision_model.pth
  处理: 推理 + Sinkhorn去重
  输出: index.json (物品ID → SID映射)
  ↓

阶段3: 数据集转换（一次性）
  输入: .inter文件 + index.json
  处理: convert_dataset.py 查表
  输出: SFT训练集CSV (SID已写入)
  ↓

阶段4: SFT训练（主要训练）
  输入: SFT训练集CSV
  模型: Qwen2.5-1.5B-Instruct
  训练: 4小时
  输出: SFT微调模型
  ↓

阶段5: RL训练（主要训练）
  输入: RL数据集 + SFT模型
  训练: 强化学习对齐
  输出: 最终推荐模型
  ↓

阶段6: 推理部署
  输入: 用户历史SID序列
  模型: RL微调后的Qwen
  输出: 预测下一个SID
  查找: SID → 物品 (反向查表)

关键: RQ-VAE在阶段2后就"退役"了！
  ✅ 阶段1-2: RQ-VAE活跃使用
  ❌ 阶段4-6: RQ-VAE不再使用
  → SFT/RL/推理只用SID字符串，不用RQ-VAE模型
```

### 10.8 去重的本质

**去重针对的是哪个数据集？**

```
✅ 去重的是: 训练RQ-VAE的数据集 (3686个物品的嵌入)
✅ 去重的是: 这些物品的SID
❌ 不是: SFT训练集

去重发生在: generate_indices.py (步骤4)
  输入: 3686个物品 → 3686个SID
  问题: 9.52%的物品SID相同 (350个物品)
  处理: Sinkhorn重新分配碰撞物品
  输出: index.json (碰撞率0.30%)

SFT训练集"间接"使用去重结果:
  - convert_dataset.py 查index.json
  - 写入CSV的SID已经是去重后的
  - sft.py 读取CSV，不需要再去重
```

**为什么训练好的RQ-VAE还会碰撞？**

```
原因1: 训练目标 ≠ 碰撞率
  Loss = recon_loss + quant_loss
  - recon_loss关注: x_recon ≈ x
  - quant_loss关注: z_q ≈ z_e
  - 但都不关注: SID是否唯一！

原因2: 量化必然有误差
  z_e_A 和 z_e_B 很接近
  → 它们离同一个码字最近
  → argmin选了相同的码字
  → SID相同

原因3: 码书覆盖不完整
  256个码字无法完美覆盖32维空间
  → 某些区域"拥挤"，某些"稀疏"
  → 相似物品容易落到同一区域

去重的作用:
  不是"修复模型"
  而是"后处理优化"
  → 碰撞率9.52% → 0.30%
  → 改善31倍
```

---

## 11. 消融实验对比（新增）

### 11.1 RQ-Kmeans实验结果

```bash
# 实验3.1.2: RQ-Kmeans (FAISS)
python rq/rqkmeans_faiss.py --dataset Industrial_and_Scientific
```

**结果:**

| 指标 | RQ-Kmeans | RQ-VAE | 差距 |
|-----|-----------|--------|-----|
| **生成碰撞率** | **20.05%** | **9.52%** | RQ-VAE优52% |
| **训练时间** | 5秒 | 47分钟 | Kmeans快564倍 |
| **唯一SID数** | 2947/3686 | 3300/3686 | RQ-VAE多10.7% |
| **码字利用率** | 100% (L1/L2/L3) | 100% | 相同 |

**为什么RQ-Kmeans碰撞率更高？**

```
1. 随机初始化问题
   → FAISS用随机初始K-means
   → 码书分布不均匀

2. 无均衡约束
   → 热门区域拥挤
   → 2947个唯一SID < 3686个物品

3. 线性量化
   → 只能做线性分割
   → RQ-VAE的非线性Encoder更好
```

### 11.2 四种方法对比矩阵

| 维度 | RQ-VAE | RQ-Kmeans | 约束RQ-Kmeans | RQ-Kmeans+ |
|-----|--------|-----------|-------------|------------|
| **核心方法** | 神经网络 | K-means | 约束K-means | K-means++ |
| **降维** | Encoder学习 | PCA | PCA | PCA |
| **量化** | 训练码书 | 聚类 | 约束聚类 | 聚类 |
| **初始化** | K-means++ | 随机 | 约束 | K-means++ |
| **碰撞率(预估)** | 9.52% | 20.05% | 5-10% | 8-15% |
| **训练时间** | 47min | 5sec | 30min | 15min |
| **重建能力** | ✅ | ❌ | ❌ | ❌ |
| **码书质量** | 高 | 低 | 中 | 中 |

### 11.3 消融实验的科学价值

```
控制变量:
  - 数据集相同 (Industrial_and_Scientific)
  - 维度相同 (32维)
  - 码字数量相同 (256×256×256)
  - 评估指标相同 (碰撞率)

对比维度:
  1. 非线性 vs 线性 (RQ-VAE vs K-means)
  2. 随机 vs 智能初始化 (Kmeans vs Kmeans++)
  3. 无约束 vs 有约束 (Kmeans vs Constrained)
  4. 端到端 vs 分步 (RQ-VAE vs 其他)

预期结论:
  - RQ-VAE最优（但最慢）
  - K-means++性价比最高
  - 约束K-means分布最均匀
  - 标准K-means最弱基线
```

---

## 12. SFT训练配置优化（新增）

### 12.1 GPU资源管理

```bash
# 查看GPU状态
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

# 选择空闲GPU
CUDA_VISIBLE_DEVICES=5,9 bash sft.sh
```

**关键规则:**
- CUDA_VISIBLE_DEVICES=5,9 → torchrun中device编号重新映射
- nproc_per_node=2 → 使用2张卡
- batch_size=32, micro_batch_size=4 → 每张卡处理4个样本

### 12.2 显存占用分析

**训练配置:**
```
模型: Qwen2.5-1.5B-Instruct (1.5B参数)
精度: BF16 (2字节/参数)
GPU: 2张 (GPU 5, GPU 9)
batch_size=32, micro_batch_size=4, gradient_accumulation=4
```

**显存详细拆解:**
```
模型参数:    3.44 GB
优化器:     6.0 GB (ZeRO-2分区后)
梯度:        1.5 GB (ZeRO-2分区后)
激活值:      1.5 GB (micro_batch=4)
其他开销:    1.0 GB
─────────────────────
训练相关:   13.44 GB

原有进程:   16.5 GB (GPU 5) / 7.4 GB (GPU 9)
PyTorch碎片: 12 GB (GPU 5) / 10.5 GB (GPU 9)
─────────────────────
实际占用:   43.9 GB (GPU 5) / 33.3 GB (GPU 9)
```

**DeepSpeed ZeRO-2效果:**
```
优化器节省: 12 GB (分区到2张卡)
梯度节省:    3 GB (分区到2张卡)
总节省:     15 GB
```

### 12.3 Batch配置对显存的影响

```
batch_size: 32 (全局)
micro_batch_size: 4 (每张卡)
gradient_accumulation: 32/(4×2) = 4步

为什么micro_batch_size影响显存？
  激活值 = 单样本激活 × micro_batch_size
  micro_batch=4:  激活值 ≈ 0.71 GB
  micro_batch=32: 激活值 ≈ 5.7 GB
  节省: 5.0 GB

为什么梯度累积不增加显存？
  梯度张量大小固定（1.5B个参数）
  只是数值累加，不是创建新张量
  → 显存不增加
```

### 12.4 训练进度监控

```bash
# 查看训练日志
tail -50 /data1/lyt/MiniOneRec/sft_train.log

# 查看GPU状态
watch -n 2 nvidia-smi
```

**训练状态示例:**
```
进度: 2897/7485 步 (38.7%)
Epoch: 1.16 / 3.0
Loss: 8.08 → 1.6 (下降80%)
学习率: 6.15e-05 (余弦衰减中期)
预计剩余: 2小时34分钟
```

---

## 13. 常见问题补充（新增）

### Q8: RQ-VAE训练集和.inter文件的过滤规则分别是什么？

```
RQ-VAE训练集 (.npy):
  过滤规则: ❌ 无过滤
  数据量: 3686条 (=物品数)
  内容: 物品嵌入(2560维)
  用途: 训练RQ-VAE生成SID

.inter文件:
  过滤规则: ✅ K-core + 时间 + 标题
    - user_k=5, item_k=5
    - 时间: 2016-10 ~ 2018-11
    - 标题: 删除无标题/HTML/过长
  数据量: 30万条 (=交互数)
  内容: 用户交互序列
  用途: 训练SFT学习推荐

为什么不同？
  RQ-VAE: 需要为"所有物品"生成SID，不能遗漏
  .inter: 需要"高质量交互"，过滤冷启动用户/冷门物品
```

### Q9: 3686个物品是专门划分给RQ-VAE的吗？

```
❌ 不是专门划分的
✅ 是整个数据集的全部物品

3686 = K-core过滤后的物品总数
  - RQ-VAE训练用全部3686个
  - SFT训练也用全部3686个（通过用户行为）
  - 推理时也是这3686个

原因:
  如果RQ-VAE只训练部分物品:
    → 其他物品没有SID
    → SFT训练中出现这些物品，无法处理
    → 推理时想推荐这些物品，没有SID
```

### Q10: 为什么SFT训练集不需要去重？

```
因为SFT训练集使用的SID来自index.json
而index.json在生成时已经去重了

流程:
  generate_indices.py → index.json (去重后，碰撞率0.30%)
  convert_dataset.py 查index.json → CSV (SID已写入)
  sft.py 读取CSV → 训练 (SID已唯一)

SFT训练集"继承"了去重结果
不需要再次去重
```

### Q11: RQ-VAE模型推理时还用吗？

```
❌ 不再使用！

RQ-VAE的使命:
  阶段1: 训练模型 (47分钟)
  阶段2: 生成SID索引 (一次性)
  阶段3: 退役

SFT/RL/推理阶段:
  只用SID字符串
  不用RQ-VAE模型

类比:
  RQ-VAE = 印刷厂
  SID索引 = 电话号码本
  SFT模型 = 接线员
  
  接线员上岗后，不需要印刷厂了
```

