# RQ-VAE：语义 ID 构造原理与代码详解

> 适用项目：MiniOneRec / `rq/` 目录  
> 核心目标：将每个物品的高维文本嵌入，压缩为一串离散的语义 token（SID），供大语言模型直接使用

---

## 目录

1. [整体架构](#1-整体架构)
2. [为什么需要降维再量化](#2-为什么需要降维再量化)
3. [模块一：MLPLayers（Encoder / Decoder）](#3-模块一mlplayersencoder--decoder)
4. [模块二：VectorQuantizer（单层 VQ）](#4-模块二vectorquantizer单层-vq)
5. [模块三：ResidualVectorQuantizer（残差 RVQ）](#5-模块三residualvectorquantizer残差-rvq)
6. [模块四：RQVAE（主模型）](#6-模块四rqvae主模型)
7. [Codebook 的 K-means 初始化](#7-codebook-的-k-means-初始化)
8. [Sinkhorn 均衡分配](#8-sinkhorn-均衡分配)
9. [损失函数设计](#9-损失函数设计)
10. [训练器 Trainer](#10-训练器-trainer)
11. [SID 生成（推理阶段）](#11-sid-生成推理阶段)
12. [运行命令](#12-运行命令)
13. [关键超参数说明](#13-关键超参数说明)

---

## 1. 整体架构

```
物品文本（title + description）
        │
        ▼  冻结的 Qwen 文本编码器
物品嵌入 x  [N, 2560]
        │
        ▼  Encoder（MLP 降维）
   z_e  [N, 32]          ← 量化潜空间，压缩比 80x
        │
        ▼  ResidualVectorQuantizer（3层残差量化）
   z_q  [N, 32]          ← 量化后向量（可重建）
   indices [N, 3]        ← SID 原始码字，如 [21, 100, 136]
        │
        ▼  Decoder（MLP 升维，仅训练时用）
  x_recon [N, 2560]      ← 重建嵌入，计算 recon_loss

SID Tokens: ['<a_22>', '<b_101>', '<c_137>']
            ↑加入 LLM 词表，供 SFT / RL 阶段使用
```

**数据流概述：**

| 阶段 | 输入 | 输出 | 关键操作 |
|------|------|------|--------|
| Encode | x [B,2560] | z_e [B,32] | MLP 逐层压缩 80x |
| Quantize | z_e [B,32] | z_q [B,32], indices [B,L] | L 层残差 VQ |
| Decode | z_q [B,32] | x_recon [B,2560] | MLP 镜像扩张（训练专用）|
| SID生成 | indices [N,L] | JSON token 文件 | +1 offset + 去重 |

---

## 2. 为什么需要降维再量化

### 维度诅咒（Curse of Dimensionality）

设两个 D 维标准正态向量 x, y，它们距离平方的分布：

```
E[||x-y||²]   = 2D
Std[||x-y||²] = 2√(2D)

区分度指标 = Std / E = √(2/D)  →  0  当 D → ∞
```

| 维度 D | 区分度 √(2/D) | 最近/最远距离比 | 含义 |
|-------|-------------|--------------|-----|
| 32 | 0.250 | 0.19 | 可有效区分 |
| 512 | 0.063 | 0.70 | 区分度弱 |
| **2560** | **0.028** | **0.86** | **近/远距离几乎相同，argmin 失效** |

**结论：** 2560 维空间中，256 个 codebook 码字与任意 z_e 的距离几乎相等，`argmin` 退化为随机选择，必须先降维到 32 维再量化。

---

## 3. 模块一：MLPLayers（Encoder / Decoder）

**文件：** `rq/models/layers.py`

```python
class MLPLayers(nn.Module):
    def __init__(self, layers, dropout=0.0, activation="relu", bn=False):
        super().__init__()
        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))

            if self.use_bn and idx != (len(self.layers)-2):
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))

            # 最后一层不加激活（保留负值，供量化使用）
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx != (len(self.layers)-2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)   # Xavier 初始化

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)   # 保持各层激活方差稳定
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)
```

**层结构（Industrial 数据集默认配置）：**

```
Encoder: [2560] → Dropout→Linear→ReLU → [512]
                → Dropout→Linear→ReLU → [256]
                → Dropout→Linear→ReLU → [128]
                → Dropout→Linear→ReLU → [64]
                → Dropout→Linear       → [32]   ← 最后层无激活！

Decoder: [32] → ... → [2560]   (Encoder 的镜像)
```

**设计要点：**

| 组件 | 作用 | 原理 |
|-----|------|-----|
| Linear [in→out] | 仿射变换，提取特征方向 | `y = Wx + b`，W 每行是一个"方向检测器" |
| Xavier 初始化 | 保持各层激活方差≈1 | `σ = √(2/(fan_in+fan_out))`，防梯度消失/爆炸 |
| ReLU | 引入非线性，稀疏激活 | `f(x)=max(0,x)`，正区间梯度恒为1，不消失 |
| **最后层无激活** | 保留负值 | z_e 需覆盖 (-∞,+∞)，否则 codebook 一半浪费 |
| Dropout(p=0) | 正则化（本项目关闭） | 量化损失已起强正则化作用 |

---

## 4. 模块二：VectorQuantizer（单层 VQ）

**文件：** `rq/models/vq.py`

单层向量量化器，将连续向量映射到离散码字索引。

```python
class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25,
                 kmeans_init=False, kmeans_iters=10,
                 sk_epsilon=0.003, sk_iters=100):
        super().__init__()
        self.n_e = n_e          # codebook 大小（码字数量）
        self.e_dim = e_dim      # 每个码字的维度

        # codebook：可学习的嵌入矩阵 [n_e, e_dim]
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()  # 等待 K-means 初始化

    def init_emb(self, data):
        """K-means 初始化 codebook（第一次 forward 时触发）"""
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim)

        # 1. 首次训练时用 K-means 初始化
        if not self.initted and self.training:
            self.init_emb(latent)

        # 2. 计算 L2 距离矩阵 [B, K]
        #    ||z_e - e_k||² = ||z_e||² + ||e_k||² - 2·z_e·e_k
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, self.embedding.weight.t())

        # 3. 分配码字（argmin 或 Sinkhorn 均衡分配）
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)           # 标准最近邻
        else:
            d = self.center_distance_for_constraint(d)  # 中心化到[-1,1]
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
            indices = torch.argmax(Q, dim=-1)           # 均衡分配

        # 4. 查表得到量化向量
        x_q = self.embedding(indices).view(x.shape)

        # 5. 计算 VQ 损失
        commitment_loss = F.mse_loss(x_q.detach(), x)  # 更新 encoder
        codebook_loss   = F.mse_loss(x_q, x.detach())  # 更新 codebook
        loss = codebook_loss + self.beta * commitment_loss

        # 6. Straight-Through Estimator（梯度直通）
        x_q = x + (x_q - x).detach()
        #     前向: 使用量化值 x_q
        #     反向: 梯度直接流向 x（绕过不可微的 argmin）

        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices
```

**VQ 损失分解：**

```
loss = codebook_loss + β × commitment_loss

codebook_loss   = MSE(x_q, z_e.detach())
  → 把码字 e_k 拉向 encoder 输出 z_e（更新 codebook）
  → 梯度：∂/∂e_k = 2(e_k - z_e) / B

commitment_loss = MSE(x_q.detach(), z_e)
  → 把 z_e 拉向最近码字（约束 encoder 不乱跑）
  → β=0.25（VQ-VAE 论文推荐值）
```

**Straight-Through Estimator（STE）：**

```python
# argmin 不可微，梯度会断
# STE 解决方案：前向用 x_q，反向梯度直通 x
x_q_ste = x + (x_q - x).detach()

# 前向：x_q_ste = x_q（量化离散值）
# 反向：∂x_q_ste/∂x = 1（恒等梯度，绕过 argmin）
```

---

## 5. 模块三：ResidualVectorQuantizer（残差 RVQ）

**文件：** `rq/models/rq.py`  
**参考论文：** SoundStream: An End-to-End Neural Audio Codec

```python
class ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_e_list, e_dim, sk_epsilons, beta=0.25,
                 kmeans_init=False, kmeans_iters=100, sk_iters=100):
        super().__init__()
        # 为每一层创建独立的 VectorQuantizer
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(n_e, e_dim, beta=beta,
                            kmeans_init=kmeans_init,
                            kmeans_iters=kmeans_iters,
                            sk_epsilon=sk_epsilon,
                            sk_iters=sk_iters)
            for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)
        ])

    def forward(self, x, use_sk=True):
        all_losses, all_indices = [], []

        x_q = 0
        residual = x            # 初始残差 = z_e

        for quantizer in self.vq_layers:
            # 量化当前残差
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res   # 更新残差（下一层处理剩余误差）
            x_q = x_q + x_res            # 累积量化向量

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)  # [B, n_levels]

        return x_q, mean_losses, all_indices
```

**残差量化的核心思想：**

```
为什么用残差而不是并行多个 VQ？

并行（差）：每个 VQ 独立量化同一个 z_e  →  信息重叠，浪费容量
残差（优）：每层专注解释"上一层解释不了的部分"

Level-1: residual_0 = z_e          → 量化主方向  → 更新 residual_1
Level-2: residual_1 = z_e - x_q1  → 量化Level-1误差 → 更新 residual_2
Level-3: residual_2 = ...          → 精细修正

实测残差 MSE 递减（Industrial 数据集，初始化时）:
  Level-1: MSE = 0.0127  （压缩了z_e主要信息）
  Level-2: MSE = 0.0013  （又压缩了90%的误差）
  Level-3: MSE ≈ 0.0000  （基本完整描述物品）

SID 词表容量 = 256³ = 16,777,216  远超物品数 ~3700，碰撞率极低
```

---

## 6. 模块四：RQVAE（主模型）

**文件：** `rq/models/rqvae.py`

```python
class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,       # 各层 codebook 大小，如 [256,256,256]
                 e_dim=64,                # 量化空间维度
                 layers=None,             # Encoder 隐藏层，如 [2048,1024,512,256,128,64]
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,   # RVQ 损失权重
                 beta=0.25,               # commitment loss 权重
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,        # Sinkhorn epsilon，>0 时启用均衡分配
                 sk_iters=100):
        super().__init__()

        # Encoder: [in_dim, *layers, e_dim]
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims, ...)

        # RVQ: L 层残差量化
        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim, ...)

        # Decoder: Encoder 的镜像 [e_dim, *layers[::-1], in_dim]
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims, ...)

    def forward(self, x, use_sk=True):
        z_e = self.encoder(x)                        # 压缩
        z_q, rq_loss, indices = self.rq(z_e, ...)   # 残差量化
        out = self.decoder(z_q)                      # 重建
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        """推理专用：只需 Encoder + RVQ，不需要 Decoder"""
        z_e = self.encoder(xs)
        _, _, indices = self.rq(z_e, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):
        loss_recon = F.mse_loss(out, xs, reduction='mean')  # 重建损失
        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        return loss_total, loss_recon
```

**参数量统计（Industrial 数据集默认配置）：**

| 模块 | 参数量 | 占比 |
|-----|-------|-----|
| Encoder (2560→32) | 1,485,792 | 49.5% |
| RVQ Codebook (3×256×32) | 24,576 | 0.8% |
| Decoder (32→2560) | 1,488,320 | 49.6% |
| **总计** | **2,998,688** | 100% |

> Codebook 参数量极少（< 1%），但它是 SID 的核心载体。

---

## 7. Codebook 的 K-means 初始化

**文件：** `rq/models/layers.py` → `kmeans()`  
**触发：** `vq.py` 第一次 `forward()` 时

```python
def kmeans(samples, num_clusters, num_iters=10):
    x = samples.cpu().detach().numpy()
    cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters).fit(x)
    centers = cluster.cluster_centers_
    return torch.from_numpy(centers).to(device)
```

```python
# vq.py 中的触发机制
def forward(self, x, use_sk=True):
    latent = x.view(-1, self.e_dim)
    if not self.initted and self.training:   # 只在第一次训练时触发
        self.init_emb(latent)               # K-means 初始化
    ...
```

**K-means 算法过程：**

```
Phase A：K-means++ 初始化（分散策略）
  1. 随机选第1个中心 c₁
  2. 以概率 ∝ D(x)²（到已选中心的最近距离²）选下一个中心
     → 距离越远的点被选中的概率越大，保证中心分散
  3. 重复直到选出 K 个中心

Phase B：Lloyd 迭代
  E步（分配）: label[i] = argmin_k ||xᵢ - cₖ||²
  M步（更新）: cₖ = mean( xᵢ  where label[i] = k )
  重复直到收敛或达到 max_iter
```

**初始化效果对比（实测 Industrial 数据集）：**

| 指标 | 随机初始化 | K-means 初始化 |
|-----|----------|-------------|
| 量化误差 MSE | 0.6022 | **0.0421** |
| 码字利用率 | 10.5% | **100%** |
| 距离区分度 | 0.0003 | **0.3722** |
| 未使用码字 | 229/256 个 | **0 个** |

> 随机初始化码字分布在 [-0.004, 0.004]，而 z_e 分布在 [-2.5, 3.1]，严重不匹配，导致 229/256 个码字永远不被使用（Codebook Collapse）。

---

## 8. Sinkhorn 均衡分配

**文件：** `rq/models/layers.py` → `sinkhorn_algorithm()`  
**启用条件：** `sk_epsilon > 0`（默认 0.0，即不启用）

标准 `argmin` 分配可能导致某些码字被过度使用而其他码字闲置。Sinkhorn 算法通过软分配约束使每个码字被均匀使用。

```python
@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    # distances: [B, K]  距离矩阵（已中心化）
    # 转化为亲和度矩阵
    Q = torch.exp(-distances / epsilon)

    B = Q.shape[0]  # 样本数
    K = Q.shape[1]  # codebook 大小

    # 全局归一化
    Q /= Q.sum()

    for it in range(sinkhorn_iterations):
        # 按行归一化：每个样本的总权重 = 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # 按列归一化：每个码字的总权重 = 1/K（均衡使用）
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K

    Q *= B   # 缩放使列和=1（有效分配矩阵）
    return Q  # [B, K] 软分配矩阵
```

**距离中心化（用于 Sinkhorn 的数值稳定性）：**

```python
@staticmethod
def center_distance_for_constraint(distances):
    # 将距离矩阵归一化到 [-1, 1]
    max_d, min_d = distances.max(), distances.min()
    middle    = (max_d + min_d) / 2
    amplitude = max_d - middle + 1e-5
    return (distances - middle) / amplitude
```

**Sinkhorn vs argmin 对比：**

| 方法 | 分配策略 | 码字利用率 | 适用场景 |
|-----|---------|---------|---------|
| argmin | 硬分配，纯最近邻 | 可能不均衡 | 数据量大、分布均匀 |
| Sinkhorn | 软分配，约束均衡 | 强制均衡 | 数据量小、防 collapse |

---

## 9. 损失函数设计

```
total_loss = recon_loss + quant_loss_weight × rq_loss

recon_loss = MSE(x_recon, x)
           = (1/D) × Σᵢ (x_recon_i - xᵢ)²

rq_loss = mean over L levels of:
            codebook_loss + β × commitment_loss
          = MSE(z_q, z_e.detach()) + β × MSE(z_q.detach(), z_e)
```

**为什么用 MSE：**

MSE 是高斯噪声假设下的最大似然估计：

```
假设: x ≈ x_recon + ε，  ε ~ N(0, σ²I)
最大化 log p(x|x_recon) ⟺ 最小化 ||x - x_recon||² ⟺ 最小化 MSE
```

**梯度流向：**

```
x_recon ──── Decoder ──── z_q
                            │  (Straight-Through Estimator)
                            ▼
                           z_e ──── Encoder ──── x（不更新）

codebook ◄── codebook_loss（独立更新）
```

---

## 10. 训练器 Trainer

**文件：** `rq/trainer.py`

```python
class Trainer:
    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()
        for batch_idx, data in enumerate(iter_data):
            data = data.to(self.device)
            self.optimizer.zero_grad()

            out, rq_loss, indices = self.model(data)                  # 前向
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)  # 计算损失

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        """评估指标：碰撞率（collision rate）"""
        self.model.eval()
        indices_set = set()
        num_sample = 0
        for data in valid_data:
            indices = self.model.get_indices(data)         # 只用 Encoder + RVQ
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        # 碰撞率 = 有相同 SID 的物品占比
        collision_rate = (num_sample - len(indices_set)) / num_sample
        return collision_rate
```

**模型保存策略（双轨制）：**

```
best_loss_model.pth         ← 训练损失最低的检查点
best_collision_model.pth    ← 碰撞率最低的检查点（推荐用于生成 SID）

同时维护一个大小为 save_limit=5 的滑动窗口，
自动删除既不在"最优"堆中、也不在"最新"队列中的检查点
```

**优化器与学习率调度：**

```python
# 默认：AdamW + Constant LR with Warmup
optimizer = optim.AdamW(params, lr=1e-3, weight_decay=0.0)
scheduler = get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps = warmup_epochs × steps_per_epoch
)
```

---

## 11. SID 生成（推理阶段）

训练完成后，用 `generate_indices.py` 或 `generate_indices_plus.py` 生成 SID 文件：

```python
@torch.no_grad()
def get_indices(self, xs, use_sk=False):
    """推理：只需 Encoder + RVQ，不需要 Decoder"""
    x_e = self.encoder(xs)
    _, _, indices = self.rq(x_e, use_sk=use_sk)
    return indices  # [N, L]
```

**去重处理（Polars 实现）：**

```python
# 若两个物品 SID 完全相同（碰撞），追加层级排名作为第4个 token
def deal_with_deduplicate(df):
    df = df.with_row_index()
    result_df = df.with_columns(
        pl.when(pl.len().over("codes") > 1)
        .then(
            pl.col("codes").list.concat(
                pl.col("index").rank(method="ordinal").over("codes").cast(pl.Int64)
            )
        )
        .otherwise(pl.col("codes"))
        .alias("codes")
    ).drop("index")
    return result_df
```

**SID Token 格式：**

```python
# 原始码字（0-based） → Token（1-based，避免与 padding 冲突）
for level_idx, val in enumerate(codes):
    prefix = chr(97 + level_idx)          # 'a', 'b', 'c', ...
    token  = f"<{prefix}_{val + 1}>"      # +1 offset

# 示例输出：
# 物品 0（SUPCO Hard Start Kit）:  ['<a_22>', '<b_101>', '<c_137>']
# 物品 1（办公椅）:                 ['<a_57>', '<b_4>',  '<c_7>']
```

**最终 JSON 格式（输入 SFT 训练）：**

```json
{
  "0": ["<a_22>", "<b_101>", "<c_137>"],
  "1": ["<a_57>", "<b_4>",  "<c_7>"],
  "2": ["<a_155>", "<b_91>", "<c_16>", "<d_2>"]
}
```

> `<d_2>` 表示发生了碰撞，通过追加第4层（排名2）消除。

---

## 12. 运行命令

**训练 RQ-VAE：**

```bash
cd rq/

python rqvae.py \
    --data_path ../data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
    --ckpt_dir  ./output/Industrial_and_Scientific \
    --lr        1e-3 \
    --epochs    10000 \
    --batch_size 20480 \
    --num_emb_list 256 256 256 \
    --e_dim     32 \
    --layers    2048 1024 512 256 128 64 \
    --kmeans_init True \
    --kmeans_iters 100 \
    --sk_epsilons 0.0 0.0 0.0 \
    --device    cuda:0
```

**生成 SID 索引文件：**

```bash
# RQ-VAE 训练后
python rq/generate_indices.py \
    --data_path  data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
    --ckpt_path  rq/output/Industrial_and_Scientific/best_collision_model.pth

# RQ-Kmeans+ 训练后
bash rq/generate_indices_plus.sh
```

---

## 13. 关键超参数说明

| 参数 | 默认值 | 含义 | 调参建议 |
|-----|-------|-----|---------|
| `num_emb_list` | `[256,256,256]` | 各层 codebook 大小 | 增大→词表更丰富，碰撞更少 |
| `e_dim` | `32` | 量化空间维度 | 太小→表达力不足；太大→量化区分度下降 |
| `layers` | `[2048,1024,512,256,128,64]` | Encoder 隐藏层 | 随 in_dim 调整，保持渐进压缩 |
| `beta` | `0.25` | commitment loss 权重 | VQ-VAE 论文推荐；增大→encoder 更保守 |
| `kmeans_init` | `True` | 是否用 K-means 初始化 | 强烈推荐开启，防 codebook collapse |
| `kmeans_iters` | `100` | K-means 迭代次数 | 数据量大时可适当增加 |
| `sk_epsilons` | `[0,0,0]` | Sinkhorn epsilon | `>0` 时启用均衡分配，数据量小时有效 |
| `quant_loss_weight` | `1.0` | RVQ 损失权重 | 增大→codebook 学习更激进 |
| `lr` | `1e-3` | 学习率 | AdamW 配合 constant schedule |
| `epochs` | `10000` | 训练轮数 | 以碰撞率收敛为准，通常 5000-10000 |
| `batch_size` | `20480` | 批大小 | 越大 K-means 初始化越准确 |
| `eval_step` | `50` | 评估间隔 | 每50 epoch 计算一次碰撞率 |
