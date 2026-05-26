## 第 1 天：建立推荐系统全局地图

### 目标

知道工业推荐系统和 MiniOneRec 分别处在什么位置。

### 学习内容

```text
推荐系统基本架构
召回、粗排、精排、重排
Top-K 推荐 vs CTR 预估
显式反馈 vs 隐式反馈
序列推荐
生成式推荐
```

### 需要搞清楚

```text
传统推荐：user/item embedding → 打分排序
序列推荐：用户历史序列 → 预测下一个 item
生成式推荐：用户历史 → 生成 item token / semantic id
```

### 当天任务

1. 画一张推荐系统架构图：

```text
用户行为日志
↓
召回：双塔 / 协同过滤 / 向量召回
↓
排序：DeepFM / DIN / DCN
↓
重排：多样性 / 规则 / 业务约束
↓
推荐结果
```

2. 写一页笔记：

```text
MiniOneRec 属于哪里？
它解决传统 ID 推荐的什么问题？
为什么要把 item 变成 Semantic ID？
```

### 产出

```text
notes/day01_recsys_overview.md
```

---
