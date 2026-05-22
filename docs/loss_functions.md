# MOLK 损失函数详解

## 总览

MOLK 的训练损失由 6 个部分组成：

```
L = L_bpr + L_reg + L_align + L_moe_balance + L_fusion_balance + λ * L_penalty
```

| 损失 | 来源 | 作用 | 系数参数 |
|------|------|------|----------|
| `L_bpr` | `criterion.BPR` | 推荐排序 | — |
| `L_reg` | `criterion.BPR` 内部 | L2 正则化 | `--reg_coeff` |
| `L_align` | `criterion.MSE` | 模态对齐 | `--align_coeff` |
| `L_moe_balance` | `model.py` MoE_Layer | 专家负载均衡 | `--moe_balance_coeff` |
| `L_fusion_balance` | `model.py` MOME_model | 模态融合均衡 | `--fusion_balance_coeff` |
| `L_penalty` | `session.py` | 方差不变性 | `--penalty_coeff` |

---

## 1. BPR Loss（贝叶斯个性化排序损失）

**代码位置**：`criterion.py` BPR 类，第 80-84 行

**输入**：`user_emb [n_user, 128]`、`item_emb [n_item, 128]`、三元组索引 `(u, i, j)`

**计算过程**：

```python
user     = user_emb[u]       # [batch, 128]  用户向量
pos_item = item_emb[i]       # [batch, 128]  正样本（用户交互过的物品）
neg_item = item_emb[j]       # [batch, 128]  负样本（用户未交互的物品）

pos_scores = (user * pos_item).sum(dim=1)  # [batch] 用户-正样本内积
neg_scores = (user * neg_item).sum(dim=1)  # [batch] 用户-负样本内积

L_bpr = mean(softplus(neg_scores - pos_scores))
```

**数学表达**：

$$L_{bpr} = \frac{1}{|B|} \sum_{(u,i,j) \in B} \log(1 + e^{s_{uj} - s_{ui}})$$

其中 $s_{ui} = \mathbf{u}^\top \mathbf{i}$ 是用户 $u$ 对物品 $i$ 的预测分数。

**直觉**：鼓励正样本分数高于负样本分数。当 `pos_scores > neg_scores` 时，`neg - pos` 为负，`softplus` 接近 0，loss 很小。反之 loss 很大，梯度会推动模型修正。

**softplus vs sigmoid**：等价于 $-\log \sigma(s_{ui} - s_{uj})$，但 `softplus` 在数值上更稳定（避免 log(0)）。

---

## 2. L2 正则化（Reg Loss）

**代码位置**：`criterion.py` BPR 类，第 86-87 行

**计算过程**：

```python
user     = user_emb[u]       # [batch, 128]
pos_item = item_emb[i]       # [batch, 128]
neg_item = item_emb[j]       # [batch, 128]

L_reg = mean(||user||² + ||pos_item||² + ||neg_item||²)
```

**数学表达**：

$$L_{reg} = \frac{1}{|B|} \sum_{(u,i,j) \in B} \left( \|\mathbf{u}\|_2^2 + \|\mathbf{i}\|_2^2 + \|\mathbf{j}\|_2^2 \right)$$

**作用**：惩罚 embedding 向量的模长，防止过拟合。权重由 `--reg_coeff`（默认 1e-6）控制。

---

## 3. Alignment Loss（模态对齐损失）

**代码位置**：`session.py` 第 71 行，使用 `criterion.MSE`

**输入**：门控融合后的 `item_emb`、MoE 处理后的 `image_emb` 和 `text_emb`

**计算过程**：

```python
user_emb, item_emb, text_emb, image_emb, _, _ = self.model()

L_align = MSE(item_emb, text_emb) + MSE(item_emb, image_emb)
```

**数学表达**：

$$L_{align} = \|\mathbf{e}_{fused} - \mathbf{e}_{text}\|_2^2 + \|\mathbf{e}_{fused} - \mathbf{e}_{image}\|_2^2$$

**作用**：让门控融合后的物品表示同时保留图像和文本的信息。如果融合后的表示偏离某个模态太远，就会被惩罚。权重由 `--align_coeff`（默认 1）控制。

**注意**：这里是对**全量物品**计算的（因为 `item_emb`、`image_emb`、`text_emb` 都是 `[n_item, 128]`），不是只对 batch 内的物品。

---

## 4. MoE Balance Loss（专家负载均衡损失）

**代码位置**：`model.py` MoE_Layer._compute_load_balancing_loss，第 41-60 行

**输入**：MoE gate 网络的 logits `[n_item, num_experts]`

**计算过程**：

```python
gates = softmax(gate_logits, dim=-1)           # [n_item, 4] 每个物品对 4 个专家的权重
importance_per_expert = gates.mean(dim=0)       # [4] 所有物品对每个专家的平均权重
target = [0.25, 0.25, 0.25, 0.25]              # 均匀分布

L_moe_balance = KL(importance_per_expert || target)
```

**数学表达**：

$$L_{moe} = \sum_{k=1}^{K} \bar{g}_k \log \frac{\bar{g}_k}{1/K}$$

其中 $\bar{g}_k = \frac{1}{N} \sum_{n=1}^{N} g_{nk}$ 是第 $k$ 个专家的平均 gate 权重，$K=4$ 是专家数量。

**作用**：防止 gate 网络总是选择同几个专家（"赢者通吃"），鼓励所有专家被均匀使用。这是一个 KL 散度，衡量实际分布与均匀分布的偏离程度。

**计算次数**：每个 MoE_Layer 各算一次，image 和 text 两个 MoE 的 aux_loss 相加：
```python
moe_balance_loss = image_aux + text_aux  # model.py 第 125 行
```

---

## 5. Fusion Balance Loss（模态融合均衡损失）

**代码位置**：`model.py` MOME_model.forward，第 117-124 行

**输入**：门控网络输出的 gate_weights `[n_item, 2]`（第 0 列是 image 权重，第 1 列是 text 权重）

**计算过程**：

```python
gate_outputs = self.gate(combined_emb)           # [n_item, 2]
gate_weights = softmax(gate_outputs, dim=1)       # [n_item, 2]

mean_gate_weights = gate_weights.mean(dim=0)      # [2] 所有物品的平均模态权重
target = [0.5, 0.5]                               # 均匀分布

L_fusion_balance = KL(mean_gate_weights || target)
```

**数学表达**：

$$L_{fusion} = \sum_{m \in \{img, txt\}} \bar{w}_m \log \frac{\bar{w}_m}{0.5}$$

其中 $\bar{w}_m = \frac{1}{N} \sum_{n=1}^{N} w_{nm}$ 是模态 $m$ 的平均 gate 权重。

**作用**：防止门控网络过度依赖某个模态（比如总是给 image 90% 权重），鼓励两个模态被均衡使用。

**与 MoE Balance 的区别**：
- MoE Balance：均衡**同一模态内的 4 个专家**
- Fusion Balance：均衡**两个模态之间的权重**

---

## 6. Penalty Loss（方差不变性损失）

**代码位置**：`session.py` 第 75-87 行

**核心思想**：通过 Dirichlet 分布构造多个"虚拟环境"，每个环境用不同的模态混合比例。如果模型对模态混合比例不敏感，那么不同环境的 BPR loss 应该接近（方差小）。

**计算过程**：

```python
alpha = 0.1  # Dirichlet 浓度参数

# 构造 3 个环境
mix_ratios = [[0.5, 0.5]]                          # 环境 0：等权混合
lam1, lam2 = np.random.dirichlet([alpha, alpha])    # 采样一对随机比例
mix_ratios.append([lam1, lam2])                      # 环境 1：偏向某个模态
mix_ratios.append([lam2, lam1])                      # 环境 2：偏向另一个模态

# 每个环境用 MoE 输出按不同比例混合
env_bpr_losses = []
for ratio in mix_ratios:
    # MoE 投影后按 ratio 混合（不走门控网络）
    item_emb = ratio[0] * image_emb + ratio[1] * text_emb
    env_bpr, _ = self.bpr(user_emb, item_emb, user, pos_item, neg_item)
    env_bpr_losses.append(env_bpr)

# 方差惩罚
L_penalty = Var(env_bpr_losses) = Var([bpr_0, bpr_1, bpr_2])
```

**数学表达**：

$$L_{penalty} = \text{Var}(\{L_{bpr}^{(e)}\}_{e=1}^{E}) = \frac{1}{E} \sum_{e=1}^{E} (L_{bpr}^{(e)} - \bar{L}_{bpr})^2$$

其中 $E=3$ 是环境数量，$L_{bpr}^{(e)}$ 是第 $e$ 个环境下的 BPR loss，$\bar{L}_{bpr}$ 是均值。

**Dirichlet 采样说明**：

`Dirichlet([α, α])` 从二维 Dirichlet 分布中采样，产出 `[λ₁, λ₂]`，满足 `λ₁ + λ₂ = 1`，`λ₁, λ₂ > 0`。

- `α = 0.1`：采样结果极端，大概率出现 `[0.95, 0.05]` 或 `[0.05, 0.95]` 这样的偏斜比例
- `α = 1.0`：采样结果均匀，`[0.5, 0.5]` 附近概率高
- `α → ∞`：趋近于始终 `[0.5, 0.5]`

**三个环境的作用**：

| 环境 | 混合比例 | 含义 |
|------|---------|------|
| env 0 | `[0.5, 0.5]` | 图文等权，基准环境 |
| env 1 | `[λ₁, λ₂]` | 随机偏斜（如 `[0.9, 0.1]`） |
| env 2 | `[λ₂, λ₁]` | 对称偏斜（如 `[0.1, 0.9]`） |

**反向传播路径**：

```
L_penalty = Var([bpr_0, bpr_1, bpr_2])
    ↑ 梯度回传
get_env_emb(ratio)
    ↑ 梯度回传
MoE_Layer 的 experts 和 gate 参数
    ↑ 梯度回传
image_feat / text_feat 的投影权重
```

这意味着 penalty loss 会**直接修改 MoE 的专家权重**，迫使它们学到对模态比例不敏感的表示。

**权重**：由 `--penalty_coeff`（默认 50）控制。

---

## 损失计算流程图

```
image_feat ──→ [MoE Experts] ──→ image_emb ──┐
                                              ├─→ gate → item_emb ──┬──→ BPR loss
text_feat  ──→ [MoE Experts] ──→ text_emb  ──┘                      │
                                              │                     ├──→ MSE(item, img) + MSE(item, txt) → align_loss
                                              │                     │
          Dirichlet 采样                      │                     │
              ↓                               │                     │
      ratio = [λ₁, λ₂]                       │                     │
              ↓                               │                     │
  image_emb × λ₁ + text_emb × λ₂ ──→ env_item_emb ──→ BPR(env)    │
              ↓                                                    │
      Var(env_bpr_losses) ──→ penalty_loss ────────────────────────┘
                                                                  │
      MoE gate 权重 → KL(gates || uniform) → moe_balance_loss ────┤
      Fusion gate 权重 → KL(weights || uniform) → fusion_balance ─┤
      L2 正则 ──→ reg_loss ───────────────────────────────────────┘
                                                                  ↓
                                                            总 loss
```

---

## 超参数建议

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `--penalty_coeff` | 50 | 0.1 ~ 100 | 方差惩罚强度，过大会压制主 BPR loss |
| `--alpha` | 0.1 | 0.01 ~ 1.0 | Dirichlet 浓度，越小采样越极端 |
| `--align_coeff` | 1 | 0.1 ~ 10 | 模态对齐强度 |
| `--moe_balance_coeff` | 1 | 0.1 ~ 5 | 专家均衡强度 |
| `--fusion_balance_coeff` | 1 | 0.1 ~ 5 | 模态均衡强度 |
| `--reg_coeff` | 1e-6 | 1e-7 ~ 1e-4 | L2 正则强度 |
