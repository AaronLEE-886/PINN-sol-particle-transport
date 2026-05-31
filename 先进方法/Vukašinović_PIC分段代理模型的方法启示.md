# Vukašinović (2026)：PIC 分段代理模型的方法启示

> 原文：*Accelerating Particle-in-Cell simulations in Tokamak Scrape-off Layer using segmented surrogate models*
> 作者：Nikola Vukašinović, Uroš Urbas, Leon Kos, Ivona Vasileska (LECAD, 卢布尔雅那大学)
> 期刊：Engineering Applications of Artificial Intelligence, Vol. 172, 114332 (2026), IF=8.0
> DOI：10.1016/j.engappai.2026.114332

---

## 一、核心问题与动机

BIT1 是 1D3V 静电并行 PIC/MC 代码，用于 SOL 磁通管等离子体模拟（电子 + 多种离子 + 中性粒子 + 原子分子过程）。典型模拟需要 **1152–2304 核运行 60–90 天**（EU Marconi 超算），严重限制了参数扫描和迭代设计。

**目标**：构建 ML 代理模型代替 PIC 模拟，实现毫秒级预测。

**你的工作中对应**：FD 求解器（秒级）虽远快于 PIC，但若需大规模参数扫描（如 $T_{\text{up}} \in [40,200]$ 全覆盖 + 多 regime）或反问题优化，PINN 的推理速度优势仍有价值。

---

## 二、核心方法：物理信息引导的空间分段

### 2.1 基本思想

| 区域 | 物理特征 | 建模方式 |
|------|---------|---------|
| **鞘层区域** (Sheath) | 德拜尺度、强电场、电荷分离、非准中性 | **独立** XGBoost 子模型 |
| **准中性体等离子体** (Bulk) | 准中性、梯度缓、Spitzer-Härm 主导 | **独立** XGBoost 子模型 |

### 2.2 为什么分段有效

SOL 中鞘层与体等离子体具有**截然不同的物理 regime**：
- 鞘层：势垒 ~ $3T_e/e$，尺度 ~ $\lambda_D$，物理量变化率极大
- 体等离子体：准中性，$T(s)$ 缓变

单一全局模型必须同时拟合两种行为 → 相互干扰 → 精度妥协。分段后每个子模型只需学习**一种物理 regime** → 学习难度降低 → 精度提升（MAPE **3.2%** vs 全局模型明显更差）。

### 2.3 对你的直接启示

你在 **$T_{\text{up}}=40$ eV 时 PINN 误差大**，根本原因之一正是**单一网络难以同时处理上游缓变区和靶板陡变区**：

| 区域 | 你的问题中的特征 | 类比 Vukašinović 的分段 |
|------|----------------|----------------------|
| 上游 ($s \approx 0$) | $T \approx T_{\text{up}}$，梯度小 | Bulk 区域（缓变） |
| 靶板附近 ($s \approx L$) | $T$ 从 $\sim 40$ eV 骤降至 $\sim 0.56$ eV，$dT/ds$ 极陡 | Sheath 区域（陡变） |

**可借鉴方案**：

1. **分段 PINN**：两个子网络分别处理上游和靶板区域，在过渡区施加连续性约束（$C^0$ 和 $C^1$ 连续）
2. **区域自适应配点**：借鉴分段思想，靶板附近加密配点、上游稀疏配点（你已部分实现 `target_refined`）
3. **加权损失分段**：对上游和靶板附近的 PDE 残差赋予不同权重

---

## 三、模型选择：XGBoost vs PINN

| 维度 | XGBoost（Vukašinović） | PINN（你的工作） |
|------|----------------------|----------------|
| **监督/无监督** | 监督学习（需大量 PIC 数据） | 无监督（仅需 PDE + BC） |
| **物理先验形式** | 空间分段 = 软物理先验 | PDE 嵌入损失 = 硬物理先验 |
| **数据效率** | 低（需大量训练数据） | 高（无数据需求） |
| **外推能力** | 有限（LOCO 验证保证泛化） | 理论上可外推 |
| **非线性处理** | 分段降低子区域非线性度 | Fourier 特征 + 多阶段优化 |
| **可解释性** | SHAP/特征重要性 | 梯度分析/敏感性 |

**对你的启示**：如果未来涉及逆问题（反推 $\alpha$、$\kappa_\parallel$ 等参数），可以考虑 **PINN 正问题 + XGBoost 代理**的混合方案——PINN 生成训练数据，XGBoost 实现实时推理。

---

## 四、验证策略：留一曲线交叉验证 (LOCO)

### 4.1 方法

- 训练集由不同输入参数下的输出曲线组成（如 10 组 $T_{\text{up}}$ 对应的温度剖面）
- 每次留出一条完整曲线作为验证集，其余 9 条训练
- 遍历所有组合，取平均性能

### 4.2 为什么有效

- 避免**随机划分**导致的"训练集和测试集来自同一参数条件"的虚假高精度
- 直接测试模型对**全新等离子体条件**的泛化能力
- 更接近实际应用场景（预测未模拟过的工况）

### 4.3 对你的启示

你在 $T_{\text{up}} \in [40,200]$ 的扫描中已经有了天然 LOCO 验证框架：

```
T_up: 40  50  60  70  80  90  100  150  200
       ↑                                   ↑
       └─── LOCO: 每次留一个 T_up 作为测试 ──┘
```

当前你的结果已经提供了这种"留一参数"的误差扫描（$T_{\text{up}}=40$ 误差最大，其他很好），这正是 LOCO 思想的体现。可以：
1. 将这种 LOCO 误差分析**正式化**，作为模型泛化能力的标准报告指标
2. 扩展到其他参数维度（$L$、$\kappa_\parallel$、$\alpha$、$\gamma$）
3. 用于指导**自适应采样**：哪些参数区域需要更多训练/优化

---

## 五、Vukašinović 工作的完整启示清单

### 可直接迁移的

| # | 启示 | 应用到你工作中的具体方式 | 优先级 |
|---|------|------------------------|:------:|
| 1 | **物理信息空间分段** | 将 SOL 分为上游缓变区和靶板陡变区，用两个子网络或加权分段损失 | ⭐⭐⭐ |
| 2 | **LOCO 交叉验证正式化** | 将多 $T_{\text{up}}$ 扫描结果整理为标准 LOCO 验证报告 | ⭐⭐⭐ |
| 3 | **区域自适应配点** | 借鉴分段思想，鞘层区域指数加密配点（在 `target_refined` 基础上加强） | ⭐⭐ |
| 4 | **物理量分段归一化** | 不同区域的温度和梯度量级差异大，可分别归一化 | ⭐⭐ |

### 需要进一步探索的

| # | 启示 | 潜在价值 | 优先级 |
|---|------|---------|:------:|
| 5 | **PINN + XGBoost 混合代理** | PINN 做正问题，XGBoost 做实时反问题推理 | ⭐⭐ |
| 6 | **不确定性量化** | 树模型可给出预测置信区间，PINN 也可通过 ensemble 实现 | ⭐ |
| 7 | **模型蒸馏** | 用 PINN 生成大量高精度数据 → 训练轻量代理模型用于实时应用 | ⭐⭐ |
| 8 | **SHAP 可解释性分析** | 对 PINN 的敏感性分析可借鉴 SHAP 框架 | ⭐ |

---

## 六、对你当前工作的具体改进路径

### 路径 A：分段 PINN（解决 $T_{\text{up}}=40$ eV 极端非线性）

```
输入 s
  │
  ├── 区域判別器 (s < s_cut ?)
  │     │
  │     ├── 上游网络 (s ∈ [0, s_cut])
  │     │   → 正常 Fourier 特征 + MLP
  │     │
  │     └── 靶板网络 (s ∈ [s_cut, L])
  │       → 高频 Fourier 特征 + 更宽 MLP
  │
  └── 过渡区损失: |T_up(s_cut) - T_tar(s_cut)| + |dT_up/ds - dT_tar/ds|
```

**关键参数**：分割点 $s_{\text{cut}}$ 的选择——可固定在 $s=0.8L$ 或自适应。

### 路径 B：加权区域损失

不分网络，但在损失函数中对不同区域赋予不同权重：

$$
\mathcal{L} = w_{\text{bulk}} \cdot \mathcal{L}_{\text{PDE, bulk}} + w_{\text{sheath}} \cdot \mathcal{L}_{\text{PDE, sheath}} + w_{\text{up}} \cdot \mathcal{L}_{\text{up}} + w_{\text{sheath}} \cdot \mathcal{L}_{\text{sheath}}
$$

其中 $w_{\text{sheath, PDE}} \gg w_{\text{bulk, PDE}}$。

### 路径 C：多阶段训练（从易到难）

1. 先训练高 $T_{\text{up}}$（近线性，易学习）→ 获得较好的初始权重
2. 逐步降低 $T_{\text{up}}$，fine-tune 模型 → 迁移学习
3. 最终在 $T_{\text{up}}=40$ eV 上微调

这与 Vukašinović 的分段思想不同但互补——是"参数空间分段"而非"空间域分段"。

---

## 七、团队前期工作脉络（BIT1 PIC 模拟）

| 年份 | 工作 | 对你工作的参考价值 |
|------|------|-----------------|
| 2017 | PIC 动理学建模 JET SOL ELM 输运 | SOL 物理基础 |
| 2018 | ITER 刮削层动理学模拟 (EPS) | 边界条件处理 |
| 2019 | ITER ELM 鞘层特征 (EPS) | 鞘层物理深入理解 |
| 2021 | BIT1-SOLPS-ITER 动理学-流体耦合 | 多尺度耦合方法论 |
| **2026** | **ML 分段代理模型加速 PIC** | **当前启示来源** |

这一演进（纯动理学 → 动理学-流体耦合 → ML 加速）与你的路径（FD 求解 → PINN 求解 → ML 加速反问题）存在**结构上的平行性**：都是从高保真但昂贵的求解器向兼顾效率与精度的代理模型演进。

---

## 八、参考文献

- Vukašinović, N., Urbas, U., Kos, L., & Vasileska, I. (2026). Accelerating Particle-in-Cell simulations in Tokamak Scrape-off Layer using segmented surrogate models. *Engineering Applications of Artificial Intelligence*, 172, 114332. https://doi.org/10.1016/j.engappai.2026.114332
- Vasileska, I., & Kos, L. (2017–2022). BIT1 PIC simulations of tokamak SOL [multiple conference papers: NENE, EPS].
- Stangeby, P. C. (2000). *The Plasma Boundary of Magnetic Fusion Devices*. IoP Publishing. [经典 SOL 物理参考]
