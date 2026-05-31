# DIV1D 对你的 PINN-SOL 项目的参考价值分析

> DIV1D: *Benchmark of a self-consistent dynamic 1D divertor model DIV1D using the 2D SOLPS-ITER code*
> 作者：G. L. Derks, J. P. K. W. Frankemölle, J. T. W. Koenders, M. van Berkel, H. Reimerdes, M. Wensing, E. Westerhof
> 期刊：*Plasma Physics and Controlled Fusion* **64**, 125013 (2022)
> 扩展：多机器 benchmark, *PPCF* **66**, 055004 (2024)，覆盖 TCV / MAST-U / ASDEX Upgrade

---

## 一、DIV1D 的核心特征

### 1.1 物理模型

DIV1D 是**时变 1D 流体模型**，沿磁力线从靶板到滞止点（或两靶之间）求解四组守恒方程：

| 方程 | 变量 | 包含的物理 |
|------|------|-----------|
| 粒子守恒 | $n$ | 电离、复合 |
| 动量守恒 | $v_\parallel$ | 电荷交换摩擦、复合动量损失 |
| 能量守恒 | $T$ | Spitzer-Härm 热导、电离/激发/辐射/复合能量损失、杂质辐射 |
| 中性粒子守恒 | $n_n$ | 再循环、充气、扩散近似输运 |

**基础假设**：$T_i = T_e$，准中性，无杂质（$n_e = n_i = n$），平行输运主导。

### 1.2 关键技术特色

| 特色 | 描述 | 对你工作的启示 |
|------|------|--------------|
| **有效磁通扩张** $f_{\text{exp}}(x) = B_u / B(x)$ | 用磁通管截面变化模拟 2D 横向输运的 1D 等效 | **可以直接引入你的 PINN 模型** |
| **外部中性粒子背景** $n_b$ | 中性粒子与背景储库交换，建模横向中性输运 | 如你将来加入中性粒子，可参考 |
| **滞止点边界条件** | $v_\parallel = 0, q_\parallel = 0$ 的自然边界 | 不同于你的上游 Dirichlet BC，但概念上等价 |
| **1D 映射程序** | 从 2D SOLPS-ITER 结果中提取 1D 剖面用于比对 | 可借鉴作为你的 FD 验证方法的形式化 |
| **多机器验证** | TCV → AUG → MAST-U，~50% 一致性 | 验证方法论值得效仿 |

### 1.3 与你的 PINN-SOL 的模型对比

| 维度 | DIV1D | 你的 PINN-SOL |
|------|-------|--------------|
| **维度** | 1D 时变（$\partial/\partial t$） | 1D 稳态（$\partial/\partial t = 0$） |
| **方程数** | 4 个耦合守恒方程 | 1 个非线性热传导方程 |
| **变量** | $n, v_\parallel, T, n_n$ | 仅 $T$ |
| **中性粒子** | 扩散模型 + 背景交换 | 无 |
| **原子物理** | 电离、复合、CX、激发、辐射 | 无（$S=0$ 或简化源） |
| **杂质** | 简化杂质辐射 $n^2 \xi_Z L_Z(T)$ | 无 |
| **横向输运** | 有效磁通扩张模拟 | 无 |
| **数值方法** | 有限体积（显式/隐式时间推进） | PINN（自动微分 + Adam/L-BFGS） |
| **时间依赖** | ✅ 动态，可模拟脱靶演化 | ❌ 仅稳态 |
| **验证** | 多机器 vs SOLPS-ITER | vs FD 求解器 |
| **计算速度** | 快（1D FV） | 推理极快（训练中等） |

---

## 二、对你最有价值的启示（按优先级排列）

### ⭐⭐⭐ 启示 1：有效磁通扩张 —— 弥补 1D 模型的最大短板

**DIV1D 的做法**：
引入随位置变化的磁通管截面积 $A(x)$，通过 $A(x)/A(0) = B_u / B(x)$ 模拟磁通扩张效应，从而在 1D 模型中隐含包含横向输运的部分影响。

**你的收益**：
你的 1D 模型目前使用**均匀截面**。引入有效磁通扩张后：

$$
\frac{d}{ds}\left(A(s)\, \kappa_\parallel T^{5/2} \frac{dT}{ds}\right) + A(s) \cdot S(s) = 0
$$

这个修改不会改变 PDE 的基本形式，只需在损失函数中加入 $A(s)$：

$$
\mathcal{L}_{\text{PDE}} = \frac{1}{N_c}\sum_i \left[\frac{1}{A(s_i)}\frac{d}{ds}\left(A(s_i)\,\kappa_\parallel T_\theta^{5/2}\frac{dT_\theta}{ds}\right) + S(s_i)\right]^2
```

**实现难度**：低 — 只需在模型中添加一个 $A(s)$ 函数，自动微分可自然处理。

### ⭐⭐ 启示 2：多场耦合 — 从 $T$ 扩展到 $T + n$

DIV1D 求解 $n, v_\parallel, T$ 的耦合系统。你的 PINN 目前仅预测 $T$，而 SOL 物理中密度和速度的演化对热输运有反馈。

**扩展路径**（分步）：
1. **第一阶段**：保持 PDE 不变，加入 $n(s)$ 作为辅助输出，通过等压假设 $n(s)T(s) = p_0$ 约束（你已有 $p_0$ 常数）
2. **第二阶段**：加入粒子守恒方程，让 PINN 同时预测 $T(s)$ 和 $n(s)$

$$
\text{NN}(s; \theta) \rightarrow [T(s), n(s)]
$$

$$
\mathcal{L} = w_T \mathcal{L}_{\text{PDE,T}} + w_n \mathcal{L}_{\text{PDE,n}} + w_{\text{coupling}} \mathcal{L}_{\text{iso-baric}} + w_{\text{BC}} \mathcal{L}_{\text{BC}}
$$

### ⭐⭐ 启示 3：时变扩展 — 从稳态到动态

DIV1D 最显著的特征是时间依赖。你已用多 $T_{\text{up}}$ 扫描展示了稳态行为，但时变能力（如模拟 ELM 热脉冲传播）会极大提升工作价值。

**借鉴 DIV1D 的源项处理**：DIV1D 通过体积源项引入来自芯部的粒子/热流：

$$
S_{\text{core}}(x) = \frac{\Gamma_{\text{core}} \cdot f_{\text{profile}}(x)}{V}, \quad
Q_{\text{core}}(x) = \frac{q_{\text{core}} \cdot f_{\text{profile}}(x)}{V}
$$

代入你的瞬态热传导方程：

$$
\frac{\partial T}{\partial t} = \frac{1}{A(s)}\frac{\partial}{\partial s}\left(A(s)\,\kappa_\parallel T^{5/2}\frac{\partial T}{\partial s}\right) + Q_{\text{core}}(s)
$$

将时间离散化后，用 PINN 做每个时间步的推进：

```
T^{n+1} = T^n + Δt · PDE_PINN(T^n, parameters)
         ↓
    自回归循环（参考 Csala 的长视野训练经验）
```

### ⭐ 启示 4：验证方法论

DIV1D 的 1D 映射程序和 multi-machine benchmark 对你的验证策略有参考价值：

| DIV1D 的做法 | 你的可借鉴点 |
|-------------|------------|
| 从 2D SOLPS-ITER 映射到 1D 剖面 | 从你的 FD 求解器生成 benchmark 数据集的形式化流程 |
| 多机器（TCV → AUG → MAST-U）交叉验证 | 多 $T_{\text{up}}$ / 多 regime 交叉验证（你已做到部分） |
| 明确报告误差范围和系统偏差 | 在你的 LOCO 验证中正式化 |
| 使用实验数据验证模型假设 | 如未来有实验数据（如 EAST/JET 探针测量），可效仿 |

### ⭐ 启示 5：原子物理的简化处理

DIV1D 的源项处理方式值得注意——尽管物理过程复杂（电离、复合、CX、激发、辐射），但每个过程都用**简洁的解析公式**表示，参数化为温度和密度的函数。

**对你的意义**：当你从 $S=0$ 扩展到有源项时，可以像 DIV1D 一样：
- 将杂质辐射简化为 $P_{\text{rad}} = n^2 L_Z(T)$（ADAS 率系数查询）
- 将中性粒子电离损失简化为 $\varepsilon_{\text{ion}} n_n n \langle \sigma v \rangle_{\text{ion}}$
- 这些函数形式对 PINN 的自动微分完全透明

---

## 三、快速对照表：DIV1D 各特征对你的价值

| DIV1D 特征 | 对你的价值 | 实现难度 | 推荐优先级 |
|-----------|:---------:|:--------:|:---------:|
| 有效磁通扩张 $A(s)$ | ⭐⭐⭐ 弥补 1D 横向输运缺失 | 低 | **#1** |
| 时变框架 | ⭐⭐ 从稳态到动态 | 中 | **#2** |
| 多场耦合 $T + n$ | ⭐⭐ 改善物理完整性 | 中 | **#3** |
| 中性粒子扩散模型 | ⭐ 若需脱靶模拟 | 高 | #4 |
| 原子物理源项 | ⭐ 增强物理真实性 | 中 | #5 |
| 验证方法论 | ⭐ 提升工作严谨性 | 低 | 持续 |
| 滞止点 BC | ⭐ 若扩展域 | 低 | 按需 |

---

## 四、结论

**DIV1D 对你的项目有参考价值，但更多是"方向性的"而非"直接可迁移的"**。

核心结论：

1. **最有价值的单一启示**：有效磁通扩张 $A(s)$ — 这是最简单、收益最高的 1D 模型改进，可以直接融入你的 PINN 而无需改变网络结构
2. **最值得借鉴的方法论**：DIV1D 的 1D 映射 + 系统化 benchmark 流程 — 可将你的多 $T_{\text{up}}$ 扫描升级为更严格的形式化验证
3. **DIV1D 相对你的 PINN 的不足**：
   - DIV1D 使用传统 FV 方法，推理速度不如 PINN（虽然 1D FV 也已很快）
   - DIV1D 无法直接处理反问题（你的 PINN 天然适合）
   - DIV1D 需要显式输入原子数据（你的纯 PDE 形式更简洁）
4. **DIV1D 和你的工作的互补性**：DIV1D 在物理完整性上领先（4 方程耦合），你在求解方法上独特（PINN 无监督 + 自动微分）。两者的结合——**用 DIV1D 的物理框架指导 PINN 的扩展方向**——是一条清晰的演进路径。

---

## 五、参考文献

- Derks, G. L., et al. (2022). Benchmark of a self-consistent dynamic 1D divertor model DIV1D using the 2D SOLPS-ITER code. *Plasma Phys. Control. Fusion*, **64**, 125013.
- Derks, G. L., et al. (2024). Multi-machine benchmark of the self-consistent 1D scrape-off layer model DIV1D from stagnation point to target with SOLPS-ITER. *Plasma Phys. Control. Fusion*, **66**, 055004.
