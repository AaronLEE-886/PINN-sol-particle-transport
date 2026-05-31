# 传统（非神经网络）SOL 刮削层输运模拟方法综述

> 涵盖解析模型、流体代码、动理学代码、中性粒子模型及验证确认
> 重点关注 2015–2025 年进展，同时覆盖 Stangeby、Pitcher 等经典工作

---

## 一、SOL 输运的物理基础与解析模型

### 1.1 两点模型（Two-Point Model）

两点模型是 SOL 物理的基石，连接上游（midplane）与靶板（target）两个位置沿磁力线的状态。

**Pitcher & Stangeby (1997)** — *Experimental divertor physics*, *Plasma Phys. Control. Fusion* **39**, 779–930.  
经典综述文章，系统给出了两点模型的三条基本方程：

1. **静压平衡**（动量守恒）：$2n_t T_t = n_u T_u$  
   （上游流速 $v_u=0$，靶板离子加速至声速 $c_{st} = \sqrt{2kT_t/m_i}$）

2. **平行热传导**（Spitzer-Härm）：$T_u^{7/2} = T_t^{7/2} + \dfrac{7}{2} \dfrac{q_\parallel L}{\kappa_{0e}}$

3. **鞘层能流传输**：$q_\parallel = \gamma\, n_t\, k T_t\, c_{st}$，$\gamma \approx 7$

**Stangeby (2000)** — *The Plasma Boundary of Magnetic Fusion Devices*, IoP Publishing.  
SOL 物理的"圣经"，717 页专著。详细给出两点模型的系统推导、传导限制区与鞘层限制区的判定准则，以及功率损失因子 $f_{\text{power}}$、动量损失因子 $f_{\text{mom}}$ 等重要扩展。

**核心判据**：
- 传导限制区（Conduction-limited）：$\dfrac{n_u L}{T_u^2} \gtrsim 1.5 \times 10^{17}$，或 SOL 碰撞率 $\nu^*_{\text{SOL}} \gtrsim 15$
- 鞘层限制区（Sheath-limited）：碰撞率低，温度剖面平坦

### 1.2 鞘层边界条件理论框架

**Bohm 鞘层判据**：离子进入鞘层的速度 $\geq c_s = \sqrt{e(T_e+T_i)/m_i}$

**鞘层热传输系数 $\gamma$**：
- 电子热输运 ~2
- 离子加速能 ~0.5
- 鞘层加速能 ~3
- 表面复合能 ~1.5
- **合计 $\gamma \approx 7$**（典型范围 5–8）

**二次电子发射（SEE）**：发射系数 $\delta$ 增加时有效 $\gamma$ 降低，显著影响靶板热负载（Stangeby 2000, Ch. 2）。

### 1.3 解析模型的推广

- **多点模型**：在两点模型基础上引入中间点，考虑体积热源/汇和杂质辐射
- **DIV1D** (Derks et al., 2023)：1D 动态偏滤器模型，从靶板到滞止点完整覆盖，用 SOLPS-ITER 验证（TCV, AUG, MAST-U）
- **ReMKiT1D**：反应性多流体 + 电子动理学耦合 1D 框架，支持 Spitzer-Härm 与动理学修正耦合
- **2POINTSOL 模块**：OMFIT 框架中实现的两点模型工具

---

## 二、大型 SOL 模拟代码套件

### 2.1 SOLPS / SOLPS-ITER（行业标准）

**历史沿革**：

| 版本 | 组件 | 时间 | 特色 |
|------|------|------|------|
| SOLPS4.0 | B2 + EIRENE₉₆ | 1990s | 原始耦合 |
| SOLPS4.2 | B2 + EIRENE₉₉ | 2000s | ITER 偏滤器设计主力 |
| SOLPS5.0 | B2.5 + EIRENE₉₉ | ~2009 | 加入 E×B 和 diamagnetic 漂移、电流 |
| SOLPS5.1 | B2.5 + EIRENE₉₉ | ~2009 | 融合 ITER 特定物理 |
| **SOLPS-ITER** | B2.5 + EIRENE₂₀₁₀ | **2015–今** | ITER 统一标准版本 |

**关键文献**：
- Wiesen et al., *J. Nucl. Mater.* **463**, 480 (2015) — SOLPS-ITER 初次发布
- Bonnin et al., *Plasma Fusion Res.* **11**, 1403102 (2016) — SOLPS-ITER 代码包介绍
- Rozhansky et al., *Nucl. Fusion* **49**, 025007 (2009) — B2.5 漂移公式体系
- Reiter et al., *Fusion Sci. Technol.* **47**, 172 (2005) — EIRENE 代码手册

**B2.5 流体求解器能力**：
- 平行电流与电势
- E×B 和 diamagnetic 漂移
- 离子/中性粒子热流和粒子流通量限制
- 壁材料混合和表面温度演化
- 重杂质（钨）捆绑处理

**EIRENE 中性粒子 Monte Carlo**：
- 原子分子数据库（AMJUEL, HYDHEL）
- 表面相互作用模型（反射、再循环、溅射 via TRIM）
- 时间相关模式（用于 ELM 等非稳态过程）
- MPI 并行化

**2022 新版特性**：
- 非结构有限体积能力（任意网格）
- 广义鞘层边界条件（任意磁场入射角）
- 与 EIRENE 的自洽耦合
- 磁力线不对齐单元的正确数值处理
- 计算域从 SOL 扩展到全真空室截面

### 2.2 EDGE2D-EIRENE（JET 主力）

与 SOLPS 同源（Braginskii 流体 + EIRENE），但在 JET 等离子体-壁相互作用研究中应用最广。

**关键文献**：
- Guillemaut et al., *Nucl. Fusion* (2014) — JET 碳壁与铍/钨壁脱靶模拟
- Wiesen et al., *Phys. Scr.* (2017) — JET 金属第一壁边缘建模综述
- Rikala, Groth et al., *Contrib. Plasma Phys.* (2024) — OEDGE vs EDGE2D-EIRENE 比对

**验证结果**：
- 低再循环条件（$\nu_e \approx 30$）：与实验吻合 10–25%
- 高再循环条件（$\nu_e \approx 55000$）：差异可达 50%（偏滤器 SOL）

### 2.3 UEDGE（LLNL 开发）

25 年 + 历史的成熟 SOL 代码，美国聚变项目的主力。

**关键文献**：
- Rensink et al., *Contrib. Plasma Phys.* **36**, 157 (1996) — UEDGE 模型与应用概论
- Rognlien et al., *J. Nucl. Mater.* (1998–2006) — DIII-D 流模拟、主腔再循环
- Dudson et al., *Comput. Phys. Commun.* (2009) — BOUT++ 框架（UEDGE 延伸）

**物理模型**：
- Braginskii 流体方程 + 反常横向输运
- 自适应隐式时间推进
- 离子温度各向异性（$T_{i\parallel}$ vs $T_{i\perp}$）
- 与 DEGAS-2（动理学 MC 中性粒子）耦合 via JFNK 方法
- 多电荷态杂质（C, Ne, W 等）

**2025 最新进展**：
- 贝叶斯优化框架推断反常输运系数（含不确定度量化）
- 支持 snowflake 和 Super-X 偏滤器几何

### 2.4 SOLEDGE2D / SOLEDGE-HDG / SOLEDGE3X（CEA IRFM，法国）

- **SOLEDGE-HDG**：使用 Hybridizable Discontinuous Galerkin 方法，高精度处理复杂壁几何
- **SOLEDGE3X**：扩展到 3D 湍流模拟，全极向截面模拟能力（WEST, ITER, JET）

**关键文献**：
- Rivals et al., *Nucl. Fusion* **65**, 026038 (2025) — SOLEDGE3X 全容器 ITER 模拟 vs SOLPS-ITER
- Tamain et al., *J. Comput. Phys.* **321**, 606 (2016) — TOKAM3X（关联湍流代码）

**特色应用**：
- WEST 全局粒子累积模拟（充气扫描，鞘层限制→高再循环→脱靶全 regime）
- ITER Limiter-Divertor 过渡模拟（电流 ramp-up 阶段）
- 简化 $k$–$\varepsilon$ 湍流强度预测模型

### 2.5 湍流代码

| 代码 | 开发单位 | 特色 | 关键文献 |
|------|---------|------|---------|
| **GBS** | EPFL 瑞士等离子体中心 | drift-reduced Braginskii，全 $f$，通量驱动，含中性粒子 | Ricci et al., *PPCF* **54**, 124047 (2012) |
| **TOKAM3X** | CEA IRFM + M2P2 | 全环面漂移简化 Braginskii，通量驱动，含 X 点 | Tamain et al., *JCP* **321**, 606 (2016) |
| **BOUT++** | 开源（Dudson 等） | 模块化 3D 流体框架，STORM 模块，Landau 流体闭包 | Dudson et al., *CPC* **180**, 1467 (2009) |
| **GRILLIX** | IPP Garching | FCI 方法，无坐标奇异性，等温静电 | Riva et al., *PPCF* **58**, 044005 (2016) |

**多代码验证**：
- Riva et al. (2016) TORPEX blob 多代码验证（BOUT++, GBS, HESEL, TOKAM3X）——所有 3D 代码与实验吻合良好

---

## 三、1D SOL 模拟方法

### 3.1 1D 沿磁力线模型

1D SOL 模型是连接解析两点模型和完整 2D 模拟的桥梁。

**数值方法**：
- **有限差分**（中心差分 / 迎风格式）— 你的 FD 求解器使用的方法
- **有限体积** — 保证通量守恒，更适合非线性问题
- **打靶法** — 对 BVP 问题有效，但高度非线性时收敛困难

**非线性迭代**：
- **Picard 迭代**：以 $T^{5/2}$ 为系数，固定系数求解 → 更新系数。收敛慢但稳健
- **Newton 迭代**：快收敛但需要良好的初值
- **Newton-Krylov**（JFNK）：无需求解 Jacobian 显式矩阵（UEDGE 使用）

### 3.2 DIV1D（动态 1D 偏滤器模型）

**Derks et al. (2023–2024)**，EUROfusion / TCV 合作：
- 从靶板到滞止点的 1D 流体模型
- 假设平行输运主导横向输运
- 在 TCV, AUG, MAST-U 上用 SOLPS-ITER 验证
- 密度、温度、热流、速度剖面的良好一致性
- 自洽捕捉脱靶演化与上游密度关系

### 3.3 ReMKiT1D

- 英国 UKAEA 开发的 1D 反应性多流体 + 电子动理学框架
- 支持非线性 ODE/PDE 系统
- 电子流体模型（Spitzer-Härm）与动理学修正耦合
- 用于研究非局域平行热输运

### 3.4 1D 模型的角色与局限

**优势**：计算成本极低、物理透明、易于解析分析
**局限**：
- 不能处理横向输运
- 无法描述 X 点几何效应
- 无法处理 2D/3D 湍流结构
- 在深度脱靶条件下的中性粒子动力学需要 2D 处理

---

## 四、核心数值方法

### 4.1 空间离散

- **有限体积法**（主流选择）：在 SOL 模拟中占主导。直接离散守恒形式，保证通量守恒
- **SOLEDGE-HDG**：使用混合不连续 Galerkin（HDG）方法，对复杂几何高精度
- **FCI 方法**（GRILLIX）：通量坐标独立，无坐标奇异性

### 4.2 非线性迭代策略

| 方法 | 收敛速度 | 稳健性 | 使用代码 |
|------|---------|--------|---------|
| Picard | 线性 | 高 | B2.5, EDGE2D |
| Newton | 二次 | 依赖初值 | UEDGE（可选） |
| JFNK | 超线性 | 高 | UEDGE + DEGAS-2 耦合 |
| 塞德尔/多重网格 | 线性-超线性 | 高 | B2.5 |

### 4.3 时间推进

- **全隐式**（UEDGE）：允许大时间步长，快速收敛到稳态
- **半隐式**（TOKAM3X）：处理湍流时间尺度
- **显式 + 子循环**（GBS）：处理快速电子动力学
- **自适应时间步长**：应对 ELM 等瞬态事件的快速变化

### 4.4 流体-动理学耦合

**SOLPS 范式 (B2.5 + EIRENE)**：
```
流体步 (B2.5) → 等离子体状态 → EIRENE MC 步 (中性粒子)
     ↑                                      │
     └────────── 状态更新 ←───────────────────┘
```

**关键挑战**：
- MC 噪声 → 需要足够的 Monte Carlo 粒子数
- 收敛准则 → 外迭代的选择
- 辐射时间尺度与离子输运时间尺度的分离

**UEDGE + DEGAS-2**：
- 使用 Jacobian-Free Newton-Krylov 方法同时求解流体和 MC 中性粒子
- 显著提高耦合收敛效率（2017 年突破）

---

## 五、中性粒子与杂质物理

### 5.1 中性粒子模型

| 模型 | 方法 | 使用代码 | 特色 |
|------|------|---------|------|
| EIRENE | 线性动理学 MC | SOLPS, EDGE2D, SOLEDGE | 原子分子数据库 AMJUEL/HYDHEL，MPI 并行 |
| DEGAS 2 | 线性动理学 MC | UEDGE | 与 UEDGE 的 JFNK 耦合 |
| 扩散中性流体 | 流体近似 | UEDGE（默认）, SOLEDGE-HDG | 计算高效，低碰撞率时精度降 |
| 动理学中性子模块 | Method of Characteristics | GBS | 全 $f$ 湍流 + 中性耦合 |

**EIRENE 核心能力**：
- 电离、复合、电荷交换、弹性碰撞
- 表面反射、再循环、溅射（TRIM 数据库）
- 中性-中子和光子-中性碰撞（非线性效应）
- 时间相关模式（ELM 模拟）

### 5.2 杂质输运模型

| 模型 | 精度 | 成本 | 适用场景 |
|------|:----:|:----:|---------|
| Corona Equilibrium | 低 | 低 | 快速估计 |
| Collisional-Radiative | 高 | 中-高 | 精确模拟（ADAS 数据库） |
| 杂质捆绑 | 中 | 低 | 重杂质（如 W）多电荷态简化 |

### 5.3 杂质辐射与偏滤器脱靶

**scr impurity 种子与脱靶控制**（大量 SOLPS/UEDGE 工作）：

| 杂质 | 辐射效率 | 偏滤器压缩 | 芯部影响 | 稀释效应 |
|:----:|:--------:|:----------:|:--------:|:--------:|
| Ar (氩) | **高** | 高（低种子量时）| 强 | 低 |
| Ne (氖) | 低 | 较低 | 中-低 | 中 |
| N (氮) | 中 | **最高**（比 Ne 好 2-3×） | 弱 | 高 |

**代表性工作**：
- Newton et al., *Nucl. Fusion* (2025) — SOLPS-ITER Ar vs Ne 对比（球马克）
- Zhang et al., *Contrib. Plasma Phys.* (2024) — UEDGE CFETR 脱靶（Ar 需 0.24%, Ne 需 1.7–2.3%）
- Kaveeva et al., PSI (2021) — SOLPS-ITER Ne vs N JET H-mode
- Ye et al., *Phys. Scr.* (2024) — HL-2A SOLPS-ITER 杂质辐射 rollover 与脱靶关联（$T_e < 5$ eV 触发）

---

## 六、特定物理问题的 SOL 模拟

### 6.1 偏滤器脱靶模拟

**脱靶的物理标志**：
- 靶板粒子通量 roll-over（随上游密度增加先增后减）
- 靶板 $T_e < 5$ eV
- 辐射 front 从靶板向上游迁移
- 体积复合开始主导表面复合

**关键模拟工作**：
- Wiesen et al., *Nucl. Fusion* **65**, 056027 (2025) — SOLPS-ITER vs SOLPS4.3 ITER 偏滤器 benchmark（网格分辨率解决长期差异）
- Park et al., *Nucl. Fusion* **61**, 016021 (2020) — ITER 早期运行阶段偏滤器性能评估
- Yang et al., *Nucl. Fusion* (2024) — 提出 $R_D = P_{\text{rad}} / P_{\text{cond}}$ 作为脱靶控制判据（SOLEDGE3X 验证，WEST/TCV/JT-60U 多机验证）

**脱靶控制的通用准则**：当 $R_D \approx 1$ 时，辐射 front 从靶板表面脱附。

### 6.2 ELM 热脉冲传播

- **平行热流在 ELM 期间增加约 10 倍**
- **鞘层热传输系数增大**（非局域动理学效应，SOL-KiT 研究）
- **丝状结构（filaments）沿磁力线对流输运**
- 两阶段热负载：快电子 → 先导高鞘层电位 → 水平面热流；高能离子 → 单块 leading edge 热流

**代码应用**：
- BOUT++：ELM crash 模拟 + 非局域平行热流模型（Omotani & Dudson, 2013）
- BIT1 PIC：JET ELM 靶板热流动理学模拟
- SOLPS：ELM-averaged 输运模拟

### 6.3 刮削层流（SOL Flow）与等离子体旋转

- 平行 SOL 流由 Pfirsch-Schlüter 流 + 湍流驱动 +
- 实验观测：AUG, DIII-D, JET 上的亚声速-超声速 SOL 流
- UEDGE + DEGAS-2 模拟主腔再循环对 SOL 流的影响

### 6.4 漂移效应

| 漂移类型 | 物理机制 | SOL 中的作用 |
|---------|---------|-------------|
| E×B | $\mathbf{E}\times\mathbf{B}/B^2$ | 极向不对称性、偏滤器靶板热负载不对称 |
| Diamagnetic | $\nabla p \times \mathbf{B}/(nqB^2)$ | 径向输运修正 |
| B×∇B | $(\mathbf{B}\times\nabla B)/B^2$ | 离子-电子分离，驱动 SOL 流 |

**关键文献**：
- Kaveeva et al., *Nucl. Fusion* **60**, 046020 (2020) — SOLPS-ITER ITER 全漂移模拟
- Rozhansky et al., *Nucl. Fusion* **49**, 025007 (2009) — B2.5 漂移公式体系

### 6.5 ITER 偏滤器 SOL 模拟

**SOLPS-ITER 是 ITER 偏滤器设计的官方代码**。

关键里程碑：
- 2020：Park et al. — ITER 早期运行（H, He）偏滤器可操作性窗口
- 2020：Kaveeva et al. — 全漂移模拟，漂移对偏滤器不对称性和脱靶的影响
- 2022：Moscheni et al. — SOLPS-ITER / SOLEDGE2D / UEDGE 三方交叉比对（DTT）
- 2025：Rivals et al. — SOLEDGE3X 全容器 ITER 非活性相模拟 vs SOLPS-ITER
- 2025：Wiesen et al. — **决定性 benchmark**：SOLPS-ITER vs SOLPS4.3 在 ITER 部分脱靶解上实现收敛（提高 4× 极向网格分辨率后差异消除）

---

## 七、验证与确认

### 7.1 代码间 Benchmark

| 比对 | 参与代码 | 关键文献 |
|------|---------|---------|
| ITER 偏滤器基准 | SOLPS-ITER vs SOLPS4.3 | Wiesen et al., *NF* **65** (2025) |
| DTT 低功率比对 | SOLPS-ITER vs SOLEDGE2D vs UEDGE | Moscheni et al., *NF* **62** (2022) |
| ITER 非活性相 | SOLEDGE3X vs SOLPS-ITER | Rivals et al., *NF* **65** (2025) |
| TORPEX blob | BOUT++ vs GBS vs HESEL vs TOKAM3X | Riva et al., *PPCF* **58** (2016) |
| 1D vs 2D | DIV1D vs SOLPS-ITER | Derks et al. (2023–2024) |

### 7.2 主要实验验证案例

| 装置 | 诊断 | 验证的代码 | 物理量 |
|------|------|-----------|--------|
| JET | Langmuir probe, Thomson, 辐射热成像 | EDGE2D-EIRENE | $T_e$, $n_e$, $q_\parallel$ |
| DIII-D | 探针, Thomson, CER | UEDGE | SOL 流、$T_i$、再循环 |
| AUG | 探针, 辐射热成像 | SOLPS-ITER | 脱靶、杂质辐射 |
| C-Mod | 探针, ECE, 辐射热成像 | BOUT++ | SOL 热流宽度 |
| KSTAR | 探针, Thomson, 偏滤器辐射 | SOLPS-ITER, UEDGE | 不对称性脱靶 |

### 7.3 SOL 热流宽度的关键验证

**ITER SOL 热流宽度预测**：$\lambda_q \approx 1$ mm（基于多机标定）
- Goldston 启发式漂移模型（HD 模型）
- BOUT++ 模拟表明：当前装置中漂移主导 → ITER/CFETR 中湍流主导
- 关键论文：Eich et al., *PRL* (2011); *Nucl. Fusion* (2013) — 多机 $\lambda_q$ 标定

---

## 八、综合分析：仍在的挑战

### 8.1 已解决的挑战

- ✅ SOLPS-ITER 与 SOLPS4.3 的 ITER 脱靶解差异（网格分辨率解决，Wiesen 2025）
- ✅ 多代码间基本 SOL 剖面一致性（中等碰撞率条件）
- ✅ 两点模型的实验验证（传导限制区）

### 8.2 当前仍存在的挑战

| 挑战 | 描述 | 代码局限 |
|------|------|---------|
| **深度脱靶** | $T_e < 1$ eV 时流体假设崩溃，分子过程（Maris）重要 | 需动理学处理，EIRENE 分子过程仍在发展 |
| **中性粒子-湍流耦合** | 中性粒子对湍流输运的反馈尚未自洽求解 | GBS 正在推进，但计算成本极高 |
| **3D 效应** | RMP、3D 场导致的三维 SOL 输运 | EMC3-EIRENE 有初步能力，与 2D 流体代码差距大 |
| **瞬态事件** | ELM、破裂在 SOL 中的完整时空演化 | 多尺度耦合（ms 湍流 + μs 动理学）未解决 |
| **壁腐蚀与杂质源** | 壁侵蚀的动态演化与等离子体耦合 | 静态壁假设为主，动态耦合处于起步 |
| **钨杂质输运** | W 的多电荷态 + 输运在 SOL 中的完整模拟 | 捆绑模型精度有限，全 Collisional-Radiative 计算昂贵 |
| **横向量化输运系数** | 反常横向输运系数 $D_\perp, \chi_\perp$ 非第一性原理 | 贝叶斯推断（UEDGE 2025）正在推进 |
| **高第一壁负载** | 主腔壁热/粒子负载的可靠预测 | 需要全真空室网格（SOLEDGE3X 推进中） |
| **氚保留与燃料循环** | SOL 模拟与燃料循环的整体耦合 | 跨尺度-跨物理场集成是重大挑战 |

### 8.3 对你的 PINN-SOL 工作的定位

```
解析模型 (两点模型)       数值模拟 (SOLPS/UEDGE)       代理模型 (PINN)
    │                          │                          │
 快速、透明、                高保真、高成本、              精度与速度的
 但简化过多                 无法实时                        平衡点
```

你的 PINN-SOL 工作处于上述谱系的右端，与传统方法的关系是：
- **物理基础**继承自两点模型和 1D 输运方程
- **数值解验证**需要参考 FD 求解器（已实现）
- **局限性**需要对照传统方法理解（如 1D 近似不能处理湍流、2D 几何）
- **独特价值**在于提供传统方法无法做到的 ultra-fast 推理和逆问题求解

---

## 九、参考文献

### 经典专著与综述

1. Stangeby, P. C. (2000). *The Plasma Boundary of Magnetic Fusion Devices*. IoP Publishing.
2. Pitcher, C. S., & Stangeby, P. C. (1997). Experimental divertor physics. *Plasma Phys. Control. Fusion*, **39**, 779–930.
3. Lipschultz, B., et al. (2007). Plasma-surface interaction, scrape-off layer and divertor physics: implications for ITER. *Nucl. Fusion*, **47**, 1189.

### SOLPS/SOLPS-ITER

4. Wiesen, S., et al. (2015). The SOLPS-ITER code package. *J. Nucl. Mater.*, **463**, 480.
5. Bonnin, X., et al. (2016). SOLPS-ITER code package. *Plasma Fusion Res.*, **11**, 1403102.
6. Rozhansky, V., et al. (2009). B2.5 drift formulation. *Nucl. Fusion*, **49**, 025007.
7. Reiter, D., et al. (2005). The EIRENE code. *Fusion Sci. Technol.*, **47**, 172.
8. Kaveeva, E., et al. (2020). SOLPS-ITER with drifts for ITER. *Nucl. Fusion*, **60**, 046020.
9. Wiesen, S., Bonnin, X., & Pitts, R. A. (2025). Conclusive benchmark of SOLPS-ITER against SOLPS4.3. *Nucl. Fusion*, **65**, 056027.

### EDGE2D-EIRENE

10. Guillemaut, C., et al. (2014). EDGE2D-EIRENE simulations of JET divertor detachment. *Nucl. Fusion*.
11. Wiesen, S., et al. (2017). Modelling of plasma-edge and plasma–wall interaction at JET. *Phys. Scr.*.

### UEDGE

12. Rensink, M. E., et al. (1996). Models and applications of the UEDGE code. *Contrib. Plasma Phys.*, **36**, 157.
13. Rognlien, T. D., et al. (1992). UEDGE edge plasma transport code.

### SOLEDGE

14. Rivals, N., et al. (2025). SOLEDGE3X full vessel ITER simulations. *Nucl. Fusion*, **65**, 026038.

### 湍流代码

15. Ricci, P., et al. (2012). GBS: Global Braginskii Solver. *Plasma Phys. Control. Fusion*, **54**, 124047.
16. Tamain, P., et al. (2016). TOKAM3X: 3D edge turbulence code. *J. Comput. Phys.*, **321**, 606.
17. Dudson, B. D., et al. (2009). BOUT++: A framework for parallel plasma fluid simulations. *Comput. Phys. Commun.*, **180**, 1467.

### 验证与基准

18. Riva, F., et al. (2016). Blob dynamics in TORPEX: a multi-code validation. *Plasma Phys. Control. Fusion*, **58**, 044005.
19. Moscheni, M., et al. (2022). Cross-code comparison for DTT. *Nucl. Fusion*, **62**, 056004.
20. Eich, T., et al. (2013). Scaling of the tokamak near SOL heat flux width. *Nucl. Fusion*, **53**, 093031.

### 1D 模型

21. Derks, G. L., et al. (2023–2024). DIV1D benchmark against SOLPS-ITER. *NF/PPCF*.

### 杂质与脱靶

22. Ye, H., et al. (2024). Impurity radiation rollover and divertor detachment. *Phys. Scr.*.
23. Yang, et al. (2024). General criterion for divertor detachment control. *Nucl. Fusion*.
24. Park, J.-S., et al. (2020). ITER divertor performance during early operation. *Nucl. Fusion*, **61**, 016021.

### ITER 与反应堆应用

25. Pitts, R. A., et al. (2019). Physics basis for the ITER divertor. *Nucl. Fusion*, **59**, 112001.
