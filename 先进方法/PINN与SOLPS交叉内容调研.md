# PINN 与 SOLPS 交叉内容调研

更新时间：2026-05-18  
调研目标：梳理 `Physics-Informed Neural Networks (PINNs)` 与 `SOLPS / SOLPS-ITER` 的直接交叉研究现状，重点关注 2024-2026 年最新成果，并判断该方向是“已有成熟文献”还是“存在明显空白但有强机会”。

## 1. 结论先行

截至 **2026 年 5 月 18 日**，我在公开可检索文献中**没有找到一篇已经明确、直接以 “PINN + SOLPS-ITER” 为核心方法组合的代表性正式论文**。更准确地说：

- **直接交叉**：几乎没有看到“用 PINN 直接替代 SOLPS 方程求解器”或“把 SOLPS 输出与 PINN 联合训练并形成主方法”的成熟论文；
- **最接近的方向**：已经出现了多篇**基于 SOLPS-ITER 数据训练深度学习 surrogate** 的工作；
- **PINN 侧邻近方向**：已经出现了多篇面向 **tokamak divertor、热流、边界层、偏滤器热分析** 的 PINN 论文；
- **研究空白的核心**：当前主流是“`SOLPS + 数据驱动 surrogate`”而不是“`SOLPS + PINN`”。

因此，这个方向的研究判断可以概括为一句话：

> `PINN + SOLPS` 目前不是一个已经拥挤的成熟赛道，而是一个“近邻工作很多、直接命中文献很少、非常适合作为新课题切入”的交叉空白点。

## 2. 什么是 SOLPS，为什么它会和 PINN 发生交叉

`SOLPS-ITER` 是磁约束聚变边界等离子体建模中的核心工具之一，主要用于模拟：

- scrape-off layer（SOL）；
- divertor（偏滤器）；
- 边界等离子体与中性粒子输运；
- plasma-wall interaction 的部分背景场。

它的痛点也非常明确：

- 计算昂贵；
- 参数扫描困难；
- 初始化和收敛敏感；
- 很难直接用于实时控制或大规模优化。

这正好对应了 PINN / 科学机器学习擅长解决的问题：

- 构建快速 surrogate；
- 处理少量数据 + 物理约束；
- 做反问题、参数识别和边界重建；
- 为实时控制提供近实时预测。

所以从研究逻辑上说，`PINN + SOLPS` 非常自然。但从公开文献现状看，社区目前优先走的是：

1. 先用普通神经网络做 SOLPS surrogate；
2. 再考虑加入 physics-informed 约束；
3. 还没有大规模形成“PINN 直接对接 SOLPS”的主流范式。

## 3. 直接相关文献：SOLPS surrogate 已经起来了，但大多不是 PINN

### 3.1 2023：SOLPS surrogate 的早期系统探索

**Dasbach, Wiesen (2023)**  
*Towards fast surrogate models for interpolation of tokamak edge plasmas*  
期刊：*Nuclear Materials and Energy* 34, 101396 (2023)  
DOI：<https://doi.org/10.1016/j.nme.2023.101396>

这篇文章是当前 `SOLPS + 神经网络 surrogate` 路线的重要起点之一。文中明确指出：

- 使用 **SOLPS-ITER** 构建 tokamak exhaust 数据集；
- 训练 surrogate 预测 **divertor temperature profiles**；
- 神经网络优于 gradient boosted trees；
- 最优模型的中位绝对误差达到 `1.6 eV`。

从摘要可见，该工作使用的是常规神经网络，而**不是 PINN**。但它直接说明了两点：

1. 社区已经明确把 `SOLPS-ITER` 看作高价值 surrogate 数据源；
2. `PINN + SOLPS` 的数据与任务入口其实已经存在，只是方法上还没切换过去。

来源：ScienceDirect 页面摘要与 highlights  
链接：<https://www.sciencedirect.com/science/article/pii/S2352179123000352>

### 3.2 2025：基于 UEDGE 的 detachment control surrogate，说明“边界等离子体实时代理”已经进入控制层

**Zhu et al. (2025)**  
*Latent Space Mapping: Revolutionizing Predictive Models for Divertor Plasma Detachment Control*  
arXiv：<https://arxiv.org/abs/2502.19654>

这篇文章不是 SOLPS，而是 **UEDGE**。但它非常关键，因为它描述了与 SOLPS 几乎同类的边界/偏滤器物理任务：

- 用超过 `70,000` 组 **2D UEDGE** 模拟训练 surrogate；
- 面向 **boundary and divertor plasma behavior**；
- 实现了准实时预测；
- 在 KSTAR 2024 实验中用于 detachment control 原型系统。

这说明在边界等离子体建模领域，社区已经明确接受这样一种路线：

> 高保真边界代码离线出数据，机器学习模型在线做代理和控制。

这条路线与 `PINN + SOLPS` 高度相容，只是目前公开实现还主要是 latent-space / 普通深度网络，而不是 PINN。

### 3.3 2025：时序 surrogate 开始直接吃 SOLPS-ITER 动态场

**Csala et al. (2025)**  
*Autoregressive long-horizon prediction of plasma edge dynamics*  
arXiv：<https://arxiv.org/abs/2512.23884>

这是非常值得关注的一篇近作。文章明确写到：

- 高保真边界流体/中性粒子代码如 **SOLPS-ITER** 很准确，但计算代价高；
- 作者使用 **SOLPS-ITER spatiotemporal data**；
- 构建 transformer-based autoregressive surrogate；
- 预测 `electron temperature`、`electron density`、`radiated power` 的二维时空场；
- 速度远快于 SOLPS-ITER；
- 作者在摘要中直接指出未来需要 **physics-informed constraints**。

这一点非常重要，因为它意味着：

- `SOLPS surrogate` 已经从静态参数插值走向动态时空 rollout；
- 社区已经自己意识到纯数据驱动 surrogate 的物理泛化问题；
- **physics-informed constraints** 很可能就是 PINN 类方法进入该赛道的接口。

### 3.4 2026：SOLPS-NN 是目前最接近 “PINN + SOLPS” 的热点，但仍不是 PINN

**Dasbach et al. (2026)**  
*Deep-Learning based surrogate models for plasma exhaust simulations -- SOLPS-NN*  
arXiv：<https://arxiv.org/abs/2604.19223>

这是截至 **2026-05-18** 检索中最重要的最新文献之一。摘要显示：

- 目标是为 **scrape-off layer** 建立快速 surrogate；
- 基于 **数千组 SOLPS-ITER simulations**；
- 比较了多种机器学习架构；
- 认为简单 fully connected neural networks 已经可用；
- 使用 reduced neutral fidelity 数据训练，再用更高 fidelity 的 ITER baseline 数据做再训练；
- 讨论了 transfer learning 和 multi-fidelity 的潜力；
- surrogate 能预测 access to detachment，并与实验趋势一致。

这篇文章的意义非常大：

- 它几乎就是 `SOLPS-NN` 这条线的代表作；
- 但它依然没有把方法表述为 PINN；
- 它把**多保真训练**问题摆到了台面上，这恰好是 PINN 非常适合切入的地方。

## 4. 最新会议信号：社区已经开始把 surrogate 嵌进 SOLPS-ITER 工作流

### 4.1 2026 PSI 会议：Inline Deep Surrogates for Accelerating SOLPS-ITER Simulations

会议页面：<https://plan.events.mpg.de/event/187/contributions/2887/>

根据会议摘要，这项 2026 年工作提出：

- 使用 **KD-tree warm start** 从已收敛 run 中找近邻初始化；
- 基于 DIII-D 数据训练两类 surrogate：
  - 一维 profile surrogate；
  - 二维 map surrogate；
- 二维 surrogate 使用 **U-Net** 和 **VAE**；
- 并尝试把 surrogate **inline 地放进 SOLPS-ITER loop**，以减少部分 **EIRENE** 调用。

虽然它仍然不是 PINN，但这件事说明研究正在向更深的方向走：

- 不再只是“离线拟合 SOLPS 输出”；
- 而是开始尝试“把 surrogate 嵌进求解流程本身”。

这正是未来 PINN 可能产生优势的位置，因为 PINN 的物理约束形式更适合嵌入到 solver-loop 或 hybrid-solver 结构中。

## 5. 邻近 PINN 文献：虽然没直接碰 SOLPS，但已经逼近边界/偏滤器问题

### 5.1 2023：W7-X 实时热流反演

**Aymerich et al. (2023)**  
*Physics Informed Neural Networks towards the real-time calculation of heat fluxes at W7-X*  
期刊：*Nuclear Materials and Energy* 34, 101401 (2023)  
DOI：<https://doi.org/10.1016/j.nme.2023.101401>

文章指出：

- 实时 heat flux estimation 对 steady-state fusion operation 很关键；
- 现有 2D 代码如 THEODOR 主要离线使用；
- 提出用 PINN 解热方程并做 divertor tile 热流估计；
- PINN 可微且可直接在 GPU 上运行。

这不是 SOLPS，也不是边界流体/中性粒子输运，但它已经说明：

- PINN 在“偏滤器相关实时物理量反演”上是可行的；
- 这为把 PINN 引向 `SOLPS 输出场 -> 反演/代理/控制` 提供了应用逻辑。

### 5.2 2025：多子域 PINN 做 tokamak divertor 瞬态热分析

**Rahman et al. (2025)**  
*A multi-domain physics-informed neural network for transient thermal analysis of a Tokamak divertor*  
期刊：*Fusion Engineering and Design* 216, 115036 (2025)  
DOI：<https://doi.org/10.1016/j.fusengdes.2025.115036>

这篇工作把 PINN 明确用到了 **tokamak divertor**，不过求解的是多层材料中的瞬态热传导问题，而不是 SOLPS 的边界等离子体输运。

它的重要性在于：

- 多子域 PINN 已经开始适配偏滤器这类复杂结构；
- 这提示未来如果要把 PINN 引入 SOLPS 相关问题，**多区域、多边界、强界面条件**会是更自然的实现方式。

### 5.3 2026：锂膜偏滤器表面热分析 PINN

**Rahman et al. (2026 preprint)**  
*Development of a Convective–Diffusion Physics-Informed Neural Network for Thermal Analysis of Lithium Film Flow on the Surface of a Tokamak Divertor*  
Research Square / Sciety 条目：<https://sciety.org/articles/activity/10.21203/rs.3.rs-9321560/v1>

这篇 2026 年预印本进一步说明，PINN 正在持续向 `tokamak divertor` 的热输运问题推进。虽然仍不涉及 SOLPS，但它强化了一个趋势：

> 偏滤器相关 PINN 研究已经成形，只是目前主要集中在热分析，而不是 SOLPS 那套边界等离子体流体-中性粒子输运方程。

## 6. 与 SOLPS 物理任务最接近、但仍未使用 PINN 的几条路线

### 6.1 边界/偏滤器动态 surrogate

- `Autoregressive long-horizon prediction of plasma edge dynamics` 使用 SOLPS-ITER 时空数据做动态预测；
- `SOLPS-NN` 使用大量 SOLPS-ITER 数据做静态或准稳态 surrogate；
- 2026 PSI 会议已经开始探索 **inline surrogate**。

这是距离 `PINN + SOLPS` 最近的主线。

### 6.2 detachment control surrogate

- `DivControlNN` 虽然基于 UEDGE，但任务是 detachment control；
- 这与 SOLPS 的典型应用问题高度一致。

这说明如果你做 `PINN + SOLPS`，最容易打动人的应用场景不是“纯数学求解”，而是：

- detachment onset prediction；
- radiative front tracking；
- divertor operational window mapping；
- control-oriented reduced model。

### 6.3 integrated modeling / proxy coupling

**Wilcox et al. (2026)**  
*Towards self-consistent integrated modeling of the tokamak pedestal, scrape-off layer, and divertor using SOLPS-ITER and EPED*  
*Nuclear Fusion* 66(3), 036005 (2026)  
DOI：<https://doi.org/10.1088/1741-4326/ae3845>

这篇文章不是 ML，但它很重要。文中明确：

- 用 **SOLPS-ITER** 约束 **EPED**；
- 还引入了 **empirical proxy function** 推断 pedestal-top / separatrix 关系；
- 目标是形成 core-edge integrated model。

这说明在 SOLPS 使用生态中，**proxy / reduced model / integrated model** 已经是现实需求，不是概念想象。PINN 若切入，完全可以作为更物理一致的 proxy。

## 7. 为什么目前还没看到明显的 “PINN + SOLPS” 论文

这是这次调研最值得讨论的部分。可能原因主要有五个。

### 7.1 SOLPS 不是一个单一、简单 PDE，而是复杂耦合代码体系

SOLPS-ITER 背后不是单个 PDE，而是：

- 边界流体输运；
- 中性粒子输运；
- 杂质、辐射、回收等复杂物理；
- B2 与 EIRENE 的耦合；
- 复杂几何和边界条件。

这比 PINN 最擅长的“中低维、结构相对明确的 PDE 系统”更难。

### 7.2 detachment / sheath / recycling 本身就很刚性

SOLPS 典型关心的问题，如 detachment、辐射前沿、target 附近强梯度，本来就是 PINN 的难点：

- 多尺度；
- 边界层；
- 剧烈非线性；
- 状态切换与 bifurcation。

所以社区自然会先选择“用普通神经网络拟合仿真输出”，而不是直接让 PINN 去硬学全套方程。

### 7.3 SOLPS 数据更适合先做 data-driven surrogate

一旦已经有大量收敛好的 SOLPS 数据，最省事的路线通常是：

- 直接监督学习；
- latent-space 压缩；
- transformer / U-Net / FCNN surrogate。

相比之下，PINN 需要重新设计方程残差、尺度归一化和损失权重，工程门槛更高。

### 7.4 社区当前优先目标是“快而能用”

从 `SOLPS-NN`、`DivControlNN`、`inline surrogate` 可以看到，当前边界等离子体社区最关心的是：

- 加速；
- 参数扫描；
- 控制；
- 工程工作流集成。

这类目标短期内更容易由数据驱动 surrogate 达成。

### 7.5 PINN 的优势还没有被放在最合适的位置上

PINN 真正的优势不一定在“全量替代 SOLPS”，而更可能在：

- 小样本高保真再训练；
- 多保真融合；
- 部分方程或部分区域物理约束；
- 反问题和边界重建；
- OOD 条件下的物理泛化。

而这些恰好是现有 `SOLPS surrogate` 论文已经开始暴露出来的短板。

## 8. 最值得做的 PINN + SOLPS 研究切入点

如果把这次调研转换成研究选题，下面几条是最有可行性的方向。

### 8.1 Multi-fidelity PINN for SOLPS-ITER

思路：

- 用大量低保真 `reduced-neutral-fidelity SOLPS-ITER` 数据；
- 配合少量高保真 full-fidelity 数据；
- 用 PINN 或 physics-informed surrogate 融合两种保真度。

为什么有价值：

- `SOLPS-NN` 已经明确指出 multi-fidelity 是未来重点；
- PINN 很适合把“少量高保真 + 物理约束”联合起来。

### 8.2 Physics-informed surrogate for detachment front prediction

目标：

- 输入上游参数、输运系数、气体 puff、几何参数；
- 输出 detachment onset、radiation front location、target `Te/ne/q_parallel`。

为什么适合 PINN：

- 这类目标变量与守恒方程、边界约束直接相关；
- 单纯监督 surrogate 往往在未见工况下失真；
- physics-informed 约束可能改善泛化。

### 8.3 Hybrid PINN restricted to subdomains

不要直接替代整套 SOLPS，而是只替代最慢或最敏感的子模块，例如：

- 目标附近强梯度区域；
- 部分中性粒子相关源项近似；
- 某些 EIRENE 调用的快速代理。

这和 2026 PSI 的 inline surrogate 思路非常接近，但可以进一步加入物理残差。

### 8.4 SOLPS-assisted inverse PINN

思路：

- 用实验诊断数据作为稀疏观测；
- 用 SOLPS 生成先验分布或初值库；
- 再用 PINN 做边界参数反演，例如回收系数、输运系数、source terms。

这类方案相比“全场 surrogate”更符合 PINN 在反问题上的长处。

### 8.5 Operator-learning + PINN 混合

如果直接 PINN 太难，可以考虑：

- neural operator / transformer 学主映射；
- PINN loss 只做局部守恒约束；
- 或用 divergence / source balance / boundary consistency 做软约束。

从当前文献趋势看，这种 hybrid 可能比纯 PINN 更现实。

## 9. 适合汇报时直接说的判断

如果你要在组会里用一句比较有力量的话概括，可以这样说：

> 截至 2026 年 5 月，SOLPS-ITER 与深度学习 surrogate 的结合已快速发展，但与 PINN 的直接融合仍基本处于空白状态；这意味着 `PINN + SOLPS` 不是跟风选题，而是一个具备明确应用需求、已有数据基础、但方法上仍待突破的前沿交叉方向。

也可以再展开成三条：

1. `SOLPS + ML surrogate` 已经是现实热点，最新代表是 `SOLPS-NN` 和 2026 年的 inline surrogate 会议工作。
2. `PINN + tokamak divertor/edge-related physics` 也已经出现一批邻近文献，但主要集中在热分析、热流反演和简化物理模型。
3. 两条线尚未真正汇合，这正构成了一个有研究价值的空白。

## 10. 参考文献与链接

1. Dasbach S, Wiesen S. *Towards fast surrogate models for interpolation of tokamak edge plasmas*. Nuclear Materials and Energy, 2023, 34:101396.  
   DOI：<https://doi.org/10.1016/j.nme.2023.101396>  
   页面：<https://www.sciencedirect.com/science/article/pii/S2352179123000352>

2. Aymerich E, Pisano F, Cannas B, et al. *Physics Informed Neural Networks towards the real-time calculation of heat fluxes at W7-X*. Nuclear Materials and Energy, 2023, 34:101401.  
   DOI：<https://doi.org/10.1016/j.nme.2023.101401>  
   页面：<https://www.sciencedirect.com/science/article/pii/S2352179123000406>

3. Zhu B, Zhao M, Xu X Q, et al. *Latent Space Mapping: Revolutionizing Predictive Models for Divertor Plasma Detachment Control*. arXiv, 2025.  
   链接：<https://arxiv.org/abs/2502.19654>

4. Rahman H U, Hussain A, Ilyas M, et al. *A multi-domain physics-informed neural network for transient thermal analysis of a Tokamak divertor*. Fusion Engineering and Design, 2025, 216:115036.  
   DOI：<https://doi.org/10.1016/j.fusengdes.2025.115036>

5. Csala H, De Pascuale S, Laiu P, et al. *Autoregressive long-horizon prediction of plasma edge dynamics*. arXiv, 2025.  
   链接：<https://arxiv.org/abs/2512.23884>

6. Dasbach S, Brezinsek S, Liang Y, Reiser D, Wiesen S. *Deep-Learning based surrogate models for plasma exhaust simulations -- SOLPS-NN*. arXiv, 2026.  
   链接：<https://arxiv.org/abs/2604.19223>

7. Wilcox R S, Canik J M, Park J M, et al. *Towards self-consistent integrated modeling of the tokamak pedestal, scrape-off layer, and divertor using SOLPS-ITER and EPED*. Nuclear Fusion, 2026, 66(3):036005.  
   DOI：<https://doi.org/10.1088/1741-4326/ae3845>  
   页面：<https://impact.ornl.gov/en/publications/towards-self-consistent-integrated-modeling-of-the-tokamak-pedest/>

8. *Inline Deep Surrogates for Accelerating SOLPS-ITER Simulations*. 27th PSI conference contribution, 2026.  
   页面：<https://plan.events.mpg.de/event/187/contributions/2887/>

9. Rahman H U, Hussain A, Ilyas M, Ahmed M, Qayyum M S. *Development of a Convective–Diffusion Physics-Informed Neural Network for Thermal Analysis of Lithium Film Flow on the Surface of a Tokamak Divertor*. Research Square / Sciety, 2026.  
   页面：<https://sciety.org/articles/activity/10.21203/rs.3.rs-9321560/v1>

## 11. 最终判断

`PINN + SOLPS` 目前更像一个**高潜力交叉空白**，而不是已经被做烂的方向。真正值得关注的不是“有没有人把 PINN 三个字写进 SOLPS 论文标题里”，而是：

- SOLPS surrogate 已经非常需要更强的物理泛化；
- 边界等离子体实时预测已经进入控制与工程应用阶段；
- 现有方法已经开始暴露 OOD 与多保真问题；
- 这些恰好都是 PINN 或 physics-informed surrogate 最可能提供增量价值的地方。

如果把这件事转成论文选题，那么最合理的题目方向不是“PINN 替代 SOLPS”，而是：

> **面向 detachment / divertor / SOL 动态预测的 multi-fidelity physics-informed surrogate for SOLPS-ITER**。
