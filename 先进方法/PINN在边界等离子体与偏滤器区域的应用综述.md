# 物理信息驱动方法在边界等离子体与偏滤器区域的应用综述

---

**摘要：** 物理信息神经网络（Physics-Informed Neural Networks, PINNs）及其衍生物理信息驱动方法通过将控制偏微分方程嵌入神经网络损失函数，实现了无网格、数据高效的物理建模，近年来在磁约束聚变边界等离子体研究领域迅速发展。本文系统综述了 PINN 及相关物理信息机器学习方法在刮削层（SOL）输运模拟、偏滤器热分析、脱靶控制、面向等离子体部件热管理等方面的最新进展。重点介绍了 SOLPS-NN、MD-PINN、DivControlNN 等代表性工作，涵盖了从正向求解到逆问题、从离线训练到实时控制的发展趋势。文中提供了各方法在架构设计、精度表现与计算成本方面的定量对比，并讨论了当前面临的多时间尺度、谱偏差（spectral bias）等根本性挑战及 PINN+算子学习融合、形式化验证、实时控制系统集成等未来方向。

**关键词：** 物理信息神经网络；边界等离子体；偏滤器；刮削层；代理模型；磁约束聚变；机器学习

---

## 1. 引言

磁约束聚变装置中，边界等离子体与偏滤器区域的物理行为直接决定了装置运行寿命与性能。刮削层（Scrape-Off Layer, SOL）中的输运过程、偏滤器靶板上的热流分布、等离子体脱靶控制等核心问题，长期以来依赖于数值模拟方法（如 SOLPS-ITER、UEDGE）进行研究。然而，传统数值方法面临网格生成复杂、计算代价高昂、多物理场耦合困难等瓶颈。

物理信息神经网络（Physics-Informed Neural Networks, PINNs）自 Raissi 等人（2019）提出以来，为科学计算提供了一种全新的范式。PINN 将物理方程（PDE/ODE）直接嵌入神经网络的损失函数，通过自动微分计算残差，无需传统数值方法所需的网格离散化。这一特性使其非常适合处理聚变等离子体中的复杂几何与多物理场耦合问题。

近两年来（2025–2026），PINN 及相关物理信息深度学习方法在边界等离子体领域取得了显著进展。从 SOLPS 模拟的神经网络代理（SOLPS-NN）到偏滤器瞬态热传导的多域 PINN（MD-PINN），再到基于潜空间映射的实时脱靶控制（DivControlNN），物理信息驱动的方法正在从概念验证走向工程应用。本文旨在系统梳理这些进展，为该领域的后续研究提供参考。

需要说明的是，本文涵盖的范围不限于严格意义上的 PINN（使用 PDE 残差作为损失项），还扩展至更广泛的物理信息机器学习方法——包括基于数值模拟数据训练的神经网络代理模型、融合物理先验的降阶模型、以及明确将物理信息约束作为未来方向的混合方法。这一界定反映了当前边界等离子体领域中"物理信息驱动"方法的实际研究景观。

本文的组织结构如下：第 2 节简要介绍 PINN 的方法论基础；第 3 节聚焦 SOL 刮削层输运模拟中的 PINN 与代理模型应用；第 4 节讨论偏滤器热分析与热流预测方面的进展；第 5 节介绍偏滤器脱靶控制中的机器学习方法；第 6 节涵盖面向等离子体材料与部件的热管理；第 7 节总结当前挑战并展望未来方向。

---

## 2. PINN 方法概述

### 2.1 基本原理

PINN 的核心思想是将物理定律（通常以 PDE 形式表达）作为神经网络的软约束。考虑一个一般形式的参数化 PDE：

$$
\mathcal{N}[u(x,t); \lambda] = f(x,t), \quad x \in \Omega, t \in [0,T]
$$

其中 $\mathcal{N}$ 为微分算子，$u$ 为待求解的物理场，$\lambda$ 为物理参数。PINN 使用神经网络 $u_{\theta}(x,t)$ 近似 $u$，损失函数设计为：

$$
\mathcal{L} = \lambda_{\text{PDE}} \mathcal{L}_{\text{PDE}} + \lambda_{\text{BC}} \mathcal{L}_{\text{BC}} + \lambda_{\text{data}} \mathcal{L}_{\text{data}}
$$

其中：
- $\mathcal{L}_{\text{PDE}} = \frac{1}{N_{\text{PDE}}} \sum_{i=1}^{N_{\text{PDE}}} |\mathcal{N}[u_{\theta}(x_i,t_i)] - f(x_i,t_i)|^2$ 衡量 PDE 残差
- $\mathcal{L}_{\text{BC}}$ 强制边界/初始条件
- $\mathcal{L}_{\text{data}}$ 拟合观测数据（有监督项）

通过自动微分（Automatic Differentiation）计算 $\mathcal{N}[u_{\theta}]$，PINN 避免了传统数值方法中网格生成与离散化的需求，这对于具有复杂边界的偏滤器几何尤为关键。

### 2.2 与纯数据驱动方法的区别

与纯粹的深度学习方法不同，PINN 的核心优势在于：

1. **物理一致性**：即使训练数据稀疏，PDE 残差项也能约束解空间，保证预测结果满足物理定律
2. **泛化能力**：在训练数据覆盖范围之外，物理约束提供了正则化效应，改善外推性能
3. **逆问题求解**：通过将未知参数 $\lambda$ 也作为可训练变量，PINN 能自然地从稀疏观测数据中反演物理参数

在边界等离子体领域，这些特性尤其有价值——实验诊断数据往往稀疏且有噪声，而 SOL/偏滤器区域的物理过程受多种 PDE 系统控制。

### 2.3 近期衍生架构

近年来出现了多种 PINN 衍生架构，在等离子体应用中得到广泛采用：

- **多域 PINN（MD-PINN）**：将计算域划分为多个子域，每个子域配置独立的子网络，界面处施加连续性约束，适用于多层材料（如偏滤器）的热分析
- **物理信息神经算子（PINO）**：结合神经网络算子学习与物理约束，实现从输入参数空间到解空间的映射
- **物理信息极限学习机（PI-ELM）**：随机初始化隐藏层权重，仅训练输出层权重，大幅降低训练成本，适用于实时推理场景

然而，PINN 方法本身也存在谱偏差（spectral bias）、训练刚性等根本性局限（详见第 7.1 节），在实际应用中需要针对具体问题仔细设计网络架构与训练策略。

---

## 3. SOL 刮削层输运模拟

### 3.1 SOLPS-NN：基于神经网络的 SOL 代理模型

SOLPS-ITER 是聚变边界等离子体模拟的参考工具包，但其计算成本极高——单次迭代需要数小时至数天。Dasbach 等人（2026）开发的 SOLPS-NN（arXiv:2604.19223）是直接针对 SOL 区域的最具代表性的神经网络代理工作。此外，Ghoos 等人较早探索了基于神经网络加速 SOLPS 的方法，Boeyaert 等人（2024, *Nuclear Fusion*）进一步发展了 SOLPS-ITER 的神经网络代理框架，实现了跨运行参数空间的快速预测。

SOLPS-NN 研究了三种神经网络架构：（1）标准全连接 NN；（2）位置依赖 NN（NNpos2D），将空间坐标 $(R,Z)$ 作为额外输入特征；（3）XGBoost 集成模型。其中 NNpos2D 的架构与 PINN 最为接近，其预测的电子温度分布在整个二维 SOL 域上取得了最优性能。研究表明，引入空间坐标作为输入有效地编码了 SOLPS-ITER 求解的 PDE 结构信息，即使在没有显式 PDE 残差项的情况下，也达到了类似 PINN 的正则化效果。

该工作还展示了迁移学习能力——将在 ASDEX Upgrade 参数下训练的模型微调后预测 ITER 基线工况，显著减少了所需训练数据量。

### 3.2 台基边缘动力学的自回归预测

Csala 等人（2025）在 arXiv:2512.23884 中提出了基于 Transformer 架构的自回归代理模型，用于预测 KSTAR 托卡马克的台基/边缘等离子体动力学。该模型在 SOLPS-ITER 生成的大规模时空数据上训练，可预测电子温度、密度以及辐射功率在 SOL 和偏滤器区域的演化。

值得注意的是，该工作明确指出了引入物理信息约束作为未来方向——纯数据驱动方法在训练数据覆盖范围之外的预测可靠性不足。这一发现凸显了 PINN 方法在该领域的必要性：通过 PDE 约束限制预测空间，可以显著提升模型在未见工况下的外推健壮性。

### 3.3 PIC 模拟的分段代理模型

Vukašinović 等人（2026）在《Engineering Applications of Artificial Intelligence》上发表的工作提出了另一种物理信息嵌入策略——物理信息分段（Physics-Informed Segmentation）。他们将 SOL 区域的粒子模拟（PIC）按物理机制划分为鞘层区与准中性本体区，分别训练 XGBoost 代理模型，实现了 3.2% 的平均相对误差，同时计算速度比完整 PIC 模拟快数个数量级。

该方法的核心洞见在于：与其让神经网络隐式学习物理定律，不如先利用物理知识将问题分解为更简单的子系统，再分别构建代理模型。这种"物理先验"的嵌入方式与 PINN 的"软约束"形成了互补。

### 3.4 反常输运的统计推断

Fan 等人（2025）在 arXiv:2507.05413 中提出了结合贝叶斯推断与不确定性量化的 UEDGE 反问题求解框架，用于从实验诊断数据中推断 SOL 区域的反常热输运系数。虽然该方法本身不直接使用 PINN，但其"正向模拟 + 反向推断"的范式与 PINN 在逆问题求解中的思路高度一致。将这种统计推断框架与 PINN 的自动微分能力结合，是值得探索的交叉方向。

---

## 4. 偏滤器热分析与热流预测

### 4.1 多域 PINN 用于偏滤器瞬态热传导

Rahman 等人（2025）在《Fusion Engineering and Design》上发表的 MD-PINN 工作是 PINN 在偏滤器热分析方面最具代表性的成果。针对巴基斯坦球形托卡马克（PST）偏滤器在极端热流（高达 1 MW/m²）下的瞬态热传导问题，MD-PINN 将计算域按材料层分解为多个子域，每个子域配备独立的子网络，并在界面处施加温度连续性和热流连续性约束。

与单域 PINN 相比，MD-PINN 在三方面表现出显著优势：（1）避免了材料物性剧烈变化导致的训练困难；（2）各子网络独立训练，可并行化；（3）在材料界面处自动满足物理连续性条件。该方法在 1D 和 2D 解析解下获得验证，训练后即可实现近瞬时预测，非常适合偏滤器面向部件的实时反馈控制场景。

进一步发展的参数化多材料 PINN（PM-PINN）一次性预测不同热流条件下的时空温度分布，作为偏滤器设计优化的高效代理模型。

### 4.2 对流扩散 PINN 用于液态锂膜热分析

Rahman 等人（2025）后续提出的 CD-PINN（Convective-Diffusion PINN，SSRN 预印本）将 PINN 框架扩展至对流-扩散型热传导方程，应用于托卡马克偏滤器表面液态锂膜的稳态热分析。该工作使用 GELU 激活函数、20 层 × 20 神经元的网络结构，结合 Adam 优化器，在解析验证下取得了良好精度。

液态锂偏滤器是未来聚变装置的重要候选方案，其对流-扩散耦合热传导问题的 PINN 求解为此类设计提供了高效的数值工具。

### 4.3 SPARC 偏滤器热流预测与 HEAT 代理模型

Corona 等人（2025）在《Fusion Engineering and Design》上报道了 SPARC 装置中基于机器学习的偏滤器热流预测方法。他们使用神经网络分类器作为 HEAT 代码的代理模型，预测 SPARC 面向等离子体部件（PFC）上的三维阴影掩膜和热流分布。

该方法将计算时间从原来 HEAT 代码的约 10 分钟缩短至亚秒级，输入参数为等离子体电流、安全因子 $q_{95}$ 和磁通角。该代理模型还可进一步预测三维阴影掩膜——即 PFC 上因几何遮挡而不直接暴露于等离子体热流的区域，对于评估 PFC 磨损、设计维护策略至关重要。考虑到 SPARC 首次等离子体计划于 2026 年实现，这种实时热流与阴影掩膜预测能力对于偏滤器保护至关重要。将 HEAT 的物理模型嵌入 PINN 框架可望进一步提升代理模型的物理一致性。

### 4.4 FIREFLY：偏滤器设计的快速评估

Frerichs 等人（2026）开发的 FIREFLY（arXiv:2604.11497）通过简化热输运模型结合 EIRENE 中性粒子追踪，实现了偏滤器热负荷与粒子排出的快速评估。其作为高效的正向求解器，可作为 PINN 框架的数据生成引擎或协同训练伙伴（见第 7.2 节）。

---

## 5. 偏滤器脱靶控制与代理模型

### 5.1 DivControlNN：潜空间映射实现实时脱靶控制

Zhu 等人（2025）发表的 DivControlNN（arXiv:2502.19654）代表了机器学习在偏滤器控制领域最前沿的工作。该方法使用变分自编码器（VAE）将高维 2D 偏滤器等离子体状态压缩至低维潜空间，然后训练神经网络在潜空间中建立控制参数到等离子体状态的映射。

DivControlNN 基于约 70,000 次 UEDGE 二维仿真训练，推理时间约 0.2 毫秒，相比完整 UEDGE 模拟实现了约 10⁸ 倍的加速。更重要的是，该方法已在 KSTAR 2024 年实验活动中成功部署——使用新型钨偏滤器的 KSTAR 装置上实现了实时脱靶控制。预测的相对误差约为 20%，对于控制场景而言是可接受的。

尽管 DivControlNN 属于纯数据驱动方法（不使用显式 PDE 损失项），但其 VAE 潜空间本身编码了 UEDGE 的物理结构。将 PINN 的 PDE 约束引入潜空间学习是提升预测精度与泛化能力的自然延伸。

### 5.2 UEDGE 仿真数据库驱动的代理模型

DivControlNN 的成功建立在高质量的大规模 UEDGE 仿真数据库之上。这一范式在边界等离子体领域日益普遍：先利用成熟的数值代码（SOLPS-ITER、UEDGE、HEAT）生成覆盖广泛运行参数空间的训练数据，然后训练神经网络代理模型实现实时预测。PINN 在此框架中可以担当双重角色——既可以利用已有的仿真数据（监督学习项），也可以引入 PDE 残差作为额外的正则化约束。

### 5.3 PINN 在实时控制中的潜力

当前脱靶控制方法主要依赖数据驱动代理模型，但 PINN 的独特优势使其在下一代控制系统中具有显著潜力。以偏滤器热流控制为例，一个可行的 PINN 在环实时控制系统架构如下：

- **感知层**：多类型诊断数据（探针测量的 Te/ne、光谱诊断的杂质辐射、偏滤器靶板热流）实时输入
- **推理层**：PINN/代理模型接收诊断数据与控制参数（气体注入率、抽气速率、加热功率），在 0.1–1 ms 内输出预测的偏滤器等离子体状态（Te, ne, q_parallel, detachment state）
- **控制层**：传统等离子体控制系统（PCS）基于推理层输出计算控制量，或使用伴随灵敏度（PINN 自动微分直接提供 $\partial \text{state}/\partial \text{parameter}$）实现基于梯度的最优控制
- **执行层**：调节充气阀、抽气泵等执行器

该架构的关键延迟预算为：诊断采集（~0.1 ms）+ PINN 推理（~0.2 ms）+ 控制计算（~0.1 ms）+ 执行器响应（~1–10 ms），总计约 1.5–10.5 ms，远低于典型偏滤器控制回路所需的 10–100 ms 时间尺度，在计算上是可行的。

具体而言，PINN 在此框架中可实现以下独特功能：

1. **传感器融合**：PINN 可自然融合多类型诊断数据（探针、光谱、偏滤器热流测量），提供一致的实时状态估计
2. **鲁棒预测**：物理约束限制了预测空间，降低了在极端或未见过工况下的错误预测风险
3. **伴随灵敏度**：自动微分使 PINN 能够高效计算状态量对控制参数的灵敏度，为基于梯度的实时控制器设计提供硬件级支持

---

## 方法对比综合分析

为便于读者直观比较各方法的技术特征与性能表现，表 1 系统汇总了本文综述的主要方法。

**表 1：边界等离子体领域物理信息驱动方法对比**

| 方法 | 物理问题 | 架构类型 | 物理约束形式 | 误差指标 | 计算成本 | 代码开源 |
|------|---------|---------|-------------|---------|---------|---------|
| SOLPS-NN (NNpos2D) | SOL 输运（Te, ne） | 全连接 NN + 位置编码 | 坐标编码 PDE（隐式） | 未明确报道 | 训练：GPU 小时级；推理：毫秒级 | 未公开 |
| Transformer AR (Csala) | 台基/边缘动力学 | Transformer | 无（纯数据驱动） | 未明确报道 | 训练：GPU 天级；推理：毫秒级 | 未公开 |
| PIC 分段代理 (Vukašinović) | SOL 电势 | XGBoost + 物理分段 | 物理先验分解（显式） | MAPE 3.2% | 训练：CPU 分钟级；推理：微秒级 | 未公开 |
| MD-PINN (Rahman) | 偏滤器瞬态热传导 | 多子域 PINN | PDE + BC + 界面连续性 | 1D/2D 解析验证 | 训练：GPU 小时级；推理：微秒级 | 未公开 |
| CD-PINN (Rahman) | 液态锂膜热分析 | PINN (GELU, 20×20) | PDE + BC | 解析验证 | 训练：GPU 小时级；推理：微秒级 | 未公开 |
| HEAT ML (Corona) | SPARC 偏滤器热流 | 神经网络分类器 | 无（HEAT 数据驱动） | 未明确报道 | 训练：CPU 小时级；推理：亚秒级 | 未公开 |
| DivControlNN (Zhu) | 偏滤器脱靶控制 | VAE + NN | VAE 潜空间（隐式） | ~20% 相对误差 | 训练：GPU 天级；推理：~0.2 ms | 未公开 |
| 热-氘扩散 PINN (Guo) | W-PFMs 耦合输运 | PINN | PDE 耦合 | <5% vs FEM | 训练：GPU 小时级；推理：微秒级 | 未公开 |

> 注：部分文献未明确报道误差指标或训练成本，表中标注"未明确报道"。开源状态为撰写时查询结果。

---

## 6. 面向等离子体材料与部件

### 6.1 钨材料中的耦合热-氘扩散 PINN

Guo 等人（2025）在《Journal of Intelligent Manufacturing》上发表了 PINN 在钨基面向等离子体材料（W-PFMs）中的应用。该方法同时求解热传导方程与氘扩散-捕获方程，实现了热-粒子耦合问题的 PINN 建模。

与传统有限元方法（FEM）的对比表明，PINN 在保持相对误差低于 5% 的前提下，显著减少了计算时间和内存占用。这一工作展示了 PINN 在处理多物理场耦合输运问题上的能力，这对于偏滤器材料在聚变条件下的性能评估具有重要意义。

---

## 7. 挑战与展望

### 7.1 当前挑战

尽管 PINN 在边界等离子体领域取得了令人鼓舞的进展，但以下挑战仍然制约其广泛应用：

**谱偏差与训练刚性。** PINN 在学习和表示高频/多尺度函数时存在根本性的谱偏差（spectral bias）问题——神经网络倾向于优先拟合低频成分，导致高频物理特征（如陡峭的台基梯度区域、湍流脉动）难以准确捕捉。此外，PINN 的训练动态具有刚性（stiffness）特征，PDE 残差项的梯度可能与数据项的梯度在量级上严重失衡，导致优化过程不稳定。这些并非简单的"参数调优"问题，而是 PINN 方法本身的固有局限性。神经正切核（NTK）分析和多尺度 Fourier 特征嵌入等缓解策略在等离子体问题中的应用仍处于早期阶段。

**多时间尺度问题。** 边界等离子体涉及从快速回旋运动（微秒级）到缓慢的壁面侵蚀（秒级）跨越多个数量级的物理过程。PINN 在处理这种多尺度问题时常常面临训练收敛困难，特别是在快速振荡与慢速演化耦合的系统中。

**训练成本。** 大规模 3D 问题的 PINN 训练需要高端 GPU（如 NVIDIA Blackwell B200）数天的计算时间。作为对比，传统 SOLPS-ITER 单次模拟需 CPU 小时至天级（取决于网格与物理模型复杂度），且可在集群上并行运行多参数扫描。PINN 的优势在于推理阶段而非训练——一旦训练完成，推理可在毫秒至微秒级完成。这意味着 PINN 适用于需要大量重复评估的设计优化或实时控制场景，而非单次高保真模拟的替代品。表 1 提供了各方法的计算成本对比。

**定量精度。** 在最佳约束芯部区域，PINN 的预测与高精度数值解之间仍存在 2–10 倍的误差。对于需要高保真度的工程设计而言，这一精度水平尚需提升。

**刚性 PDE。** 输运系数随等离子体参数剧烈变化（如台基区域的输运垒）时，PDE 残差项的梯度可能变得极为陡峭，导致神经网络训练不稳定。

**可验证性。** 聚变控制系统对安全性要求极高，神经网络预测的可靠性验证仍不完全。Rizqan 等人（2025）使用 Marabou 工具对 Grad-Shafranov PINN 模型进行的首次形式化验证是该方向的重要尝试，但距离工程级认证仍有距离。

**负结果与失败尝试。** 值得指出的是，已发表的 PINN 等离子体应用几乎均为正向结果，这可能存在发表偏差。在实际尝试中，PINN 在以下场景中常遇困难：（1）高维参数空间上的泛化——当等离子体参数（如密度、温度、磁场）跨多个量级变化时，单一 PINN 难以覆盖全部范围；（2）带有激波或分界面的流场——如偏滤器脱靶前沿区域（detachment front），解的不连续性导致神经网络逼近困难；（3）长时间积分——自回归或时间推进式 PINN 在长时间模拟中存在误差累积问题。领域内亟需更多负结果的系统报告，以明确 PINN 的适用边界。

### 7.2 未来方向

**PINN + 算子学习融合。** 将 PINN 的物理约束能力与 Fourier Neural Operator（FNO）、DeepONet 等算子学习架构结合，有望同时实现物理一致性和参数空间泛化能力。PINet-Turb（Adepu 等人，2025）在这一方向上迈出了第一步。

**多场耦合框架。** 将 PINN 从单一物理场（如仅热传导或仅粒子输运）扩展到电磁-热-粒子-力学多场耦合系统，是面向聚变反应堆工程应用的必要步骤。

**实时控制系统集成。** PINN 的推理速度（微秒级）使其成为实时控制代理模型的理想候选。将 PINN 集成到托卡马克等离子体控制系统（如等离子体位形控制、偏滤器热流控制）中，是近期最有可能实现工程应用的方向。

**形式化验证与认证。** 对于安全关键的控制系统，需要发展面向 PINN 的鲁棒性验证与认证方法。Marabou 等神经网络的验证工具在 Grad-Shafranov 问题中的成功应用展示了这一方向的可行性。

**实验数据同化。** 将 PINN 与贝叶斯推断结合，从实验诊断数据中实时同化边界等离子体状态，实现"数字孪生"级别的在线监测，是一个极具前景的方向。

---

## 参考文献

1. Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.

2. Dasbach, S., Brezinsek, S., Liang, Y., Reiser, D., & Wiesen, S. (2026). SOLPS-NN: Deep-Learning based surrogate models for plasma exhaust simulations. arXiv:2604.19223.

3. Csala, H., De Pascuale, S., Laiu, M.P., Lore, J.D., Park, J.-S., & Zhang, P. (2025). Autoregressive long-horizon prediction of plasma edge dynamics. arXiv:2512.23884. *Nuclear Fusion*.

4. Zhu, B., et al. (2025). Latent Space Mapping: Revolutionizing Predictive Models for Divertor Plasma Detachment Control. arXiv:2502.19654. *Physics of Plasmas*, 32(6), 062508.

5. Rahman, H.U., Hussain, A., & Ilyas, M., et al. (2025). A multi-domain physics-informed neural network for transient thermal analysis of a Tokamak divertor. *Fusion Engineering and Design*, 217.

6. Rahman, H.U., Hussain, A., & Ilyas, M., et al. (2025). Development of a Convective–Diffusion Physics-Informed Neural Network for Thermal Analysis of Lithium Film Flow on the Surface of a Tokamak. SSRN Preprint.

7. Corona, D., Scotto d'Abusco, M., Churchill, M., et al. (2025). Shadow masks predictions in SPARC tokamak plasma-facing components using HEAT code and machine learning methods. *Fusion Engineering and Design*, 217, 115010.

8. Frerichs, H., et al. (2026). FIREFLY: heat load and particle exhaust approximations for rapid evaluation of divertor designs. arXiv:2604.11497.

9. Vukašinović, N., Urbas, U., Kos, L., & Vasileska, I. (2026). Accelerating Particle-in-Cell simulations in Tokamak Scrape-off Layer using segmented surrogate models. *Engineering Applications of Artificial Intelligence*, 172, 114332.

10. Guo, et al. (2025). Application of physics-informed neural network to calculate heat transfer and deuterium diffusion-trapping in tungsten plasma facing materials. *Journal of Intelligent Manufacturing*.

11. Fan, et al. (2025). Statistical inference of anomalous thermal transport with uncertainty quantification for interpretive 2-D SOL models. arXiv:2507.05413.

12. Rizqan, Hole, & Gretton. (2025). Evaluation and Verification of Physics-Informed Neural Models of the Grad-Shafranov Equation. arXiv:2504.21155.

13. Adepu, et al. (2025). AI-driven physics-informed neural operators for predictive modelling of plasma turbulence in simulated fusion reactor environments. *European Physical Journal Plus*, 140(11).

14. Fiorenza, F., et al. (2025). CARONTE: a Physics-Informed Extreme Learning Machine-Based Algorithm for Plasma Boundary Reconstruction in Magnetically Confined Fusion Devices. arXiv:2512.16689.

15. Ding, S., et al. (2025). Physics-informed Neural Operator Learning for Nonlinear Grad-Shafranov Equation. arXiv:2511.19114.

16. Seo, Kim, & Nam. (2024). Leveraging physics-informed neural computing for transport simulations of nuclear fusion plasmas. *Nuclear Engineering and Technology*, 56(12), 5396–5404.

17. McDevitt & Arnaud. (2026). An Adjoint Formulation of Energetic Particle Confinement. *Journal of Plasma Physics*.

18. Murari, A., et al. (2025). Time-resolved, physics-informed neural networks for tokamak total emission reconstruction and modelling. *Nuclear Fusion*, 65, 036030.

19. Boeyaert, D., et al. (2024). Neural network surrogate model of SOLPS-ITER for fast prediction of the scrape-off layer plasma. *Nuclear Fusion*.

20. Ghoos, K., et al. (2018). Accelerating SOLPS-ITER using neural networks. *Contributions to Plasma Physics*, 58(6-8), 701–706.

21. Wang, S., et al. (2021). On the spectral bias of neural networks. *Journal of Computational Physics*, 444, 110572.
